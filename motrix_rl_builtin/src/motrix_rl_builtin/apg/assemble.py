# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Modeling-state assembly: FieldDesc-driven layout plus exact FK derivation.

The declared read set is the modeling state's field set (P2.3). For the
``floating_base`` scheme the world model predicts only the physical block —
root pose and velocities plus joint positions and velocities — located inside
the declared fields by query semantics:

- the root rows come from the tracked-body batch queries' row of the FK
  tree's root link (validated present, with the batch kinds agreeing on one
  tracked-body order);
- joint fields (``Joint*``/``BodyJoint*`` queries) map into the tree's q
  order by joint name, and every tree joint on the path to a tracked body
  must be covered by the task's joint set.

Every other declared field derives exactly from the physical block: tracked
batch fields through the differentiable :class:`KinematicTree` FK (a D-A
shared constant — the world model learns dynamics, never kinematics; the
root row reproduces the block's own root values exactly), single-link fields
as rows of a same-kind batch field, and duplicate joint fields as
name-mapped gathers of the q block. The FK parity fixture pins the
derivation against the simulator's own read views. The ``residual`` scheme
instead predicts the flat concatenation of every field and skips FK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from motrix_env_core.math.kinematics import KinematicTree
from motrix_env_core.math.quaternion import to_matrix
from motrix_env_core.sim.read import (
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    JointPositionQuery,
    JointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SimDataQuery,
)

_POSITION = "position"
_QUATERNION = "quaternion"
_LINEAR = "linear_velocity"
_ANGULAR = "angular_velocity"
_BATCH_KIND: dict[type, str] = {
    BatchLinkPositionQuery: _POSITION,
    BatchLinkQuaternionQuery: _QUATERNION,
    BatchLinkLinearVelocityQuery: _LINEAR,
    BatchLinkAngularVelocityQuery: _ANGULAR,
}
_LINK_KIND: dict[type, str] = {
    LinkPositionQuery: _POSITION,
    LinkQuaternionQuery: _QUATERNION,
    LinkLinearVelocityQuery: _LINEAR,
    LinkAngularVelocityQuery: _ANGULAR,
}
_JOINT_KIND: dict[type, str] = {
    JointPositionQuery: "position",
    BodyJointPositionQuery: "position",
    JointVelocityQuery: "velocity",
    BodyJointVelocityQuery: "velocity",
}


def _link_names(query: SimDataQuery) -> tuple[str, ...]:
    if type(query) in _LINK_KIND:
        return (query.link,)  # type: ignore[attr-defined]
    links = query.links  # type: ignore[attr-defined]
    return links if isinstance(links, tuple) else (links,)


def _body_joint_names(env: Any, body: str) -> tuple[str, ...]:
    """Joints of the scene body whose base link is ``body``.

    ``BodyJoint*`` queries carry the physics body (base link) name while
    ``env.model.bodies`` is keyed by scene-obj name; resolve through each
    body model's ``base_link_name``.
    """
    for model in env.model.bodies.values():
        if model.base_link_name == body:
            return tuple(model.joint_names)
    raise ValueError(f"No scene body with base link {body!r}")


def _joint_names(query: SimDataQuery, env: Any) -> tuple[str, ...]:
    if isinstance(query, (JointPositionQuery, JointVelocityQuery)):
        return tuple(query.joints)
    return _body_joint_names(env, query.body)  # type: ignore[attr-defined]


def matrix_to_quat_xyzw(rotation: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` to xyzw quaternions ``(..., 4)``.

    Robust branch selection over the four candidate roots, sign-normalized;
    ported from the prior attempt's ``_mat_to_quat_wxyz``.
    """
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(((m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw), qw), -1),
            torch.stack((qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx), (m21 - m12) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), qy, (m01 + m10) / (4 * qy), (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    quat_wxyz = torch.gather(candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))).squeeze(-2)
    quat_wxyz = torch.nn.functional.normalize(quat_wxyz, dim=-1)
    quat_wxyz = quat_wxyz * torch.where(quat_wxyz[..., :1] < 0, -1.0, 1.0)
    return torch.cat((quat_wxyz[..., 1:], quat_wxyz[..., :1]), -1)


@dataclass(frozen=True)
class Assembly:
    """Per-field materialization plan for one modeling view."""

    scheme: str
    field_keys: tuple[str, ...]
    field_shapes: dict[str, tuple[int, ...]]
    field_slots: dict[str, range]  # residual scheme: flat slot per field
    tree: KinematicTree | None
    joint_fields: dict[str, str]  # field -> "position" | "velocity"
    joint_gathers: dict[str, torch.Tensor]  # field -> task-order -> tree q index
    batch_fields: dict[str, str]  # field -> semantic kind (one per kind)
    tracked_links: tuple[str, ...]
    fk_rows: torch.Tensor  # tree body indices in tracked order
    root_row: int
    row_fields: dict[str, tuple[str, int]]  # field -> (batch field key, row)

    @property
    def flat_width(self) -> int:
        if self.scheme == "residual":
            return sum(slot.stop - slot.start for slot in self.field_slots.values())
        assert self.tree is not None
        return 13 + 2 * len(self.tree.joint_names)

    def fields_to_flat(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Gather the flat vector (physical block, or every field residual)."""
        if self.scheme == "residual":
            flats = []
            for key in self.field_keys:
                shape = self.field_shapes[key]
                value = state[key]
                flats.append(value.reshape(*value.shape[: -len(shape)], -1) if shape else value.unsqueeze(-1))
            return torch.cat(flats, dim=-1)
        assert self.tree is not None
        by_kind = {kind: key for key, kind in self.batch_fields.items()}
        position = state[by_kind[_POSITION]][..., self.root_row, :]
        quaternion = state[by_kind[_QUATERNION]][..., self.root_row, :]
        lin = state[by_kind[_LINEAR]][..., self.root_row, :]
        ang = state[by_kind[_ANGULAR]][..., self.root_row, :]
        num_joints = len(self.tree.joint_names)
        qpos = torch.zeros((*position.shape[:-1], num_joints), dtype=position.dtype, device=position.device)
        qvel = torch.zeros_like(qpos)
        for key, kind in self.joint_fields.items():
            block = qpos if kind == "position" else qvel
            block[..., self.joint_gathers[key]] = state[key]
        return torch.cat((position, quaternion, lin, ang, qpos, qvel), dim=-1)

    def flat_to_fields(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Materialize every declared field from the flat vector.

        Floating base: FK over the physical block produces each tracked batch
        field (the root row reproduces the block's root values exactly);
        single-link fields slice their batch counterpart, joint fields gather
        the q block by name. Differentiable throughout.
        """
        if self.scheme == "residual":
            fields: dict[str, torch.Tensor] = {}
            for key in self.field_keys:
                slot = self.field_slots[key]
                shape = self.field_shapes[key]
                chunk = flat[..., slot]
                fields[key] = chunk.reshape(*chunk.shape[:-1], *shape) if shape else chunk[..., 0]
            return fields
        assert self.tree is not None
        num_joints = len(self.tree.joint_names)
        position = flat[..., 0:3]
        quaternion = flat[..., 3:7]
        lin = flat[..., 7:10]
        ang = flat[..., 10:13]
        qpos = flat[..., 13 : 13 + num_joints]
        qvel = flat[..., 13 + num_joints : 13 + 2 * num_joints]
        p, rotations, v, w = self.tree.forward(position, to_matrix(quaternion), lin, ang, qpos, qvel, bodies=self.fk_rows.tolist())
        outputs = {
            _POSITION: p,
            _QUATERNION: matrix_to_quat_xyzw(rotations),
            _LINEAR: v,
            _ANGULAR: w,
        }
        fields = {key: outputs[kind] for key, kind in self.batch_fields.items()}
        for key, (batch_key, row) in self.row_fields.items():
            fields[key] = fields[batch_key][..., row, :]
        for key, kind in self.joint_fields.items():
            block = qpos if kind == "position" else qvel
            fields[key] = block[..., self.joint_gathers[key]]
        return fields


def build_assembly(view: Any, env: Any, tree: KinematicTree | None, scheme: str) -> Assembly:
    """Classify every modeling field by its query semantics.

    Raises:
        ValueError: a declared field's semantics cannot be served by the
            scheme (missing batch counterpart, joint outside the tree, a tree
            joint on a tracked path uncovered by the task's joint set...).
    """
    queries = {key: env.sim_data.query(key) for key in view.fields}
    field_shapes = {key: tuple(desc.shape) for key, desc in view.fields.items()}
    field_keys = tuple(view.fields)

    if scheme == "residual":
        slots: dict[str, range] = {}
        cursor = 0
        for key in field_keys:
            width = 1
            for dim in field_shapes[key]:
                width *= dim
            slots[key] = range(cursor, cursor + width)
            cursor += width
        return Assembly(
            scheme=scheme,
            field_keys=field_keys,
            field_shapes=field_shapes,
            field_slots=slots,
            tree=None,
            joint_fields={},
            joint_gathers={},
            batch_fields={},
            tracked_links=(),
            fk_rows=torch.zeros(0, dtype=torch.int64),
            root_row=0,
            row_fields={},
        )

    if tree is None:
        raise ValueError("floating_base assembly requires a KinematicTree (declare a KinematicTreeQuery)")

    batch_fields: dict[str, str] = {}
    tracked: tuple[str, ...] | None = None
    row_fields: dict[str, tuple[str, int]] = {}
    joint_fields: dict[str, str] = {}
    task_joints: set[str] = set()

    for key, query in queries.items():
        query_type = type(query)
        if query_type in _BATCH_KIND:
            kind = _BATCH_KIND[query_type]
            links = _link_names(query)
            if tracked is None:
                tracked = links
            elif links != tracked:
                raise ValueError(
                    f"Modeling field {key!r} tracks {links} but earlier batch fields track {tracked}; "
                    "floating_base requires one shared tracked-body order."
                )
            if kind in batch_fields.values():
                raise ValueError(f"Two modeling fields declare the {kind} batch kind; one per kind is required.")
            batch_fields[key] = kind
        elif query_type in _LINK_KIND:
            row_fields[key] = (_LINK_KIND[query_type], _link_names(query)[0])
        elif query_type in _JOINT_KIND:
            joint_fields[key] = _JOINT_KIND[query_type]
            task_joints.update(_joint_names(query, env))
        else:
            raise ValueError(
                f"Modeling field {key!r} ({query_type.__name__}) has no floating-base assembly rule; "
                "the residual scheme covers arbitrary field sets."
            )
    if tracked is None:
        raise ValueError("floating_base assembly requires tracked-body batch fields (position/quaternion/velocities)")

    root_link = tree.link_names[0]
    try:
        root_row = tracked.index(root_link)
    except ValueError:
        raise ValueError(f"The tracked body order {tracked} must include the FK tree root link {root_link!r}.") from None
    try:
        fk_rows = torch.tensor([tree.link_names.index(link) for link in tracked], dtype=torch.int64)
    except ValueError:
        missing = [link for link in tracked if link not in tree.link_names]
        raise ValueError(f"Tracked bodies absent from the kinematic tree (fixed links?): {missing}.") from None

    # Every tree joint on a path from the root to a tracked body must be
    # covered by the task's declared joint set, or FK is underdetermined.
    required: set[int] = set()
    for body in fk_rows.tolist():
        node = int(body)
        while node >= 0:
            joint = tree.joint_idx[node]
            if joint >= 0:
                required.add(joint)
            node = tree.parent[node]
    tree_joint_names = tree.joint_names
    missing = sorted(tree_joint_names[j] for j in required if tree_joint_names[j] not in task_joints)
    if missing:
        raise ValueError(
            f"Tree joints on tracked paths are outside the declared joint fields: {missing}; "
            "FK cannot derive the tracked bodies from the predicted block."
        )

    joint_gathers: dict[str, torch.Tensor] = {}
    gather_device = tree.axes.device
    for key in joint_fields:
        joints = _joint_names(queries[key], env)
        try:
            joint_gathers[key] = torch.tensor(
                [tree_joint_names.index(name) for name in joints], dtype=torch.int64, device=gather_device
            )
        except ValueError:
            absent = [name for name in joints if name not in tree_joint_names]
            raise ValueError(f"Joint field {key!r} names joints absent from the tree: {absent}.") from None

    # Single-link fields slice their same-kind batch counterpart.
    resolved_rows: dict[str, tuple[str, int]] = {}
    by_kind_field = {kind: key for key, kind in batch_fields.items()}
    for key, (kind, link) in row_fields.items():
        if kind not in by_kind_field:
            raise ValueError(f"Modeling field {key!r} ({kind} of one link) has no same-kind batch field to slice.")
        if link not in tracked:
            raise ValueError(f"Modeling field {key!r} reads link {link!r} outside the tracked body order {tracked}.")
        resolved_rows[key] = (by_kind_field[kind], tracked.index(link))

    return Assembly(
        scheme=scheme,
        field_keys=field_keys,
        field_shapes=field_shapes,
        field_slots={},
        tree=tree,
        joint_fields=joint_fields,
        joint_gathers=joint_gathers,
        batch_fields=batch_fields,
        tracked_links=tracked,
        fk_rows=fk_rows,
        root_row=root_row,
        row_fields=resolved_rows,
    )
