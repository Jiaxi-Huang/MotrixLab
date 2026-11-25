# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bind GenCoord to declared simulator query semantics, without task imports."""

import torch

from motrix_env_core.math import quaternion as quat
from motrix_env_core.math.kinematics import KinematicTree
from motrix_env_core.mdp.action import JointPositionActionTerm
from motrix_env_core.sim import read
from motrix_rl.transport.schema import TransitionField, wm_transition_schema
from motrix_rl_builtin.apg.dynamics import represent


def matrix_quaternion(matrix):
    """Stable differentiable rotation matrix to xyzw, choosing a well-scaled branch."""
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    square = torch.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22, 1 + m00 + m11 + m22], dim=-1)
    roots = square.clamp_min(1e-8).sqrt()
    candidates = torch.stack(
        [
            torch.stack([square[..., 0], m01 + m10, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m01 + m10, square[..., 1], m12 + m21, m02 - m20], dim=-1),
            torch.stack([m02 + m20, m12 + m21, square[..., 2], m10 - m01], dim=-1),
            torch.stack([m21 - m12, m02 - m20, m10 - m01, square[..., 3]], dim=-1),
        ],
        dim=-2,
    ) / (2 * roots[..., :, None])
    index = roots.argmax(-1)[..., None, None].expand(*roots.shape[:-1], 1, 4)
    return torch.nn.functional.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)


class GenCoordLayout:
    """One floating base, position-controlled joints and declared body queries."""

    def __init__(self, env, view, device):
        self.device = torch.device(device)
        trees = [value for value in env.model.others.values() if isinstance(value, KinematicTree)]
        if len(trees) != 1:
            raise ValueError("APG requires exactly one declared KinematicTreeQuery")
        tree = trees[0]
        self.tree = KinematicTree(
            tree.parent,
            tree.joint_idx,
            tree.axes,
            tree.off_p,
            tree.off_R,
            tree.link_names,
            tree.joint_names,
            device=device,
        )
        self.queries = {key: env.sim_data.query(key) for key in view.fields}
        self.root_keys = []
        for kind in (
            read.LinkPositionQuery,
            read.LinkQuaternionQuery,
            read.LinkLinearVelocityQuery,
            read.LinkAngularVelocityQuery,
        ):
            keys = [
                key
                for key, query in self.queries.items()
                if isinstance(query, kind) and query.link == tree.link_names[0]
            ]
            if len(keys) != 1:
                raise ValueError(f"APG requires one root {kind.__name__}")
            self.root_keys.append(keys[0])
        self.joint_keys = []
        for kind in (read.JointPositionQuery, read.JointVelocityQuery):
            keys = [key for key, query in self.queries.items() if isinstance(query, kind)]
            if len(keys) != 1:
                raise ValueError(f"APG requires one {kind.__name__}")
            self.joint_keys.append(keys[0])
        joint_names = self.queries[self.joint_keys[0]].joints
        if tuple(self.queries[self.joint_keys[1]].joints) != tuple(joint_names):
            raise ValueError("APG joint position and velocity query orders must agree")
        self.joint_count = len(joint_names)
        self.tree_order = [joint_names.index(name) for name in tree.joint_names]
        self.body_indices = {}
        for key, query in self.queries.items():
            if key in self.root_keys + self.joint_keys:
                continue
            if not isinstance(
                query,
                (
                    read.BatchLinkPositionQuery,
                    read.BatchLinkQuaternionQuery,
                    read.BatchLinkLinearVelocityQuery,
                    read.BatchLinkAngularVelocityQuery,
                ),
            ):
                raise ValueError(f"APG GenCoord cannot reconstruct declared field {key}: {type(query).__name__}")
            self.body_indices[key] = [tree.link_names.index(name) for name in query.links]
        terms = list(env.action_terms.values())
        if len(terms) != 1 or not isinstance(terms[0], JointPositionActionTerm) or terms[0].state.delay_hi:
            raise ValueError("APG GenCoord requires one undelayed joint-position action term")
        if tuple(spec.target_name for spec in env.model.actuators) != tuple(joint_names):
            raise ValueError("APG action and joint query orders must agree")
        self.control_program = view.program.fork(1, device)
        self.action_name = next(iter(env.action_terms))

    def control(self, action):
        return self.control_program.action_controls(action)[self.action_name]

    def encode(self, state):
        position, rotation, velocity, angular = (state[key] for key in self.root_keys)
        joints, joint_velocity = (state[key] for key in self.joint_keys)
        aux = torch.cat([position, rotation], dim=-1)
        return aux, represent(aux, velocity, angular, joints, joint_velocity)

    def decode(self, aux, rep):
        n = self.joint_count
        rotation = aux[..., 3:7]
        position = aux[..., :3]
        velocity = quat.rotate_vector(rotation, rep[..., 4:7])
        angular = quat.rotate_vector(rotation, rep[..., 7:10])
        joints, joint_velocity = rep[..., 10 : 10 + n], rep[..., 10 + n :]
        bodies = self.tree.forward(
            position,
            quat.to_matrix(rotation),
            velocity,
            angular,
            joints[..., self.tree_order],
            joint_velocity[..., self.tree_order],
        )
        pos, rotations, vel, ang = bodies
        quaternions = matrix_quaternion(rotations)
        result = dict(zip(self.root_keys, (position, rotation, velocity, angular), strict=True))
        result.update(zip(self.joint_keys, (joints, joint_velocity), strict=True))
        values = {
            read.BatchLinkPositionQuery: pos,
            read.BatchLinkQuaternionQuery: quaternions,
            read.BatchLinkLinearVelocityQuery: vel,
            read.BatchLinkAngularVelocityQuery: ang,
        }
        for key, indices in self.body_indices.items():
            result[key] = values[type(self.queries[key])][..., indices, :]
        return result


class TransitionAssembler:
    """Owned live snapshots with terminal rows replaced from FinalState."""

    def __init__(self, env, view, proprio_dim):
        self.env, self.view = env, view
        fields = [TransitionField(key, tuple(desc.shape), torch.float32) for key, desc in view.fields.items()]
        for channel in view.exogenous:
            tensor = self._channel(channel)
            fields.append(TransitionField("command/" + channel, tuple(tensor.shape[1:]), tensor.dtype))
        self.schema = wm_transition_schema(fields, env.action_space.shape[0], obs_dim=proprio_dim)

    def _channel(self, name):
        command, *path = name.split(".")
        value = self.env.command_terms[command]
        for part in path:
            value = getattr(value, part)
        return value

    def snapshot(self):
        result = {key: self.env.sim_data.view(key).clone() for key in self.view.fields}
        result.update({"command/" + name: self._channel(name).clone() for name in self.view.exogenous})
        return result

    def assemble(self, before, actions, obs, state):
        after = self.snapshot()
        final = state.final_state
        if final is not None:
            for key in self.view.fields:
                after[key][final.env_ids] = final.sim[key]
            for name in self.view.exogenous:
                after["command/" + name][final.env_ids] = final.commands[name]
        return {
            **before,
            **{"next_" + key: value for key, value in after.items()},
            "actions": actions,
            "obs": obs,
            "rewards": state.reward,
            "terminated": state.terminated.long(),
            "truncated": state.truncated.long(),
        }
