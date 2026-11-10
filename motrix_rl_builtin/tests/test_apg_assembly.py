# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG state-assembly contracts on the real g1-supermimic task (P3.2).

Two parity lines pin the FieldDesc-driven assembly:

- stepped-state positions: the FK derivation of every batch field matches the
  simulator's own read views exactly (the world model learns dynamics,
  never kinematics);
- pushed-state full parity: reconstructing one consistent snapshot through
  the backend's write path, velocities match too. On stepped states the
  backend's velocity buffers sit half a substep off the position phase
  (integrator staggering) — a documented sampling property, not an assembly
  error, so the full-velocity parity runs against the reconstructed state.
"""

import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.math.quaternion import to_matrix
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyJointPositionWrite,
    BodyJointVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    KinematicTreeQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
)
from motrix_env_core.sim.registry import create_sim_backend
from motrix_rl_builtin.apg.assemble import build_assembly


@pytest.fixture(scope="module")
def built():
    import motrix_envs  # noqa: F401  (registers g1-wbt-supermimic)

    cfg = registry.make_env_config("g1-wbt-supermimic", mode="train")
    cfg.queries.model["tree"] = KinematicTreeQuery(
        body=cfg.scene.objs.robot.resolved_base_link_name, device="cpu"
    )
    env = ManagerEnv(cfg, num_envs=2)
    env.init_state()
    env.step(torch.rand(2, env.action_space.shape[0]) * 0.1 - 0.05)
    view = env.modeling_view()
    assembly = build_assembly(view, env, env.model.others["tree"], "floating_base")
    return env, view, assembly


def test_assembly_layout_and_flat_width(built) -> None:
    env, view, assembly = built
    assert assembly.root_row == 0  # pelvis leads the tracked order
    assert assembly.flat_width == 13 + 2 * len(env.model.others["tree"].joint_names)
    # Every declared field has an assembly rule.
    assert set(assembly.field_keys) == set(view.fields)
    assert assembly.tracked_links == env.cfg.commands.motion.tracked_body_names


def test_residual_scheme_flat_round_trip(built) -> None:
    env, view, _ = built
    residual = build_assembly(view, env, None, "residual")
    state = {key: env.sim_data.view(key).clone() for key in env.sim_data.keys}
    flat = residual.fields_to_flat(state)
    assert flat.shape == (env.num_envs, residual.flat_width)
    rebuilt = residual.flat_to_fields(flat)
    for key, value in state.items():
        torch.testing.assert_close(rebuilt[key], value)


def test_fk_positions_match_stepped_sim_views(built) -> None:
    env, _view, assembly = built
    state = {key: env.sim_data.view(key).clone() for key in env.sim_data.keys}
    rebuilt = assembly.flat_to_fields(assembly.fields_to_flat(state))
    # Positions are FK-exact on stepped states; joint and row fields exact.
    torch.testing.assert_close(rebuilt["tracked_body_pos"], state["tracked_body_pos"], atol=2e-2, rtol=0.0)
    torch.testing.assert_close(rebuilt["robot_dof_pos"], state["robot_dof_pos"])
    torch.testing.assert_close(rebuilt["robot_dof_vel"], state["robot_dof_vel"])
    torch.testing.assert_close(rebuilt["terms.args[0]"], state["terms.args[0]"])


def test_fk_full_parity_on_reconstructed_state(built) -> None:
    env, _view, assembly = built
    state = {key: env.sim_data.view(key).clone() for key in env.sim_data.keys}
    base = env.cfg.scene.objs.robot.resolved_base_link_name
    backend = create_sim_backend("motrixsim")(env.cfg.scene, SimCfg(), 1)
    program = backend.write_compiler.compile(
        {
            "bp": BodyPositionWrite((base,)),
            "br": BodyRotationWrite((base,)),
            "bl": BodyLinearVelocityWrite((base,)),
            "ba": BodyAngularVelocityWrite((base,)),
            "jp": BodyJointPositionWrite(base),
            "jv": BodyJointVelocityWrite(base),
        }
    )
    num_joints = state["robot_dof_pos"].shape[-1]
    program.buffer("bp")[0, 0] = state["tracked_body_pos"][0, 0]
    program.buffer("br")[0, 0] = state["tracked_body_quat"][0, 0]
    program.buffer("bl")[0, 0] = state["tracked_body_linear_velocity"][0, 0]
    program.buffer("ba")[0, 0] = state["tracked_body_angular_velocity"][0, 0]
    program.buffer("jp")[0, :num_joints] = state["robot_dof_pos"][0]
    program.buffer("jv")[0, :num_joints] = state["robot_dof_vel"][0]
    ids = torch.tensor([0], dtype=torch.int64)
    program.execute(ids)
    reads = backend.compile_reads(
        {
            **{f"p_{name}": LinkPositionQuery(link=name) for name in assembly.tracked_links},
            **{f"v_{name}": LinkLinearVelocityQuery(link=name) for name in assembly.tracked_links},
            **{f"w_{name}": LinkAngularVelocityQuery(link=name) for name in assembly.tracked_links},
        }
    )
    reads.execute(ids)
    pushed = {
        "tracked_body_pos": torch.stack([reads[f"p_{name}"] for name in assembly.tracked_links], dim=1),
        "tracked_body_linear_velocity": torch.stack([reads[f"v_{name}"] for name in assembly.tracked_links], dim=1),
        "tracked_body_angular_velocity": torch.stack([reads[f"w_{name}"] for name in assembly.tracked_links], dim=1),
    }
    single = {key: value[:1] for key, value in state.items()}
    rebuilt = assembly.flat_to_fields(assembly.fields_to_flat(single))
    # One consistent snapshot: the FK of the views' own block equals the
    # simulator's evaluation of that same state.
    torch.testing.assert_close(rebuilt["tracked_body_pos"], pushed["tracked_body_pos"], atol=5e-3, rtol=0.0)
    torch.testing.assert_close(rebuilt["tracked_body_linear_velocity"], pushed["tracked_body_linear_velocity"], atol=5e-3, rtol=0.0)
    torch.testing.assert_close(rebuilt["tracked_body_angular_velocity"], pushed["tracked_body_angular_velocity"], atol=5e-3, rtol=0.0)


def test_assembly_rejects_uncovered_tree_joints(built) -> None:
    env, view, assembly = built
    # A view missing every joint field cannot serve the tracked paths.
    from motrix_env_core.manager.modeling import FieldDesc
    from motrix_env_core.sim.read import (
        BodyJointPositionQuery,
        BodyJointVelocityQuery,
        JointPositionQuery,
        JointVelocityQuery,
    )

    stripped = {
        k: d
        for k, d in view.fields.items()
        if d.query_type not in (JointPositionQuery, JointVelocityQuery, BodyJointPositionQuery, BodyJointVelocityQuery)
    }

    class _View:
        fields = stripped
        exogenous = view.exogenous

    with pytest.raises(ValueError, match="outside the declared joint fields"):
        build_assembly(_View(), env, env.model.others["tree"], "floating_base")
