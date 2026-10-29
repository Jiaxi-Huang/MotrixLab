# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-side APG parsing head for the quadruped velocity-walk tasks.

Implements the APG ``ApgTask`` protocol (see the provider's
``motrix_rl_builtin.apg.contracts``) duck-typed — no provider types or
training logic are imported, only the shared torch rotation/state-layout
math from ``motrix_env_core.torch.geometry`` — over
:class:`QuadrupedWalkTask` direct environments such as
``go2-walk-flat``. This head exercises the protocol's command-driven shape:
goals and rewards read the joystick command from the context vector instead
of a reference corpus, and the replay index ``f`` encodes gait-phase steps.

Task-side design choices, mirrored from the env's own semantics:

- The world-root state (position, quaternion, velocities) is read through a
  head-compiled Link query program; joints come from the env's public
  accessors.
- The imagination reward mirrors the env's ``_reward_*`` terms. Gait phase
  is reconstructed analytically from ``f`` (``phase = fmod(f * ctrl_dt *
  gait_frequency, 1)``), which matches the env's phase generator while a
  gait is active; the env's standing freeze and terrain sampling are not
  modeled (flat-ground tasks). Foot contact in imagination is a height
  proxy (body-relative foot height below half the base-height target)
  instead of a collision query.
- The context vector is ``[command (3) | friction scale (1) | mass scale
  (1) | CoM x/y offset (2) | action-delay flag (1)]``; randomization values
  come from the task's override buffers, falling back to nominal model
  values when a randomization is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from motrix_env_core.sim import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_env_core.torch.geometry import flat_state_dim, invariant_features
from motrix_envs.locomotion.kinematics import ModelKinematics

# Context layout: [command 3 | friction scale 1 | mass scale 1 | CoM x 1 |
# CoM y 1 | action-delay flag 1].
_CONTEXT_HEAD_DIM = 3
_CONTEXT_TAIL_DIM = 5
# Feet below this fraction of the base-height target count as in contact
# in imagination.
_CONTACT_HEIGHT_FACTOR = 0.5
_TERMINATION_MARGIN = 0.05


def _feet_phase(frames: torch.Tensor, gait_step: float, trot_pairs, num_feet: int) -> torch.Tensor:
    """Analytic trot phase per foot from integer gait steps ``f``."""
    phase = torch.remainder(frames.to(torch.float32) * gait_step, 1.0)
    feet = torch.zeros(frames.shape + (num_feet,), device=frames.device)
    for pair_index, (i, j) in enumerate(trot_pairs):
        value = torch.remainder(phase + 0.5 * pair_index, 1.0)
        feet[..., i] = value
        feet[..., j] = value
    return feet


class QuadrupedWalkApgHead:
    """The quadruped velocity-walk parsing head: analytic torch heads plus env reads.

    Attributes:
        num_joints: Actuated joint count ``N`` (12 for the go2).
        context_dim: Context width (command + randomization), ``8``.
        obs_dim: Proprioceptive observation width ``3N + 10`` (local angular
            velocity, gravity, joint offsets, joint velocities, previous
            action, per-foot gait phase).
        goal_dim: Goal width — the 3-D velocity command. The command is
            quasi-static over an imagination horizon, so the goal is a
            single frame read from the context.
        max_step_reward: Dt-free per-step reward ceiling from the positive
            reward scales.
    """

    def __init__(self, env, device: torch.device | str = "cpu") -> None:
        cfg = env.cfg
        robot = cfg.scene.objs.robot
        base_link = robot.resolved_base_link_name
        self._env = env
        self.device = torch.device(device)
        self.ctrl_dt = float(cfg.ctrl_dt)
        self.num_joints = int(env.num_actuators)
        self.num_feet = int(env.num_feet)
        self.context_dim = _CONTEXT_HEAD_DIM + _CONTEXT_TAIL_DIM
        self.obs_dim = 3 * self.num_joints + 10
        self.goal_dim = _CONTEXT_HEAD_DIM
        self.state_dim = flat_state_dim(self.num_joints)
        self._gait_step = self.ctrl_dt * float(cfg.gait_frequency)
        self._trot_pairs = tuple(tuple(pair) for pair in cfg.trot_pairs)
        self.default_q = torch.tensor(np.asarray(env.default_angles, dtype=np.float32), device=self.device)
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self._reward_cfg = cfg.reward_config
        self._scales = cfg.reward_config.scales
        self.max_step_reward = float(
            max(self._scales.tracking_lin_vel, 0.0)
            + max(self._scales.tracking_ang_vel, 0.0)
            + max(self._scales.contact, 0.0)
            + max(self._scales.swing_feet_z, 0.0)
        )
        noise = cfg.noise_config
        level = float(noise.level)
        self.obs_noise = torch.tensor(
            np.concatenate(
                (
                    np.full(3, level * noise.scale_gyro),
                    np.full(3, level * noise.scale_gravity),
                    np.full(self.num_joints, level * noise.scale_joint_angle),
                    np.full(self.num_joints, level * noise.scale_joint_vel),
                    np.zeros(self.num_joints),
                    np.zeros(self.num_feet),
                )
            ).astype(np.float32),
            device=self.device,
        )
        self._contact_height = -_CONTACT_HEIGHT_FACTOR * float(self._reward_cfg.base_height_target)
        self._swing_gate = 0.6

        # World-root state reads through the live sim backend; the env's own
        # reads do not include base rotation or world velocities.
        self._reads = env.sim.compile_reads(
            {
                "p": LinkPositionQuery(link=base_link),
                "quat": LinkQuaternionQuery(link=base_link),
                "v": LinkLinearVelocityQuery(link=base_link),
                "w": LinkAngularVelocityQuery(link=base_link),
            }
        )
        joint_names = [str(spec.target_name) for spec in env.model.actuators]
        foot_links = self._resolve_foot_links(robot)
        self.kin = ModelKinematics(robot.model.file, joint_names, foot_links, device=device)
        others = env.model.others
        self._nominal_friction = float(np.asarray(others["ground_friction"]).reshape(-1)[0])
        self._nominal_mass = float(np.asarray(others["base_mass"]).reshape(-1)[0])
        self._nominal_com = np.asarray(others["base_com"], dtype=np.float64).reshape(3)

    def _resolve_foot_links(self, robot) -> list[str]:
        import motrixsim as mtx

        model = mtx.load_model(str(robot.model.file))
        links = []
        for geom_name in robot.foot_contact_geom_names:
            if geom_name is None:
                raise TypeError("The quadruped APG head requires a contact geom on every leg")
            link = model.get_geom(geom_name).link
            if link is None:
                raise ValueError(f"Contact geom {geom_name!r} has no owning link")
            links.append(str(link.name))
        return links

    def to(self, device: torch.device | str) -> QuadrupedWalkApgHead:
        self.device = torch.device(device)
        self.default_q = self.default_q.to(self.device)
        self.obs_noise = self.obs_noise.to(self.device)
        self.kin.axes = self.kin.axes.to(self.device)
        self.kin.off_p = self.kin.off_p.to(self.device)
        self.kin.off_R = self.kin.off_R.to(self.device)
        self.kin.q_lo = self.kin.q_lo.to(self.device)
        self.kin.q_hi = self.kin.q_hi.to(self.device)
        return self

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def ref_state(self, frames: torch.Tensor):
        """The standing default state; used only for shape probing."""
        shape = frames.shape
        position = torch.zeros(shape + (3,), device=frames.device)
        position[..., 2] = float(self._env.cfg.initial_base_position[2])
        eye = torch.eye(3, device=frames.device).expand(shape + (3, 3))
        q = self.default_q.to(frames.device).expand(shape + (self.num_joints,))
        return (position, eye, torch.zeros_like(position), torch.zeros_like(position), q, torch.zeros_like(q))

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        """Command tasks have no corpus clips: every start is valid."""
        return torch.ones(frames.shape, dtype=torch.bool, device=frames.device)

    def _feet_heights(self, state) -> torch.Tensor:
        """Body-relative foot heights along an imagined rollout."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        foot_p = self.kin.forward(position, rotation, lin_vel, ang_vel, joint_pos, joint_vel)[0]
        return foot_p[..., 2] - position[..., 2:3]

    # ------------------------------------------------------------------
    # Observation / goal features
    # ------------------------------------------------------------------

    def obs(self, state, previous_action: torch.Tensor, context: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Proprioceptive features in the env policy group's block order.

        Local angular velocity, gravity direction, joint offsets, joint
        velocities, previous action, per-foot gait phase — every policy
        block except the command.
        """
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        gyro = (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1)
        gravity = -rotation[..., 2, :]
        phase = _feet_phase(frames, self._gait_step, self._trot_pairs, self.num_feet)
        return torch.cat((gyro, gravity, joint_pos - self.default_q, joint_vel, previous_action, phase), -1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def goal(self, frames: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """The velocity command, quasi-static over the horizon."""
        del frames
        return context[..., :_CONTEXT_HEAD_DIM]

    # ------------------------------------------------------------------
    # Analytic reward / termination heads
    # ------------------------------------------------------------------

    def reward(
        self,
        state,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """The dt-weighted walk reward, mirroring the env reward group.

        Uses the analytic gait phase and a foot-height contact proxy in
        place of the env's phase generator state and collision queries; the
        command rides in the context vector.
        """
        cfg = self._reward_cfg
        scales = self._scales
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        command = context[..., :_CONTEXT_HEAD_DIM]
        rotation_t = rotation.transpose(-1, -2)
        local_linvel = (rotation_t @ lin_vel.unsqueeze(-1)).squeeze(-1)
        gyro = (rotation_t @ ang_vel.unsqueeze(-1)).squeeze(-1)
        phase = _feet_phase(frames, self._gait_step, self._trot_pairs, self.num_feet)
        feet_z = self._feet_heights(state)
        contacts = feet_z < self._contact_height
        swing = phase >= self._swing_gate

        tracking_lin = torch.exp(
            -((command[..., 0:2] - local_linvel[..., 0:2]) ** 2).sum(-1) / cfg.tracking_lin_vel_sigma
        )
        tracking_ang = torch.exp(-((command[..., 2] - gyro[..., 2]) ** 2) / cfg.tracking_ang_vel_sigma)
        lin_vel_z = local_linvel[..., 2] ** 2
        ang_vel_xy = (gyro[..., :2] ** 2).sum(-1)
        base_height_error = position[..., 2] - cfg.base_height_target
        action_rate = ((action - previous_action) ** 2).sum(-1)
        similar_to_default = (joint_pos - self.default_q).abs().sum(-1)
        target_z = cfg.target_foot_height - cfg.base_height_target
        height_error = (feet_z - target_z) ** 2
        swing_feet = (torch.exp(-height_error / cfg.swing_feet_height_sigma**2) * (swing & ~contacts)).sum(
            -1
        ) / self.num_feet
        contact_match = (contacts == (phase < self._swing_gate)).to(torch.float32).sum(-1) / self.num_feet
        swing_contact = (contacts & swing).to(torch.float32).sum(-1) / self.num_feet

        total = (
            scales.tracking_lin_vel * tracking_lin
            + scales.tracking_ang_vel * tracking_ang
            + scales.lin_vel_z * lin_vel_z
            + scales.ang_vel_xy * ang_vel_xy
            + scales.base_height * base_height_error**2
            + scales.action_rate * action_rate
            + scales.similar_to_default * similar_to_default
            + scales.swing_feet_z * swing_feet
            + scales.contact * contact_match
            + scales.swing_contact * swing_contact
        )
        return total * self.ctrl_dt

    def soft_termination(self, state, frames: torch.Tensor) -> torch.Tensor:
        """Differentiable tilt hazard around ``termination_min_up_z``."""
        rotation = state[1]
        up_z = rotation[..., 2, 2]
        return torch.sigmoid((float(self._env.cfg.termination_min_up_z) - up_z) / _TERMINATION_MARGIN)

    # ------------------------------------------------------------------
    # Critic features / context
    # ------------------------------------------------------------------

    def critic_features(
        self,
        state,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Privileged value features: invariant state plus the context."""
        return torch.cat((invariant_features(state), normalized_context), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                context[..., 0:3],
                context[..., 3:4] - 1.0,
                context[..., 4:5] - 1.0,
                context[..., 5:7] / 0.01,
                context[..., 7:8],
            ),
            -1,
        )

    # ------------------------------------------------------------------
    # Collection hooks over the direct task's surfaces
    # ------------------------------------------------------------------

    def current_transition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World-root state, gait-phase index, and context of the pending step."""
        self._reads.execute()
        state = self._read_state()
        frames = np.rint(np.asarray(self._env.gait_phase) / self._gait_step).astype(np.int64)
        return state, frames, self._read_context()

    def endpoint_state(self) -> np.ndarray:
        """The world-root state after the step just taken."""
        self._reads.execute()
        return self._read_state()

    def _read_state(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self._reads["p"], dtype=np.float32),
                np.asarray(self._reads["quat"], dtype=np.float32),
                np.asarray(self._reads["v"], dtype=np.float32),
                np.asarray(self._reads["w"], dtype=np.float32),
                np.asarray(self._env.get_dof_pos(), dtype=np.float32),
                np.asarray(self._env.get_dof_vel(), dtype=np.float32),
            ),
            axis=-1,
        )

    def _read_context(self) -> np.ndarray:
        env = self._env
        command = np.asarray(env.velocity_commands, dtype=np.float32)
        friction_scale = env.sliding_friction() / max(self._nominal_friction, 1e-6)
        mass_scale = env.base_mass() / max(self._nominal_mass, 1e-6)
        com_xy = (env.base_com()[:, :2] - self._nominal_com[:2]).astype(np.float32)
        delay = np.asarray(env.action_delay_steps, dtype=np.float32)
        return np.concatenate((command, friction_scale[:, None], mass_scale[:, None], com_xy, delay[:, None]), axis=-1)

    def split_observation(self, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy group: [gyro | gravity | joints | action | command | phase]."""
        command_start = 6 + 3 * self.num_joints
        command_end = command_start + _CONTEXT_HEAD_DIM
        obs = torch.cat((full[..., :command_start], full[..., command_end:]), -1)
        return obs, full[..., command_start:command_end]

    def termination_counts(self, metrics: Mapping[str, np.ndarray]) -> float:
        """Tilt terminations, when the metrics carry the raw flag."""
        if "terminated" not in metrics:
            return 0.0
        return float(np.asarray(metrics["terminated"], dtype=bool).sum())

    def sampling_state(self):
        return None

    def load_sampling_state(self, state) -> None:
        del state

    def sampling_metrics(self) -> Mapping[str, float]:
        return {}


def build_head(env) -> QuadrupedWalkApgHead:
    """APG head factory: assemble the parsing head from the environment."""
    return QuadrupedWalkApgHead(env)


__all__ = ["QuadrupedWalkApgHead", "build_head"]
