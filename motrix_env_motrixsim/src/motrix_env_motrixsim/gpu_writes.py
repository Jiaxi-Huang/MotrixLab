# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute batched writes to MotrixSim CUDA state."""

import torch
import warp as wp

from motrix_env_core.sim import write as declarations
from motrix_env_core.sim.write import CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite, WriteProgram


class CudaWrites(WriteProgram):
    def __init__(self, backend, writes, reset, forward_kinematics):
        self._backend = backend
        self._reset = reset
        self._refresh = reset
        self._buffers = {}
        self._targets = {}
        self._mocap = {}
        self._parameters = {}
        for key, write in writes.items():
            shape = None
            if isinstance(write, (declarations.KinematicBodyPositionWrite, declarations.KinematicBodyRotationWrite)):
                rotation = isinstance(write, declarations.KinematicBodyRotationWrite)
                bodies = [backend.model.get_body(n).mocap for n in write.bodies]
                self._mocap[key] = bodies, slice(3, 7) if rotation else slice(3)
                self._buffers[key] = backend.position.new_zeros(
                    (backend.data.num_worlds, len(bodies), 4 if rotation else 3)
                )
                self._refresh = True
                continue
            overrides = []
            if isinstance(write, (declarations.ActuatorKpWrite, declarations.ActuatorDampingWrite)):
                for name in write.actuators:
                    actuator = backend.model.get_actuator(name)
                    if isinstance(write, declarations.ActuatorKpWrite):
                        overrides.append((name, actuator.get_kp_override, actuator.set_kp_override))
                    else:
                        overrides.append((name, actuator.get_damping_override, actuator.set_damping_override))
            elif isinstance(write, (declarations.LinkMassWrite, declarations.LinkComWrite)):
                for name in write.links:
                    link = backend.model.get_link(name)
                    if isinstance(write, declarations.LinkMassWrite):
                        overrides.append((name, link.get_mass_override, link.set_mass_override))
                    else:
                        overrides.append((name, link.get_center_of_mass_override, link.set_center_of_mass_override))
            elif isinstance(write, declarations.GeomFrictionWrite):
                for name in write.geoms:
                    geom = backend.model.get_geom(name)
                    overrides.append((name, geom.get_friction_override, geom.set_friction_override))
            if overrides:
                setters = [setter for _, _, setter in overrides]
                vector = isinstance(write, (declarations.GeomFrictionWrite, declarations.LinkComWrite))
                shape = (backend.data.num_worlds, len(setters)) + ((3,) if vector else ())
                self._buffers[key] = backend.position.new_zeros(shape)
                spin_roll = None
                if isinstance(write, declarations.GeomFrictionWrite):
                    spin_roll = backend.position.new_tensor(
                        [backend.host_model.get_geom(n).friction[1:] for n in write.geoms]
                    )
                for name, getter, setter in overrides:
                    identity = (type(write), name)
                    if identity not in backend._reset_parameters:
                        default = backend.position.new_tensor(getter(backend.data))
                        width = 3 if isinstance(write, declarations.LinkComWrite) else 1
                        default = default.reshape(-1, width).expand(backend.data.num_worlds, width).clone()
                        if width == 1:
                            default = default[:, 0]
                        backend._reset_parameters[identity] = setter, default
                self._parameters[key] = setters, spin_roll
                continue
            if isinstance(write, CtrlTargetsWrite):
                target = backend.control
                names = backend.model.actuator_names
                cols = range(len(names)) if write.actuators is None else [names.index(n) for n in write.actuators]
            elif isinstance(write, (JointPositionWrite, JointVelocityWrite)):
                position = isinstance(write, JointPositionWrite)
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.joints:
                    joint = backend.model.get_joint(name)
                    if joint is None or joint.num_dof_vel != 1:
                        raise ValueError(f"Expected a one-DOF joint for {key!r}, got {name!r}")
                    cols.append(joint.dof_pos_index if position else joint.dof_vel_index)
                self._refresh = True
            elif isinstance(write, (declarations.BodyJointPositionWrite, declarations.BodyJointVelocityWrite)):
                body = backend.model.get_body(write.body)
                position = isinstance(write, declarations.BodyJointPositionWrite)
                target = backend.position if position else backend.velocity
                cols = body.get_dof_pos_indices(False) if position else body.get_dof_vel_indices(False)
                self._refresh = True
            elif isinstance(
                write,
                (
                    declarations.BodyPositionWrite,
                    declarations.BodyRotationWrite,
                    declarations.BodyLinearVelocityWrite,
                    declarations.BodyAngularVelocityWrite,
                ),
            ):
                position = isinstance(write, (declarations.BodyPositionWrite, declarations.BodyRotationWrite))
                angular = isinstance(write, (declarations.BodyRotationWrite, declarations.BodyAngularVelocityWrite))
                width = 4 if isinstance(write, declarations.BodyRotationWrite) else 3
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.bodies:
                    base = backend.model.get_body(name).floatingbase
                    start = (base.dof_pos_start if position else base.dof_vel_start) + (3 if angular else 0)
                    cols.extend(range(start, start + width))
                shape = (len(write.bodies), width)
                self._refresh = True
            elif isinstance(write, (declarations.JointQuaternionWrite, declarations.JointAngularVelocityWrite)):
                position = isinstance(write, declarations.JointQuaternionWrite)
                width = 4 if position else 3
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.joints:
                    joint = backend.model.get_joint(name)
                    start = joint.dof_pos_index if position else joint.dof_vel_index
                    cols.extend(range(start, start + width))
                shape = (len(write.joints), width)
                self._refresh = True
            else:
                raise NotImplementedError(f"MotrixSim CUDA cannot write {key!r} ({type(write).__name__})")
            self._buffers[key] = target.new_zeros((target.shape[0], *(shape or (len(cols),))))
            self._targets[key] = target, torch.tensor(list(cols), dtype=torch.long, device=target.device)
        self._refresh &= forward_kinematics

    def buffer(self, name):
        return self._buffers[name]

    def execute(self, env_ids=None):
        backend = self._backend
        rows = slice(None) if env_ids is None else env_ids
        if self._reset:
            backend.reset_rows(rows)
        for key, (target, columns) in self._targets.items():
            row_index = rows if env_ids is None else rows[:, None]
            target[row_index, columns] = self._buffers[key].flatten(1)[rows]
        if self._refresh or self._parameters:
            with backend.physics_stream():
                for key, (setters, spin_roll) in self._parameters.items():
                    values = self._buffers[key]
                    if spin_roll is not None:
                        if (values[rows, :, 1:] != spin_roll).any():
                            raise NotImplementedError("MotrixSim CUDA 1.0.0 supports only sliding-friction overrides")
                        values = values[:, :, 0]
                    if env_ids is not None:
                        selected = torch.full_like(values, torch.nan)
                        selected[rows] = values[rows]
                        values = selected
                    for column, setter in enumerate(setters):
                        setter(backend.data, wp.from_torch(values[:, column].contiguous()))
                for key, (bodies, components) in self._mocap.items():
                    for i, mocap in enumerate(bodies):
                        poses = wp.to_torch(mocap.get_pose(backend.data))
                        poses[rows, components] = self._buffers[key][rows, i]
                        backend.set_mocap_pose(mocap, poses, rows)
                if self._refresh:
                    backend.model.forward_kinematic(backend.data)


class CudaWriteCompiler:
    def __init__(self, backend):
        self.backend = backend

    def compile(self, writes, *, reset=False, forward_kinematics=True):
        return CudaWrites(self.backend, writes, reset, forward_kinematics)
