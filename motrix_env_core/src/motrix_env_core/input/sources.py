# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Declarative command-source configs that build ``CommandBinding`` runtime objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.input.bindings import (
    BoundedGamePadPlanarVelocityBinding,
    CommandBinding,
    ConstantPlanarVelocityBinding,
    KeyboardPlanarVelocityBinding,
)
from motrix_env_core.input.device import GamePadDevice, KeyboardDevice

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@configclass(kw_only=True)
class CommandSourceCfg(ABC):
    """Declared command source for a manager ``CommandCfg``.

    ``__call__`` runs once at ``ManagerEnv`` construction: it builds the
    runtime ``CommandBinding`` the environment polls every transition. Fields
    mirror the binding constructor parameters, so each mapping knob (command
    range, deadzone, deadman button, axis inversion) stays declarative and
    hydra-overridable. Device-coupled sources hold the device instance here;
    concrete devices are owned by the viewer/backend layer that created them.
    """

    @abstractmethod
    def __call__(self, env: ManagerEnv) -> CommandBinding:
        """Build the runtime command binding for one environment."""


@configclass(kw_only=True)
class KeyboardPlanarVelocitySourceCfg(CommandSourceCfg):
    """Map held keyboard keys to planar velocity (W/S, A/D, Q/E)."""

    device: KeyboardDevice
    command_lower: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    command_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __call__(self, env: ManagerEnv) -> CommandBinding:
        del env
        return KeyboardPlanarVelocityBinding(
            self.device,
            command_lower=self.command_lower,
            command_upper=self.command_upper,
        )


@configclass(kw_only=True)
class BoundedGamePadPlanarVelocitySourceCfg(CommandSourceCfg):
    """Map normalized gamepad axes into bounded planar velocity commands."""

    device: GamePadDevice
    linear_x_axis: str
    linear_y_axis: str
    yaw_axis: str
    command_lower: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    command_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)
    deadzone: float = 0.0
    range_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    invert_linear_x: bool = False
    invert_linear_y: bool = False
    invert_yaw: bool = False
    deadman_button: str | None = None

    def __call__(self, env: ManagerEnv) -> CommandBinding:
        del env
        return BoundedGamePadPlanarVelocityBinding(
            self.device,
            linear_x_axis=self.linear_x_axis,
            linear_y_axis=self.linear_y_axis,
            yaw_axis=self.yaw_axis,
            command_lower=self.command_lower,
            command_upper=self.command_upper,
            deadzone=self.deadzone,
            range_scale=self.range_scale,
            invert_linear_x=self.invert_linear_x,
            invert_linear_y=self.invert_linear_y,
            invert_yaw=self.invert_yaw,
            deadman_button=self.deadman_button,
        )


@configclass(kw_only=True)
class ConstantPlanarVelocitySourceCfg(CommandSourceCfg):
    """Replicate one configured planar velocity across the batch."""

    value: tuple[float, float, float]

    def __call__(self, env: ManagerEnv) -> CommandBinding:
        del env
        return ConstantPlanarVelocityBinding(np.asarray(self.value, dtype=np.float32))


__all__ = [
    "BoundedGamePadPlanarVelocitySourceCfg",
    "CommandSourceCfg",
    "ConstantPlanarVelocitySourceCfg",
    "KeyboardPlanarVelocitySourceCfg",
]
