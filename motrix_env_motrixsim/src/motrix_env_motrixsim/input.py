# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Keyboard command source bound to the MotrixSim renderer window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from motrix_env_core.config import configclass
from motrix_env_core.input.bindings import KeyboardPlanarVelocityBinding
from motrix_env_core.input.device import KeyboardDevice
from motrix_env_core.input.sources import CommandSourceCfg

if TYPE_CHECKING:
    from motrixsim.render import Input

    from motrix_env_core.numba.manager.env import ManagerEnv


class RendererKeyboardDevice(KeyboardDevice):
    """Keyboard device reading the environment's MotrixSim renderer window.

    The device is constructed before the renderer exists (command sources are
    resolved at ``ManagerEnv`` construction), so it stays neutral — no keys
    pressed — until the backend has created a renderer. Key state reflects the
    renderer's last sync, which the play loop performs once per rendered step;
    ``poll`` is therefore a no-op.
    """

    def __init__(self, env: ManagerEnv) -> None:
        self._env = env

    def _frame(self) -> Input | None:
        renderer = getattr(self._env.sim, "renderer", None)
        return None if renderer is None else renderer.input

    def poll(self) -> None:
        """No-op: the input frame is frozen by the renderer's per-step sync."""

    def is_key_down(self, key: str) -> bool:
        frame = self._frame()
        return False if frame is None else bool(frame.is_key_just_pressed(key))

    def is_key_up(self, key: str) -> bool:
        raise NotImplementedError("the MotrixSim input frame exposes no key-release edge")

    def is_pressing(self, key: str) -> bool:
        frame = self._frame()
        return False if frame is None else bool(frame.is_key_pressed(key))


@configclass(kw_only=True)
class RendererKeyboardPlanarVelocitySourceCfg(CommandSourceCfg):
    """Map the renderer window's held keys (W/S, A/D, Q/E) to planar velocity."""

    command_lower: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    command_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __call__(self, env: ManagerEnv) -> KeyboardPlanarVelocityBinding:
        return KeyboardPlanarVelocityBinding(
            RendererKeyboardDevice(env),
            command_lower=self.command_lower,
            command_upper=self.command_upper,
        )


__all__ = ["RendererKeyboardDevice", "RendererKeyboardPlanarVelocitySourceCfg"]
