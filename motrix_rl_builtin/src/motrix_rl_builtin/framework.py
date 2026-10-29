# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The builtin Motrix framework: one provider per bundled agent.

Adding an agent means adding its provider here; the agent's own module owns
its implementation and never imports another agent's package.
"""

from motrix_rl.contracts import RlFramework
from motrix_rl_builtin.apg.framework import ApgProvider
from motrix_rl_builtin.fastsac.framework import FastSacProvider


class MotrixFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__((FastSacProvider(), ApgProvider()))

    @property
    def name(self) -> str:
        return "motrix"


__all__ = ["MotrixFramework"]
