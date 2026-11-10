# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The shared running normalizer for observation statistics.

Exposed attribute surface (``_mean``/``_std``/``_var``/``count``) is the
``NORM_KEYS`` contract of :mod:`motrix_rl.transport.inference_bundle`:
providers that normalize with this class sync their statistics over
``WeightChannelShared`` in async training without any adapter. Hoisted from
``motrix_rl_builtin.fastsac.buffer`` (which re-exports it) so world-model
trainers consume the one implementation.
"""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.

    Also maintains optional LOCAL accumulators (``_local_*``) for the
    multi-learner cross-rank merge: the public ``_mean``/``_var``/``count``
    may be overwritten by a merge (they then hold GLOBAL stats), so they
    must never feed the next merge — that would double-count all history on
    every sync (count doubles per merge until float32 overflows). The
    accumulators stay disabled (``local_enabled=False``) until a learner
    turns them on; single-learner runs never pay the mirrored-update cost.
    Local accumulators are plain attributes, deliberately NOT registered
    buffers — they are per-process bookkeeping, not checkpoint state.
    """

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.local_enabled = False
        # MPS has no float64; the local accumulators exist only for the
        # multi-learner merge (a CUDA-only path), so keep them on the CPU.
        acc_device = device if torch.device(device).type != "mps" else torch.device("cpu")
        self._local_mean = torch.zeros(shape, dtype=torch.float64, device=acc_device)
        self._local_var = torch.ones(shape, dtype=torch.float64, device=acc_device)
        self._local_count = 0
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        big_m2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(big_m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)
        # Mirror into the LOCAL accumulators: the per-rank cumulative stats
        # that cross-rank merges consume. The public _mean/_var/_count may be
        # overwritten by a merge (they then hold GLOBAL stats), so they must
        # never feed the next merge — that would double-count all history on
        # every sync (count doubles per merge until float32 overflows).
        if self.local_enabled:
            b_mean = batch_mean.squeeze(0)
            b_var = batch_var.squeeze(0)
            l_count = self._local_count + batch_size
            delta = b_mean - self._local_mean
            self._local_mean += delta * (batch_size / l_count)
            delta2 = b_mean - self._local_mean
            l_m2 = (
                self._local_var * self._local_count
                + b_var * batch_size
                + delta2.pow(2) * (self._local_count * batch_size / l_count)
            )
            self._local_var.copy_(l_m2 / l_count)
            self._local_count = l_count

    def seed_local_accumulators(self) -> None:
        """Enable local accumulation, seeded from the current public stats.

        Used on the first cross-rank merge (including after resume): seeding
        from the loaded global stats means the first merge does not discard
        pre-resume history (the shared history is counted once per rank, an
        acceptable one-off bias that washes out as new data arrives).
        """
        self.local_enabled = True
        self._local_mean = self._mean.detach().clone().squeeze(0).double()
        self._local_var = self._var.detach().clone().squeeze(0).double()
        self._local_count = int(self.count)

    def local_sufficient_stats_flat(self) -> torch.Tensor:
        """Pack the LOCAL accumulators as ``[count, sum, sumsq]`` (float64).

        Single-tensor layout so a cross-rank merge needs exactly one
        collective. ``sumsq`` is reconstructed from var via
        ``E[x^2] = var + mean^2`` — SUM is additive while mean/var are not.
        """
        count = torch.tensor(float(self._local_count), dtype=torch.float64, device=self._mean.device)
        total = count * self._local_mean
        sumsq = self._local_var * count + total * total / count.clamp_min(1.0)
        return torch.cat([count.reshape(1), total, sumsq])

    def apply_global_sufficient_stats(self, flat: torch.Tensor) -> None:
        """Restore merged ``[count, sum, sumsq]`` into the public buffers.

        Computes the global mean/var in float64 (``sumsq`` is a sum of
        same-scale squares; float32 would catastrophically cancel), clamps
        the sample count to ``until`` to match the update freeze, and writes
        ``_mean``/``_var``/``_std``/``count``. The local accumulators are
        untouched: they keep tracking rank-local samples only.
        """
        d = (flat.numel() - 1) // 2
        n, s, q = flat[0], flat[1 : 1 + d], flat[1 + d :]
        if n <= 0:
            return
        mean = (s / n).float()
        var = ((q - s * s / n) / n).clamp_min_(0.0).float()
        if self.until is not None:
            n = torch.minimum(n, torch.tensor(float(self.until), dtype=n.dtype, device=n.device))
        self._mean.copy_(mean.unsqueeze(0))
        self._var.copy_(var.unsqueeze(0))
        self._std.copy_(var.sqrt().unsqueeze(0))
        self.count.copy_(n.long())


__all__ = ["EmpiricalNormalization"]
