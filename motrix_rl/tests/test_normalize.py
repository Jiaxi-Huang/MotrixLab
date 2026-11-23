# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared normalizer contracts (P3.1, T-E).

The class moved from ``motrix_rl_builtin.fastsac.buffer`` (which re-exports
it — the identity test lives in the builtin suite) to ``motrix_rl.normalize``
unchanged; these tests pin the NORM_KEYS surface that feeds the async weight
channel.
"""

import torch

from motrix_rl.normalize import EmpiricalNormalization
from motrix_rl.transport.inference_bundle import NORM_KEYS


def test_exposes_the_norm_keys_surface() -> None:
    normalizer = EmpiricalNormalization(shape=5, device="cpu")
    for key in NORM_KEYS:
        assert hasattr(normalizer, key), key
    # The weight channel reads (1, dim) stat rows: reshape(1, -1) must hold.
    assert normalizer._mean.shape == (1, 5)
    assert normalizer._std.shape == (1, 5)
    assert normalizer._var.shape == (1, 5)
    assert int(normalizer.count) == 0


def test_update_tracks_batch_statistics() -> None:
    normalizer = EmpiricalNormalization(shape=3, device="cpu")
    data = torch.randn(4096, 3) * 2.0 + 5.0
    normalizer.update(data)
    torch.testing.assert_close(normalizer._mean.squeeze(0), data.mean(dim=0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(normalizer._var.squeeze(0), data.var(dim=0, unbiased=False), rtol=1e-3, atol=1e-3)
    assert int(normalizer.count) == 4096
    # Sequential batches match one pooled update.
    split = EmpiricalNormalization(shape=3, device="cpu")
    split.update(data[:1000])
    split.update(data[1000:])
    torch.testing.assert_close(split._mean, normalizer._mean, rtol=1e-5, atol=1e-5)


def test_until_freezes_updates() -> None:
    normalizer = EmpiricalNormalization(shape=2, device="cpu", until=16)
    first = torch.randn(16, 2)
    normalizer.update(first)
    frozen_mean = normalizer._mean.clone()
    normalizer.update(torch.randn(64, 2) * 10.0)
    assert torch.equal(normalizer._mean, frozen_mean)
    assert int(normalizer.count) == 16


def test_forward_normalizes_with_frozen_stats() -> None:
    normalizer = EmpiricalNormalization(shape=2, device="cpu")
    data = torch.randn(512, 2) * 3.0 + 1.0
    normalizer.update(data)
    normalizer.eval()
    out = normalizer(torch.tensor([[1.0, 1.0]]), update=False)
    torch.testing.assert_close(
        out, (torch.tensor([[1.0, 1.0]]) - normalizer._mean) / (normalizer._std + normalizer.eps)
    )


def test_local_accumulators_merge_round_trip() -> None:
    a = EmpiricalNormalization(shape=3, device="cpu")
    b = EmpiricalNormalization(shape=3, device="cpu")
    a.update(torch.randn(100, 3) + 1.0)
    b.update(torch.randn(100, 3) - 1.0)
    a.seed_local_accumulators()
    b.seed_local_accumulators()
    a.update(torch.randn(50, 3))
    b.update(torch.randn(50, 3))
    # Cross-rank merge: sum the packed [count, sum, sumsq] rows then restore.
    fa = a.local_sufficient_stats_flat()
    fb = b.local_sufficient_stats_flat()
    dim = 3
    merged = torch.cat(
        [fa[0].reshape(1) + fb[0].reshape(1), fa[1 : 1 + dim] + fb[1 : 1 + dim], fa[1 + dim :] + fb[1 + dim :]]
    )
    reference = EmpiricalNormalization(shape=3, device="cpu")
    reference.seed_local_accumulators()
    reference.apply_global_sufficient_stats(merged)
    assert int(reference.count) == 300
