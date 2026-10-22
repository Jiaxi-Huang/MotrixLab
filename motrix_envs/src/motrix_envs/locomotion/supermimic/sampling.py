# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic return-based start-bin sampling (host side).

Port of UniSuperMimic ``manager_terms.start_bin_probabilities`` /
``update_start_bin_returns`` (arXiv 2610.28480, App. D.1): every corpus bin
keeps a running estimate ``G_b`` of the discounted return episodes earn when
started from it, and reset frames are drawn from

.. math::

    p_b = 0.5/N + 0.4\\,\\mathrm{softmax}(-25\\,G_b/G_{max}) + 0.1\\,\\bar w_b

where :math:`\\bar w_b` is a slow EMA of the softmax weights (0.999/0.001 per
update) so a hard bin keeps some practice mass after it is learned, and
:math:`G_{max} = r_{max}\\,dt/(1-\\gamma)` is the largest attainable
discounted return. ``G_b`` is an EMA with the upstream 0.8/0.2 blend.

Episode ends bootstrapped like upstream: hard terminations (height/tilt)
credit the raw discounted return, while time-outs and clip-end episodes —
successes — additionally receive :math:`\\gamma^L \\bar r_L /(1-\\gamma)` with
:math:`\\bar r_L` the episode's mean per-step reward.

The sampler is owned and ticked by the task env after each control step; it
publishes the mixture CDF into the command's shared ``bin_cdf`` so the reset
kernel draws start frames from it.
"""

from __future__ import annotations

import numpy as np

# Upstream mixture shares and softmax temperature (App. D.1).
UNIFORM_SHARE = 0.5
SOFTMAX_SHARE = 0.4
PAST_SHARE = 0.1
SOFTMAX_TEMPERATURE = 25.0
PAST_DECAY = 0.999
PAST_GAIN = 0.001


def _low_return_softmax(returns: np.ndarray, g_max: float) -> np.ndarray:
    """Softmax over ``-temperature * G / G_max`` — high weight on low returns."""
    logits = -SOFTMAX_TEMPERATURE * returns / max(g_max, 1e-12)
    weights = np.exp(logits - np.max(logits))
    return weights / np.sum(weights)


class SuperMimicStartSampler:
    """Host-side episode bookkeeping and bin-mixture publisher.

    The sampler is passive: :meth:`tick` receives the step reward and the
    done-cause masks, accumulates each lane's discounted return, credits
    finished episodes to their start bins, and republishes the mixture CDF
    into the command. Episode boundaries are detected through the command's
    per-lane reset counter, so every reset cause (failure, time-out, clip
    end, auto-reset) is credited exactly once.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        bin_start: np.ndarray,
        bin_end: np.ndarray,
        gamma: float,
        return_ema: float,
        ctrl_dt: float,
        max_step_reward: float,
    ) -> None:
        self.num_bins = int(bin_start.shape[0])
        self.gamma = float(gamma)
        self.return_ema = float(return_ema)
        # Largest attainable discounted return (upstream G_max).
        self.g_max = float(max_step_reward) * float(ctrl_dt) / (1.0 - self.gamma)
        # Per-bin return estimate and the past-softmax retention weights.
        self.bin_returns = np.zeros((self.num_bins,), dtype=np.float64)
        self.bin_seen = np.zeros((self.num_bins,), dtype=bool)
        self.bin_past_weights = np.full((self.num_bins,), 1.0 / self.num_bins, dtype=np.float64)
        self.probabilities = np.full((self.num_bins,), 1.0 / self.num_bins, dtype=np.float64)
        # Per-lane episode accounting.
        self._lane_return = np.zeros((num_envs,), dtype=np.float64)
        self._lane_sum = np.zeros((num_envs,), dtype=np.float64)
        self._lane_power = np.ones((num_envs,), dtype=np.float64)
        self._lane_steps = np.zeros((num_envs,), dtype=np.int64)
        self._lane_start_bin = np.zeros((num_envs,), dtype=np.int64)
        # Lanes start un-anchored: ``attach`` runs before the env's initial
        # reset, so the first detected reset only anchors the start bin
        # (the construction bookend is not a creditable episode).
        self._lane_anchored = np.zeros((num_envs,), dtype=bool)
        self._lane_counters = None

    def attach(self, term) -> None:
        """Anchor to a freshly constructed command term.

        ``attach`` runs before the env's initial reset, so lanes stay
        un-anchored: the first reset detected by :meth:`tick` is the
        construction bookend and only anchors the start bins, without
        crediting an episode.
        """
        self._lane_counters = term.reset_counter[:, 0].astype(np.int64).copy()

    def tick(self, term, reward: np.ndarray, hard_done: np.ndarray) -> dict[str, float]:
        """Accumulate one step, credit finished episodes, republish the CDF.

        Args:
            term: The task's :class:`SuperMimicMotionCommand`.
            reward: Per-environment step reward (already dt-weighted).
            hard_done: Per-environment failure masks (terminations that end
                the episode without bootstrapping); every other reset cause
                (time-out, clip end) is bootstrapped from the mean reward.

        Returns:
            Batch metrics for the step, or an empty dict on steps without
            resets (the mixture is unchanged there).
        """
        self._lane_return += self._lane_power * reward.astype(np.float64)
        self._lane_sum += reward.astype(np.float64)
        self._lane_power *= self.gamma
        self._lane_steps += 1
        counters = term.reset_counter[:, 0].astype(np.int64)
        ended = np.flatnonzero(counters != self._lane_counters)
        if ended.size == 0:
            return {}
        self._lane_counters = counters.copy()
        ran = ended[(self._lane_steps[ended] > 0) & self._lane_anchored[ended]]
        if ran.size:
            lengths = self._lane_steps[ran].astype(np.float64)
            bootstrap = np.where(
                hard_done[ran],
                0.0,
                self.gamma**lengths * self._lane_sum[ran] / lengths / (1.0 - self.gamma),
            )
            credited = self._lane_return[ran] + bootstrap
            bins = self._lane_start_bin[ran]
            # Upstream blends unconditionally (bin_G <- 0.8 * bin_G + 0.2 * G),
            # so a bin's first credit starts at 0.2 * G.
            self.bin_returns[bins] = (1.0 - self.return_ema) * self.bin_returns[bins] + self.return_ema * credited
            self.bin_seen[bins] = True
        self._republish(term)
        # Re-anchor the lanes to the bins the reset kernel drew; zero the
        # accounting for the new episodes.
        self._lane_start_bin[ended] = term.start_bin[ended, 0]
        self._lane_return[ended] = 0.0
        self._lane_sum[ended] = 0.0
        self._lane_power[ended] = 1.0
        self._lane_steps[ended] = 0
        self._lane_anchored[ended] = True
        return self.metrics()

    def _republish(self, term) -> None:
        """Recompute the upstream mixture and publish it as the kernel CDF."""
        soft = _low_return_softmax(self.bin_returns, self.g_max)
        self.bin_past_weights *= PAST_DECAY
        self.bin_past_weights += PAST_GAIN * soft
        past = self.bin_past_weights / np.sum(self.bin_past_weights)
        probabilities = UNIFORM_SHARE / self.num_bins + SOFTMAX_SHARE * soft + PAST_SHARE * past
        self.probabilities = (probabilities / np.sum(probabilities)).astype(np.float32)
        term.bin_cdf[:] = np.cumsum(self.probabilities, dtype=np.float32)
        term.bin_cdf[-1] = 1.0

    def metrics(self) -> dict[str, float]:
        """Summary statistics for the trainer's metrics curves."""
        probabilities = self.probabilities
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        seen = self.bin_seen
        return {
            "supermimic_sampling_entropy": entropy / float(np.log(max(self.num_bins, 2))),
            "supermimic_sampling_top1_prob": float(np.max(probabilities)),
            "supermimic_seen_bins": float(np.mean(seen)),
            "supermimic_bin_return_mean": (float(np.mean(self.bin_returns[seen])) if np.any(seen) else 0.0),
        }


__all__ = ["SuperMimicStartSampler"]
