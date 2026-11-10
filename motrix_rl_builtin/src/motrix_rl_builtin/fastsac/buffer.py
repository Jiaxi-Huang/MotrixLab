# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Replay buffer for FastSAC.

Ported from holosoma's ``fast_sac_utils`` with asymmetric observations: each
transition stores both the actor observation and the (privileged) critic
observation. The observation normalizer is shared infrastructure living in
:mod:`motrix_rl.normalize` (see the re-export at the bottom).
"""

from __future__ import annotations

import torch
from torch import nn

# The observation normalizer now lives in ``motrix_rl`` as the shared
# NORM_KEYS-conformant implementation; re-exported here for the established
# ``fastsac.buffer.EmpiricalNormalization`` import path (also unpickles
# checkpoints saved while the class lived here).
from motrix_rl.normalize import EmpiricalNormalization as EmpiricalNormalization


class SimpleReplayBuffer(nn.Module):
    """Per-environment circular replay buffer with n-step returns and
    asymmetric (actor / critic) observations.

    Observations are stored once per time step in a ``(n_env, buffer_size + 1,
    dim)`` sequence: slot ``t % (buffer_size + 1)`` holds the observation of
    step ``t`` and doubles as the next-observation of step ``t - 1``. With
    auto-reset envs the observation returned by step ``t`` is exactly the
    observation stored for step ``t + 1`` (at episode ends it is the reset
    observation), the invariant already relied upon by the async transition
    ring, so a dedicated ``next_observations`` copy would be pure redundancy
    and doubles GPU memory for high-dimensional observation tasks.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        final_obs_bootstrap: bool = False,
    ):
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        # All fields share the same capacity so one absolute-time indexing
        # scheme (t % cap) covers them; the extra slot over buffer_size keeps
        # the newest transition's next-observation slot distinct from its own.
        # The scalar fields' extra slot is negligible and NOT independently
        # shrinkable — a different modulus would break the shared indexing.
        cap = buffer_size + 1
        self._cap = cap
        z = lambda d: torch.zeros((n_env, cap, d), device=device, dtype=torch.float)  # noqa: E731
        self.observations = z(n_obs)
        self.actions = z(n_act)
        self.rewards = torch.zeros((n_env, cap), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.critic_observations = z(n_critic_obs)
        self.ptr = 0

        # Final rows are allocated only for live truncated transitions. Capacity
        # grows geometrically, bounded by the replay's transition capacity; no
        # assumption about time between truncations is required. Allocation and
        # ownership bookkeeping are eager ingest work, outside compiled updates.
        self._final_obs_table = None
        self._final_critic_table = None
        self._final_slot = None
        self._final_capacity = 0
        self._final_free: list[int] = []
        self._final_owners: list[dict[int, int]] = [{} for _ in range(cap)]
        if final_obs_bootstrap:
            self._enable_final_storage()

    def _enable_final_storage(self) -> None:
        self._final_slot = torch.full((self.n_env, self._cap), -1, device=self.device, dtype=torch.int32)
        self._grow_final_storage(self.n_env)

    def _grow_final_storage(self, needed: int) -> None:
        old = self._final_capacity
        capacity = min(self.n_env * self._cap, max(self.n_env, old * 2, old + needed))
        # Tables may be referenced by work on the compute and ingest streams.
        # Growth is rare and eager; finish those uses before replacing backing.
        if self.observations.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        policy = torch.empty((capacity, self.n_obs), device=self.device)
        critic = torch.empty((capacity, self.n_critic_obs), device=self.device)
        if old:
            policy[:old].copy_(self._final_obs_table)
            critic[:old].copy_(self._final_critic_table)
        else:
            # Samples without a final row still gather index zero before masking.
            policy[0].zero_()
            critic[0].zero_()
        self._final_obs_table = policy
        self._final_critic_table = critic
        self._final_free.extend(range(capacity - 1, old - 1, -1))
        self._final_capacity = capacity

    @property
    def num_stored(self) -> int:
        # A transition is only complete once the following step's observation
        # has been written (its next observation), so ingestion runs one step
        # ahead of what is sampleable.
        return min(max(self.ptr - 1, 0), self.buffer_size)

    def extend(self, obs, critic_obs, actions, rewards, dones, truncations) -> None:
        """Append one transition per environment.

        Only the observation of step ``ptr`` is written. Its next-observation
        slot (``ptr + 1``) is filled by the following call — with auto-reset
        envs the observation returned by step ``t`` is exactly the observation
        of step ``t + 1`` (at episode ends it is the reset observation), the
        invariant already relied upon by the async transition ring.
        """
        slot = self.ptr % self._cap
        self._invalidate_final_slot(slot, 1)
        self.observations[:, slot] = obs
        self.critic_observations[:, slot] = critic_obs
        self.actions[:, slot] = actions
        self.rewards[:, slot] = rewards
        self.dones[:, slot] = dones
        self.truncations[:, slot] = truncations
        self.ptr += 1

    def extend_batch(self, obs, critic_obs, actions, rewards, dones, truncations, *, final_payloads=None) -> None:
        """Append ``k`` transitions per environment from slot-major tensors.

        Inputs are shaped ``(k, n_env, dim)`` for obs / critic_obs / actions
        and ``(k, n_env)`` for the scalar fields — the contiguous multi-slot
        layout produced by the async transition ring's :meth:`read_span`.
        Writes go directly into the strided buffer slices (one copy per field
        per contiguous chunk instead of one per slot) on the caller's current
        stream, so an async ingest can issue non-blocking pinned H2D copies on
        a side stream.
        """
        k = obs.shape[0]
        if k > self._cap:
            # Preserve only the newest replay-capacity transitions, but keep
            # absolute time progression identical to successive extends.
            skip = k - self._cap
            self.ptr += skip
            self.extend_batch(
                *(v[skip:] for v in (obs, critic_obs, actions, rewards, dones, truncations)),
                final_payloads=None if final_payloads is None else final_payloads[skip:],
            )
            return
        start = self.ptr % self._cap
        self._invalidate_final_slot(start, k)
        n = min(k, self._cap - start)
        self._write_span(start, (t[:n] for t in (obs, critic_obs, actions, rewards, dones, truncations)))
        if k > n:
            self._write_span(0, (t[n:] for t in (obs, critic_obs, actions, rewards, dones, truncations)))
        self.ptr += k
        if final_payloads is not None:
            for offset, payload in enumerate(final_payloads):
                if payload is not None:
                    self.write_final(payload.env_ids, self.ptr - k + offset, payload.obs, payload.critic_obs)

    def _invalidate_final_slot(self, start: int, count: int) -> None:
        """Clear the final-obs grid for slots about to be overwritten.

        The transitions being written carry no terminal observation (yet — a
        ``write_final`` for exactly these slots may follow the ingest); stale
        pointers from a full ring cycle earlier must not survive."""
        if self._final_slot is None:
            return
        for offset in range(count):
            owners = self._final_owners[(start + offset) % self._cap]
            self._final_free.extend(owners.values())
            owners.clear()
        n = min(count, self._cap - start)
        self._final_slot[:, start : start + n] = -1
        if count > n:
            self._final_slot[:, : count - n] = -1

    @torch.no_grad()
    def write_final(self, env_ids: torch.Tensor, t_abs: int, obs_rows: torch.Tensor, critic_rows: torch.Tensor) -> None:
        """Attach owned final rows to an already-ingested transition.

        Host allocation tracks live references, not episode frequency. Device
        tables are read by eager sampling, outside compiled gradient updates.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long).cpu().tolist()
        if not ids:
            return
        slot = t_abs % self._cap
        owners = self._final_owners[slot]
        for env_id in ids:
            previous = owners.pop(env_id, None)
            if previous is not None:
                self._final_free.append(previous)
        if len(self._final_free) < len(ids):
            self._grow_final_storage(len(ids) - len(self._final_free))
        rows = [self._final_free.pop() for _ in ids]
        owners.update(zip(ids, rows))
        device_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        row_ids = torch.tensor(rows, device=self.device, dtype=torch.long)
        self._final_obs_table[row_ids] = obs_rows.to(self.device, torch.float)
        self._final_critic_table[row_ids] = critic_rows.to(self.device, torch.float)
        self._final_slot[device_ids, slot] = row_ids.to(torch.int32)

    def _write_span(self, slot: int, fields) -> None:
        obs, critic_obs, actions, rewards, dones, truncations = fields
        n = obs.shape[0]
        end = slot + n
        self.observations[:, slot:end].copy_(obs.permute(1, 0, 2), non_blocking=True)
        self.critic_observations[:, slot:end].copy_(critic_obs.permute(1, 0, 2), non_blocking=True)
        self.actions[:, slot:end].copy_(actions.permute(1, 0, 2), non_blocking=True)
        self.rewards[:, slot:end].copy_(rewards.permute(1, 0), non_blocking=True)
        self.dones[:, slot:end].copy_(dones.permute(1, 0), non_blocking=True)
        self.truncations[:, slot:end].copy_(truncations.permute(1, 0), non_blocking=True)

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict:
        if self.num_stored == 0:
            raise RuntimeError("cannot sample from a replay buffer with no complete transitions")
        n_env, no, na, nco = self.n_env, self.n_obs, self.n_act, self.n_critic_obs
        flat = n_env * batch_size

        if self.n_steps == 1:
            idx = torch.randint(0, self.num_stored, (n_env, batch_size), device=self.device)
            # Sampled ring positions are expressed as absolute time steps over
            # the contiguous valid window, then folded into the ring. The next
            # observation of step ``t`` is the observation stored at ``t + 1``.
            t = self.ptr - 1 - self.num_stored + idx
            ring = t % self._cap
            next_ring = (t + 1) % self._cap
            oi = ring.unsqueeze(-1).expand(-1, -1, no)
            ni = next_ring.unsqueeze(-1).expand(-1, -1, no)
            ai = ring.unsqueeze(-1).expand(-1, -1, na)
            ci = ring.unsqueeze(-1).expand(-1, -1, nco)
            nci = next_ring.unsqueeze(-1).expand(-1, -1, nco)
            out = {
                "obs": torch.gather(self.observations, 1, oi).reshape(flat, no),
                "next_obs": torch.gather(self.observations, 1, ni).reshape(flat, no),
                "critic_obs": torch.gather(self.critic_observations, 1, ci).reshape(flat, nco),
                "next_critic_obs": torch.gather(self.critic_observations, 1, nci).reshape(flat, nco),
                "actions": torch.gather(self.actions, 1, ai).reshape(flat, na),
                "rewards": torch.gather(self.rewards, 1, ring).reshape(flat),
                "dones": torch.gather(self.dones, 1, ring).reshape(flat),
                "truncations": torch.gather(self.truncations, 1, ring).reshape(flat),
            }
            if self._final_slot is not None:
                # Time-limit-truncated transitions bootstrap from the stored
                # TERMINAL observation, not the successor slot's auto-reset
                # observation: the reset state's value has nothing to do with
                # the episode that just timed out.
                k_idx = self._final_slot.gather(1, ring.long())
                has_final = (k_idx >= 0) & (out["truncations"] > 0).view(n_env, batch_size)
                k_safe = k_idx.clamp(min=0).long()
                fin_obs = self._final_obs_table[k_safe]
                fin_cobs = self._final_critic_table[k_safe]
                out["next_obs"] = torch.where(
                    has_final.unsqueeze(-1), fin_obs, out["next_obs"].view(n_env, batch_size, no)
                ).reshape(flat, no)
                out["next_critic_obs"] = torch.where(
                    has_final.unsqueeze(-1), fin_cobs, out["next_critic_obs"].view(n_env, batch_size, nco)
                ).reshape(flat, nco)
            out["effective_n_steps"] = torch.ones_like(out["dones"])
            return out

        # n-step (>1)
        if self.ptr - 1 < self.buffer_size:
            max_start = max(1, self.ptr - self.n_steps)
            idx = torch.randint(0, max_start, (n_env, batch_size), device=self.device)
            t = idx
        else:
            idx = torch.randint(0, self.buffer_size, (n_env, batch_size), device=self.device)
            t = self.ptr - 1 - self.buffer_size + idx
        ring = t % self._cap

        oi = ring.unsqueeze(-1).expand(-1, -1, no)
        ai = ring.unsqueeze(-1).expand(-1, -1, na)
        ci = ring.unsqueeze(-1).expand(-1, -1, nco)
        obs = torch.gather(self.observations, 1, oi).reshape(flat, no)
        actions = torch.gather(self.actions, 1, ai).reshape(flat, na)
        critic_obs = torch.gather(self.critic_observations, 1, ci).reshape(flat, nco)

        offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
        all_idx = (t.unsqueeze(-1) + offsets) % self._cap
        all_rew = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)
        all_done = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)
        all_trunc = torch.gather(self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)

        # A window must not read past the newest stored observation; folded
        # ring positions beyond it hold stale data from a full cycle earlier.
        max_off = (self.ptr - 2 - t).clamp(min=0)
        in_win = offsets <= max_off.unsqueeze(-1)
        all_rew = all_rew * in_win
        all_done = all_done * in_win
        all_trunc = all_trunc * in_win

        done_shift = torch.cat([torch.zeros_like(all_done[:, :, :1]), all_done[:, :, :-1]], dim=2)
        done_mask = torch.cumprod(1.0 - done_shift, dim=2) * in_win
        # Keep the same integer dtype as the 1-step branch's ones_like(dones).
        eff_n = done_mask.sum(2).long()
        discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
        n_step_rew = (all_rew * done_mask * discounts.view(1, 1, -1)).sum(dim=2)

        first_done = torch.argmax((all_done > 0).float(), dim=2)
        first_trunc = torch.argmax((all_trunc > 0).float(), dim=2)
        first_done = torch.where(all_done.sum(2) == 0, self.n_steps - 1, first_done)
        first_trunc = torch.where(all_trunc.sum(2) == 0, self.n_steps - 1, first_trunc)
        final = torch.minimum(torch.minimum(first_done, first_trunc), max_off)
        # The next observation of the final transition lives one slot ahead.
        final_next = (t + final + 1) % self._cap
        final_ring = (t + final) % self._cap

        next_obs = self.observations.gather(1, final_next.unsqueeze(-1).expand(-1, -1, no)).reshape(flat, no)
        next_critic_obs = self.critic_observations.gather(1, final_next.unsqueeze(-1).expand(-1, -1, nco)).reshape(
            flat, nco
        )
        dones = self.dones.gather(1, final_ring).reshape(flat)
        truncations = self.truncations.gather(1, final_ring).reshape(flat)
        if self._final_slot is not None:
            # Same terminal-observation correction as the 1-step branch,
            # applied at the window's final transition (the one that
            # bootstraps when the window ends on a truncation).
            k_idx = self._final_slot.gather(1, final_ring.long())
            has_final = (k_idx >= 0) & (truncations > 0).view(n_env, batch_size)
            k_safe = k_idx.clamp(min=0).long()
            fin_obs = self._final_obs_table[k_safe]
            fin_cobs = self._final_critic_table[k_safe]
            next_obs = torch.where(has_final.unsqueeze(-1), fin_obs, next_obs.view(n_env, batch_size, no)).reshape(
                flat, no
            )
            next_critic_obs = torch.where(
                has_final.unsqueeze(-1), fin_cobs, next_critic_obs.view(n_env, batch_size, nco)
            ).reshape(flat, nco)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "critic_obs": critic_obs,
            "next_critic_obs": next_critic_obs,
            "actions": actions,
            "rewards": n_step_rew.reshape(flat),
            "dones": dones,
            "truncations": truncations,
            "effective_n_steps": eff_n.reshape(flat),
        }
