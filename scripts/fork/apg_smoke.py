# Copyright (C) 2026 Motphys Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

import torch
from motrix_rl_builtin.apg.bridge import GenCoordLayout, TransitionAssembler
from motrix_rl_builtin.apg.collector import Collector
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.learner import Learner

from motrix_env_core.manager.env import ManagerEnv
from motrix_envs.locomotion.wbt.g1.supermimic import G1SuperMimicWbtEnvCfg
from motrix_rl.replay import SequenceReplay

torch.set_num_threads(2)
torch.manual_seed(7)
env_cfg = G1SuperMimicWbtEnvCfg(motion_files=(sys.argv[1],))
env = ManagerEnv(env_cfg, num_envs=4, seed=7, capture_final_state=True)
state = env.init_state()
view = env.modeling_view()
layout = GenCoordLayout(env, view, "cpu")
cfg = ApgCfg()
a = cfg.agent
a.compile = False
a.amp = False
a.actor_hidden_dims = (16, 16, 16)
a.dynamics_hidden_dims = (16, 16, 16)
a.critic_hidden_dims = (16, 16)
a.latent_dim = 4
a.history_length = 2
a.batch_size = 4
a.horizon = 2
a.critic_epochs = 1
a.critic_minibatches = 1
a.world_model.batch_size = 4
a.world_model.horizon = 2
a.world_model.num_updates = 1
learner = Learner(env, view, layout, cfg, "cpu", seed=7)
assembler = TransitionAssembler(env, view, learner.proprio_dim)
collector = Collector(
    env, state, learner.actor, learner.policy, assembler, cfg, "cpu", observation_layout=learner.observation_layout
)
replay = SequenceReplay(schema=assembler.schema, num_envs=4, capacity=16, device="cpu", seed=7, compile=False)
for _ in range(5):
    collection = collector.collect()
    replay.extend(collection.transition)
metrics = learner.update(replay)
assert all(bool(torch.isfinite(v).all()) for v in metrics.values())
print(
    "PASS: MotrixSim 1.0, SuperMimic 4 lanes, 5 collections, Muon world-model/actor + critic update; finite metrics",
    metrics.keys(),
)

checkpoint = learner.state_dict()
learner.load_state_dict(checkpoint)
print("PASS: training-state resume dimensions and optimizer state")
