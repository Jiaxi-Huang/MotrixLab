# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SuperMimic world-model integration: real env, real MDP, learner loop.

Covers the NumPy env contract (buffers, randomization context), the plugin's
torch MDP built from those surfaces (kinematics vs MotrixSim FK; goal /
proprio / reward parity with the manager kernels), and the learner / trainer
loop with checkpointing.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from motrix_rl_builtin.supermimic.config import SuperMimicCfg
from motrix_rl_builtin.supermimic.contracts import SupportsSuperMimic
from motrix_rl_builtin.supermimic.learner import SuperMimicLearner
from motrix_rl_builtin.supermimic.mdp import build_supermimic_mdp

from motrix_envs.locomotion.supermimic.g1 import SuperMimicManagerEnv, make_g1_supermimic_cfg
from motrix_envs.locomotion.supermimic.mdp import SuperMimicMotionCommand

_DANCE_MOTION = (
    Path(__file__).resolve().parents[1]
    / "motrix_envs"
    / "src"
    / "motrix_envs"
    / "locomotion"
    / "wbt"
    / "assets"
    / "motion"
    / "g1"
    / "dance"
    / "dance1_subject2.npz"
)

pytestmark = [pytest.mark.numba, pytest.mark.integration]


def _corpus(tmp_path, lengths=(160, 140)) -> tuple[str, ...]:
    with np.load(_DANCE_MOTION, allow_pickle=False) as data:
        total = int(np.asarray(data["num_frames"]).reshape(-1)[0])
        fields = {key: data[key] for key in data.files}
    paths = []
    start = 0
    for index, length in enumerate(lengths):
        part = {"num_frames": np.int32(length)}
        for name, value in fields.items():
            if name == "num_frames":
                continue
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                part[name] = value[start : start + length]
            else:
                part[name] = value
        path = tmp_path / f"corpus{index}.npz"
        np.savez(path, **part)
        paths.append(str(path))
        start += length
    return tuple(paths)


def _learner_cfg(num_envs: int) -> SuperMimicCfg:
    cfg = SuperMimicCfg()
    cfg.device = "cpu"
    agent = cfg.agent
    agent.buffer_steps = 24
    agent.utd = 0.5
    agent.horizon = 4
    agent.goal_frames = 2
    agent.hist = 2
    agent.z_dim = 8
    agent.wm_hidden = [32, 32]
    agent.policy_hidden = [32, 32]
    agent.critic_hidden = [16, 16]
    agent.wm_window = 2
    agent.wm_batches = 1
    agent.wm_batch = 16
    agent.pi_batch = 8
    agent.lr = 1e-3
    agent.gamma = 0.99
    agent.lam = 0.95
    agent.action_noise = 0.1
    agent.critic_epochs = 1
    agent.critic_minibatches = 2
    agent.warmup_steps = 1
    agent.wm_only_iters = 0
    agent.lr_decay_iters = 100
    agent.amp = "none"
    agent.compile = False
    return cfg


def test_supermimic_env_contract_and_torch_mdp_dimensions(tmp_path):
    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(_corpus(tmp_path)), num_envs=2, seed=1)
    assert isinstance(env, SupportsSuperMimic)
    motion = env.command_terms["motion"]
    assert isinstance(motion, SuperMimicMotionCommand)
    assert motion.state.shape == (2, 71)
    assert motion.context.shape == (2, 4)

    mdp = build_supermimic_mdp(env)
    assert mdp.num_joints == 29
    assert mdp.context_dim == 4
    assert mdp.obs_dim == 99
    assert mdp.goal_frame_dim == 73
    assert mdp.state_dim == 71
    assert mdp.max_step_reward == 5.0


def test_supermimic_torch_mdp_matches_env_outputs(tmp_path):
    """Goal, proprio, and reward features match the manager kernels."""
    cfg = make_g1_supermimic_cfg(_corpus(tmp_path, lengths=(40, 35)))
    cfg.commands.motion.num_goal_frames = 4
    env = SuperMimicManagerEnv(cfg, num_envs=3, seed=5)
    env.init_state()
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (3, env.action_space.shape[0])).astype(np.float32)
    previous = np.zeros_like(actions)
    env.step(previous)
    result = env.step(actions)

    motion = env.command_terms["motion"]
    mdp = build_supermimic_mdp(env)
    frames = torch.tensor(motion.steps[:, 0], dtype=torch.long)
    unflattened = mdp.unflatten_state(torch.tensor(motion.state))

    # Done lanes reset before the returned observation; compare live episodes.
    live = ~(np.asarray(result.terminated) | np.asarray(result.truncated))
    assert bool(live.any())

    def slice_of(group: str, name: str):
        layout = env.manager_layout.observations[group]
        return next(term for term in layout.terms if term.name == name).output_slice

    # Goal features equal the manager goal observation term exactly.
    policy_goal = next(
        term for term in env.manager_layout.observations["policy"].terms if term.name == "goal"
    ).output_slice
    torch_goal = mdp.goal(frames, 4).detach().numpy()
    for lane in np.flatnonzero(live):
        np.testing.assert_allclose(torch_goal[lane], result.obs.policy[lane][policy_goal], atol=2e-4)

    # Proprio blocks equal the noise-free value-group slices.
    proprio = mdp.obs(unflattened, torch.tensor(actions), frames).detach().numpy()
    for name in ("dof_pos", "dof_vel", "base_ang_vel", "gravity", "actions"):
        obs_slice = slice_of("value", name)
        np.testing.assert_allclose(
            proprio[live][:, obs_slice], result.obs.value[live][:, obs_slice], atol=2e-3, err_msg=name
        )

    # Analytic reward tracks the manager reward on the same state.
    reward = mdp.reward(unflattened, frames, torch.tensor(actions), torch.tensor(previous))
    np.testing.assert_allclose(reward.detach().numpy()[live], result.reward[live], atol=0.02)

    # Imagination-start validity follows the owning clip's end.
    valid = mdp.valid_imagination_start(frames, 32)
    ends = motion.clip.frame_clip_end[frames.numpy()]
    np.testing.assert_array_equal(valid.numpy(), (frames.numpy() + 32) <= ends)


def test_supermimic_kinematics_matches_motrixsim(tmp_path):
    """The differentiable FK reproduces MotrixSim link poses on random states."""
    import motrixsim as mtx
    from motrix_rl_builtin.supermimic.mdp import SuperMimicKinematics, quat_xyzw_to_mat
    from motrix_robots.unitree import UNITREE_G1_ASSET_DIR
    from scipy.spatial.transform import Rotation as R

    from motrix_envs.locomotion.wbt.g1.common import G1_TRACKED_BODY_NAMES

    model_file = UNITREE_G1_ASSET_DIR / "g1_29dof.xml"
    model = mtx.load_model(str(model_file))
    joint_names = [str(n) for n in model.joint_names]
    kin = SuperMimicKinematics(str(model_file), joint_names, G1_TRACKED_BODY_NAMES)

    rng = np.random.default_rng(3)
    batch = 6
    pos = rng.uniform(-1, 1, (batch, 3)).astype(np.float32)
    pos[:, 2] += 1.0
    quats = R.random(batch, random_state=11).as_quat().astype(np.float32)
    q = rng.uniform(-0.4, 0.4, (batch, len(joint_names))).astype(np.float32)
    qd = (rng.standard_normal((batch, len(joint_names))) * 0.5).astype(np.float32)
    v = rng.standard_normal((batch, 3)).astype(np.float32)
    w = rng.standard_normal((batch, 3)).astype(np.float32)

    body_p, body_R, _, _ = kin.forward(
        torch.tensor(pos),
        quat_xyzw_to_mat(torch.tensor(quats)),
        torch.tensor(v),
        torch.tensor(w),
        torch.tensor(q),
        torch.tensor(qd),
    )

    data = mtx.SceneData(model, batch=[batch])
    data.set_dof_pos(np.concatenate([pos, quats, q], 1).astype(np.float32), model)
    data.set_dof_vel(np.concatenate([v, w, qd], 1).astype(np.float32))
    model.forward_kinematic(data)
    poses = np.asarray(model.get_link_poses(data))
    link_index = {str(n): i for i, n in enumerate(model.link_names)}
    tracked = [link_index[name] for name in G1_TRACKED_BODY_NAMES]

    assert float(np.abs(body_p.detach().numpy() - poses[:, tracked, :3]).max()) < 1e-4
    reference_R = quat_xyzw_to_mat(torch.tensor(poses[:, tracked, 3:7])).numpy()
    cosine = np.clip((np.einsum("tkij,tkij->tk", reference_R, body_R.detach().numpy()) - 1) / 2, -1, 1)
    assert float(np.arccos(cosine).max()) < 2e-3


def test_supermimic_learner_runs_on_real_env(tmp_path):
    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(_corpus(tmp_path)), num_envs=4, seed=1)
    cfg = _learner_cfg(env.num_envs)
    learner = SuperMimicLearner(
        env,
        cfg.agent,
        build_supermimic_mdp(env),
        device=torch.device("cpu"),
        action_low=np.full(29, -1.0, np.float32),
        action_high=np.full(29, 1.0, np.float32),
        seed=0,
    )
    learner.collect(6)
    assert learner.buffer.size == 6
    assert np.all(np.isfinite(learner.motion.state))
    wm_loss, wm_error = learner.wm_update()
    assert torch.isfinite(wm_loss)
    assert torch.isfinite(wm_error)
    pi_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()
    assert torch.isfinite(pi_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(imagined_reward)
    assert torch.isfinite(survival)

    checkpoint = learner.state_dict()
    assert set(checkpoint) >= {"wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v"}


def test_supermimic_trainer_runs_two_iterations(tmp_path, monkeypatch):
    from motrix_rl_builtin.supermimic.trainer import Trainer

    from motrix_rl.config import CheckpointConfig, LoggingConfig
    from motrix_rl.contracts import TrainerContext

    corpus = _corpus(tmp_path, lengths=(200, 180))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 2
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "run" / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )

    Trainer(context=context).train()

    final = tmp_path / "run" / "checkpoints" / "latest.pt"
    assert final.is_file()
    assert (tmp_path / "run" / "checkpoints" / "manifest.json").is_file()
    checkpoint = torch.load(final, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert set(checkpoint) >= {"wm", "pi", "critic"}


def test_supermimic_per_clip_evaluation(tmp_path, monkeypatch):
    """Train briefly, then run deterministic per-clip evaluation twice.

    The trainer checkpoint carries the policy hyperparameters; the evaluator
    pins one lane per clip to its first frame, and repeated evaluation of the
    same checkpoint is bit-stable.
    """
    from motrix_rl_builtin.supermimic.evaluate import evaluate, load_policy
    from motrix_rl_builtin.supermimic.trainer import Trainer

    from motrix_rl.config import CheckpointConfig, LoggingConfig
    from motrix_rl.contracts import TrainerContext

    corpus = _corpus(tmp_path, lengths=(120, 100))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 1
    run_dir = tmp_path / "run"
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )
    Trainer(context=context).train()
    checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["args"]["goal_frames"] == cfg.agent.goal_frames

    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(corpus).for_eval(), num_envs=2, seed=3)
    mdp = build_supermimic_mdp(env)
    policy = load_policy(mdp, checkpoint, torch.device("cpu"))
    names = ["corpus0", "corpus1"]
    first = evaluate(
        env,
        policy,
        mdp,
        goal_frames=cfg.agent.goal_frames,
        hist=cfg.agent.hist,
        device=torch.device("cpu"),
        clip_names=names,
    )
    second = evaluate(
        env,
        policy,
        mdp,
        goal_frames=cfg.agent.goal_frames,
        hist=cfg.agent.hist,
        device=torch.device("cpu"),
        clip_names=names,
    )

    assert len(first["per_clip"]) == 2
    assert {entry["clip"] for entry in first["per_clip"].values()} == {"corpus0", "corpus1"}
    for entry in first["per_clip"].values():
        assert entry["length"] in (120, 100)
        assert 0 <= entry["steps"] <= entry["length"]
    assert first["norm_return"] <= 1.0 + 1e-6
    assert 0 <= first["n_failed"] <= 2
    # Physics-level warm-start differences across repeated rollouts keep the
    # comparison at solver tolerance rather than bit equality.
    for key in ("norm_return", "norm_return_framew"):
        assert first[key] == pytest.approx(second[key], abs=2e-3)
    for index in first["per_clip"]:
        assert first["per_clip"][index]["steps"] == second["per_clip"][index]["steps"]
        assert first["per_clip"][index]["failed"] == second["per_clip"][index]["failed"]


def test_supermimic_eval_script_runs(tmp_path, monkeypatch, capsys):
    """The repo eval script resolves a run's checkpoint and writes json/csv."""
    import importlib.util

    from motrix_rl_builtin.supermimic.trainer import Trainer

    from motrix_rl.config import CheckpointConfig, LoggingConfig
    from motrix_rl.contracts import TrainerContext

    corpus = _corpus(tmp_path, lengths=(80, 70))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 1
    run_dir = tmp_path / "run"
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )
    Trainer(context=context).train()

    spec = importlib.util.spec_from_file_location(
        "eval_supermimic", Path(__file__).resolve().parents[1] / "scripts" / "eval_supermimic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--run_dir", str(run_dir), "--device", "cpu"])

    out = capsys.readouterr().out
    assert "norm_return" in out
    assert (run_dir / "checkpoints" / "latest_eval.json").is_file()
    assert (run_dir / "checkpoints" / "eval.csv").is_file()
    with (run_dir / "checkpoints" / "latest_eval.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    assert set(result["failed"]) <= {"corpus0", "corpus1"}
    assert len(result["per_clip"]) == 2
