import numpy as np
import pytest
import torch
import zai_sim

from arena.matches import play
from zai_env.config import config_dir, load_config
from zai_rl.algos.ppo_human import (
    EpisodeStats,
    HumanPPO,
    PPOConfig,
    RunningMeanStd,
    gae,
    save_checkpoint,
)
from zai_rl.bots import ScriptedZombies
from zai_rl.envs import HumanTrainingEnv
from zai_rl.models import HumanActor
from zai_rl.policy import RecurrentHumanPolicy


def test_gae_stops_at_episode_end():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    adv, ret = gae(rewards, values, dones, torch.tensor([10.0]), gamma=0.5, lam=1.0)
    # Step 1 ends the episode: no bootstrap from step 2 or the last value.
    assert ret[1].item() == pytest.approx(1.0)
    assert ret[0].item() == pytest.approx(1.0 + 0.5 * 1.0)
    assert ret[2].item() == pytest.approx(1.0 + 0.5 * 10.0)
    torch.testing.assert_close(adv, ret - values)


def test_running_mean_std_matches_numpy():
    rms = RunningMeanStd(eps=0.0)
    rms.count.fill_(0.0)
    rng = np.random.default_rng(0)
    chunks = [rng.normal(3, 2, 500) for _ in range(4)]
    for c in chunks:
        rms.update(torch.tensor(c, dtype=torch.float32))
    allv = np.concatenate(chunks)
    assert rms.mean.item() == pytest.approx(allv.mean(), rel=1e-4)
    assert rms.var.item() == pytest.approx(allv.var(), rel=1e-3)


def test_actor_respects_action_mask():
    actor = HumanActor(hidden=32)
    obs = torch.randn(64, zai_sim.HUMAN_OBS_DIM)
    mask = torch.ones(64, 20, dtype=torch.bool)
    mask[:, 17 + 1] = False  # pickup invalid
    mask[:, 17 + 2] = False  # close door invalid
    move, interact, h = actor.step(obs, actor.initial_state(64), mask)
    assert h.shape == (64, 32)
    assert (interact.sample() == 0).all()
    assert move.probs.shape == (64, 17)


def tiny_config():
    return load_config(
        config_dir() / "s0.yaml", overrides={"episode_cap_s": 6.0}, env_overrides=False
    )


def test_one_training_iteration_and_checkpoint_round_trip(tmp_path):
    cfg = tiny_config()
    ppo_cfg = PPOConfig(num_envs=4, rollout=16, bptt=8, epochs=2, minibatches=2, hidden=32)
    env = HumanTrainingEnv(
        cfg, [2, 3], 4, lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s), num_threads=1
    )
    algo = HumanPPO(ppo_cfg, torch.device("cpu"))
    batch = env.reset()
    h = algo.actor.initial_state(4)
    start = torch.ones(4, dtype=torch.bool)
    stats = EpisodeStats()
    rollout, batch, h, start = algo.collect(env, batch, h, start, np.zeros(4), stats)
    assert rollout["obs"].shape == (16, 4, zai_sim.HUMAN_OBS_DIM)
    assert rollout["h0"].shape == (2, 4, 32)
    assert len(stats.returns) >= 4, "6 s episodes end within 16 decisions"
    out = algo.update(rollout, progress=0.5)
    assert all(np.isfinite(v) for v in out.values())

    path = tmp_path / "latest.pt"
    save_checkpoint(path, algo, {"iteration": 1, "steps": 64, "run_id": "t"})
    policy = RecurrentHumanPolicy.from_checkpoint(path, num_envs=4)
    res = play(
        cfg,
        lambda c, e, s: policy,
        lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s),
        episodes=4,
    )
    assert res.episodes == 4


def test_ppo_config_validates_shapes():
    with pytest.raises(ValueError):
        PPOConfig(rollout=100, bptt=64)


def test_sequence_matches_step_by_step_with_resets():
    torch.manual_seed(0)
    actor = HumanActor(hidden=32)
    B, L = 5, 12
    obs = torch.randn(B, L, zai_sim.HUMAN_OBS_DIM)
    mask = torch.ones(B, L, 20, dtype=torch.bool)
    starts = torch.zeros(B, L, dtype=torch.bool)
    starts[1, 4] = starts[3, 7] = starts[0, 0] = True
    h0 = torch.randn(B, 32)
    move, interact = actor.sequence(obs, h0, starts, mask)

    h = h0
    logits = []
    for t in range(L):
        h = h * (~starts[:, t]).unsqueeze(1)
        m, _, h = actor.step(obs[:, t], h, mask[:, t])
        logits.append(m.logits)
    expected = torch.stack(logits, dim=1).reshape(B * L, -1)
    torch.testing.assert_close(move.logits, expected, rtol=1e-4, atol=1e-5)
