import numpy as np
import pytest
import torch
import zai_sim

from arena.export_replays import record_episode
from arena.matches import play
from zai_env.config import config_dir, deep_merge, load_config
from zai_env.layout import ZOMBIE
from zai_rl.algos.common import EpisodeStats, save_checkpoint
from zai_rl.algos.mappo_zombie import MAPPOConfig, ZombieMAPPO
from zai_rl.bots import ScriptedHuman
from zai_rl.envs import ZombieTrainingEnv
from zai_rl.models import ZombieActor
from zai_rl.policy import RecurrentZombiePolicy
from zai_rl.train_zombie import catch_rates, wilson


def tiny_config(cap_s: float = 6.0):
    return load_config(
        config_dir() / "s0.yaml", overrides={"episode_cap_s": cap_s}, env_overrides=False
    )


def scripted_human(c, e, s):
    return ScriptedHuman(c, e, seed=s)


def test_zombie_sequence_matches_step_by_step_with_resets():
    torch.manual_seed(0)
    actor = ZombieActor(hidden=32)
    B, L = 6, 12
    obs = torch.randn(B, L, zai_sim.ZOMBIE_OBS_DIM)
    starts = torch.zeros(B, L, dtype=torch.bool)
    starts[1, 4] = starts[3, 7] = starts[0, 0] = starts[5, 7] = True
    h0 = torch.randn(B, 32)
    dist = actor.sequence(obs, h0, starts)

    h = h0
    logits = []
    for t in range(L):
        h = h * (~starts[:, t]).unsqueeze(1)
        d, h = actor.step(obs[:, t], h)
        logits.append(d.logits)
    expected = torch.stack(logits, dim=1).reshape(B * L, -1)
    torch.testing.assert_close(dist.logits, expected, rtol=1e-4, atol=1e-5)


def test_flat_agent_axis_matches_the_groups():
    env = ZombieTrainingEnv(tiny_config(), [2, 3], 5, scripted_human, num_threads=1)
    # Group sizes 3 and 2: agents are (env 0..2) x 2 zombies, then (env 3..4) x 3 zombies.
    assert env.agent_env.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
    assert env.num_agents == 12
    batch = env.reset()
    assert batch.obs.shape == (12, zai_sim.ZOMBIE_OBS_DIM)
    assert batch.state.shape == (5, zai_sim.STATE_DIM)
    # The `num_zombies` feature tells each agent its group's count.
    num = ZOMBIE.scalar(batch.obs, "self", "num_zombies")
    assert len(set(num[:6].tolist())) == 1 and len(set(num[6:].tolist())) == 1
    assert num[0] != num[6]

    rng = np.random.default_rng(0)
    for _ in range(20):
        step = env.step(rng.integers(0, zai_sim.MOVE_ACTIONS, env.num_agents))
        assert step.reward.shape == (12,)
        assert step.final_obs.shape == (12, zai_sim.ZOMBIE_OBS_DIM)
        assert step.terminated.shape == (5,)


def test_zombie_rewards_follow_the_shaping_of_each_agent():
    env = ZombieTrainingEnv(tiny_config(), [3], 2, scripted_human, num_threads=1)
    env.reset()
    step = env.step(np.full(env.num_agents, zai_sim.MOVE_ACTIONS - 1))
    # Per-agent distance shaping: zombies of the same env get different rewards.
    r = step.reward.reshape(2, 3)
    assert not np.allclose(r[:, 0:1], r)


def mappo(num_envs=4, **kw):
    cfg = MAPPOConfig(
        num_envs=num_envs, rollout=16, bptt=8, epochs=2, minibatches=2, hidden=32, **kw
    )
    return ZombieMAPPO(cfg, torch.device("cpu"))


def test_one_training_iteration_bootstraps_truncation_and_round_trips(tmp_path):
    env = ZombieTrainingEnv(tiny_config(), [2, 3], 4, scripted_human, num_threads=1)
    algo = mappo()
    batch = env.reset()
    h = algo.actor.initial_state(env.num_agents)
    start = torch.ones(env.num_agents, dtype=torch.bool)
    stats = EpisodeStats()
    rollout, batch, h, start = algo.collect(env, batch, h, start, np.zeros(env.num_agents), stats)
    A = env.num_agents
    assert rollout["obs"].shape == (16, A, zai_sim.ZOMBIE_OBS_DIM)
    assert rollout["state"].shape == (16, 4, zai_sim.STATE_DIM)
    assert rollout["h0"].shape == (2, A, 32)
    assert len(stats.returns) >= 4, "6 s episodes end within 16 decisions"
    # Episodes end on the same step for every zombie of an env.
    dones = rollout["dones"].numpy()
    for e in range(4):
        agents = env.agent_env == e
        assert (dones[:, agents] == dones[:, agents][:, :1]).all()
    out = algo.update(rollout, progress=0.5)
    assert all(np.isfinite(v) for v in out.values())

    path = tmp_path / "latest.pt"
    save_checkpoint(path, algo, {"iteration": 1, "steps": 64, "run_id": "t"})
    cfg = tiny_config()
    res = play(
        cfg,
        scripted_human,
        lambda c, e, n, s: RecurrentZombiePolicy.from_checkpoint(path, e, n),
        episodes=4,
    )
    assert res.episodes == 4


def test_truncation_reward_includes_the_bootstrapped_value(monkeypatch):
    # 0.5 s episodes truncate after one decision for every env.
    env = ZombieTrainingEnv(tiny_config(cap_s=0.5), [2], 2, scripted_human, num_threads=1)
    algo = ZombieMAPPO(MAPPOConfig(num_envs=2, rollout=2, bptt=1, minibatches=1, hidden=32), "cpu")
    monkeypatch.setattr(algo, "value", lambda state, obs: torch.full((len(obs),), 10.0))
    batch = env.reset()
    rollout, *_ = algo.collect(
        env,
        batch,
        algo.actor.initial_state(4),
        torch.ones(4, dtype=torch.bool),
        np.zeros(4),
        EpisodeStats(),
    )
    assert rollout["dones"].all()
    # Rewards are at most ~1 without the bootstrap; gamma * 10 is folded in on truncation.
    assert (rollout["rewards"] > 9.0).all()


def test_policy_reset_clears_every_zombie_of_finished_envs():
    policy = RecurrentZombiePolicy(ZombieActor(hidden=16), num_envs=3, num_zombies=2)
    policy.act(np.random.default_rng(0).normal(size=(3, 2, zai_sim.ZOMBIE_OBS_DIM)).astype("f4"))
    policy.reset(np.array([False, True, False]))
    norms = policy.h.norm(dim=1).reshape(3, 2)
    assert (norms[1] == 0).all() and (norms[[0, 2]] > 0).all()


def test_zombie_policy_episode_re_simulates_as_a_replay():
    import json

    cfg = deep_merge(tiny_config(cap_s=20.0), {"zombie_count": 2})
    zombies = RecurrentZombiePolicy(ZombieActor(hidden=16), 1, 2)
    ep = record_episode(cfg, 5, ScriptedHuman(cfg, 1, seed=1), zombies)
    env = zai_sim.Env(json.dumps(ep["config"]), ep["seed"])
    for a in ep["actions"]:
        env.step(a[0], a[1], a[2:])
    assert f"{env.state_hash():016x}" == ep["final_hash"]


def test_catch_rates_pool_counts_with_a_wilson_interval():
    out = catch_rates(
        tiny_config(),
        [2, 3],
        scripted_human,
        lambda c, e, n, s: RecurrentZombiePolicy(ZombieActor(hidden=16), e, n),
        episodes=4,
        seed=0,
        prefix="eval",
    )
    for key in ("eval/caught_z2", "eval/caught_z3", "eval/caught", "eval/caught_lo"):
        assert 0.0 <= out[key] <= 1.0
    assert out["eval/caught_lo"] <= out["eval/caught"] <= out["eval/caught_hi"]


def test_wilson_interval():
    lo, hi = wilson(0.1, 768)
    assert lo == pytest.approx(0.080, abs=2e-3) and hi == pytest.approx(0.124, abs=2e-3)
    assert wilson(0.0, 10)[0] == pytest.approx(0.0, abs=1e-12)
