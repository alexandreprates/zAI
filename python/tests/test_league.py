import sys

import numpy as np
import pytest
import torch
import yaml

from zai_env.config import config_dir, load_config
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED
from zai_rl.envs import MatchTrainingEnv, league_groups
from zai_rl.league.pool import Entry, Pool
from zai_rl.league.slots import OpponentSlots
from zai_rl.models import HumanActor, ZombieActor
from zai_rl.train_league import Guard, LeagueParams, main, save_snapshot


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(
        config_dir() / "s0_small.yaml", overrides={"episode_cap_s": 10.0}, env_overrides=False
    )


def test_groups_split_kinds_and_counts():
    groups = league_groups(16, [2, 3], (0.5, 0.25, 0.25))
    assert [(g.kind, g.zombie_count, g.num_envs) for g in groups] == [
        ("A", 2, 4),
        ("A", 3, 4),
        ("B", 2, 2),
        ("B", 3, 2),
        ("C", 2, 2),
        ("C", 3, 2),
    ]


def test_match_env_rows_round_trip(tiny_cfg):
    env = MatchTrainingEnv(tiny_cfg, league_groups(16, [2, 3], (0.5, 0.25, 0.25)), num_threads=1)
    assert env.num_agents == 4 * 2 + 4 * 3 + 2 * 2 + 2 * 3 + 2 * 2 + 2 * 3
    assert set(env.kind[env.human_rows]) == {"A", "B"}
    assert set(env.kind[env.zombie_envs]) == {"A", "C"}
    # Every learner agent points at its own env in the learner's env axis.
    np.testing.assert_array_equal(
        env.zombie_envs[env.zombie_agent_env], env.agent_env[env.zombie_agents]
    )
    assert set(env.kind[env.agent_env[env.opp_zombie_agents]]) == {"B"}
    obs = env.reset()
    step = env.step(np.zeros((16, 2), dtype=np.int64), np.full(env.num_agents, 16))
    hv, zv = env.human_view(step), env.zombie_view(step)
    np.testing.assert_array_equal(hv.next.obs, step.next.human_obs[env.human_rows])
    assert zv.next.obs.shape == (len(env.zombie_agents), obs.zombie_obs.shape[1])
    assert zv.terminated.shape == (len(env.zombie_envs),)


def test_pool_pfsp_weights_decay_eviction_and_round_trip(tmp_path):
    pool = Pool(cap=4, decay=0.5, pfsp="hard", uniform=0.0, protect_every=2)
    for i in range(4):
        pool.add(Entry(pool.next_id("zombie"), "zombie", f"z{i}.pt", source="league"))
    ids = [e.id for e in pool.entries["zombie"] if not e.scripted]
    assert len(pool.entries["zombie"]) == 4  # scripted + 3 left after evicting the oldest
    assert pool.get(ids[0]).protected or pool.get(ids[1]).protected  # every 2nd snapshot
    for _ in range(20):
        pool.record("human", ids[-1], 1.0)  # the human always beats the newest
    pool.end_iteration()
    w = pool.weights("human")
    assert w[ids[-1]] < min(v for k, v in w.items() if k != ids[-1])
    assert abs(sum(w.values()) - 1) < 1e-9
    # Decay: n = 0.5 * 20 + 0 after another iteration.
    pool.end_iteration()
    assert pool.scores["human"][ids[-1]].n == 10.0
    rng = np.random.default_rng(0)
    draws = [pool.sample("human", rng).id for _ in range(3000)]
    freq = {k: draws.count(k) / len(draws) for k in w}
    assert all(abs(freq[k] - w[k]) < 0.03 for k in w)
    back = Pool.from_state_dict(pool.state_dict())
    assert back.weights("human") == pool.weights("human")
    pool.write_summary(tmp_path / "pool.json")


def test_guard_engages_after_consecutive_windows_and_releases():
    g = Guard(LeagueParams(guard_windows=2))
    caught = np.full(10, OUTCOME_CAUGHT)
    g.update(caught)
    assert g.state == 0
    g.update(caught)
    assert g.state == 1
    assert g.overrides("human", 4) == (5, 1.0) and g.overrides("zombie", 3) == (3, 0.5)
    g.update(np.array([OUTCOME_CAUGHT] * 5 + [OUTCOME_TRUNCATED] * 5))
    assert g.state == 0
    for _ in range(2):
        g.update(np.full(10, OUTCOME_STARVED))
    assert g.alarm


def test_slots_never_switch_an_opponent_within_an_episode(tiny_cfg, tmp_path):
    env = MatchTrainingEnv(tiny_cfg, league_groups(16, [2], (0.5, 0.25, 0.25)), num_threads=1)
    pool = Pool()
    for i in range(3):
        path = tmp_path / f"z{i}.pt"
        save_snapshot(path, ZombieActor(hidden=8), "zombie", i, "s0_small")
        pool.add(Entry(f"z{i}", "zombie", str(path)))
    slots = OpponentSlots(
        "zombie",
        env,
        pool,
        torch.device("cpu"),
        np.random.default_rng(1),
        slots=3,
        scripted_share=0.3,
        lease_iters=1,
    )
    obs = env.reset()
    entry_at_start = dict(slots.env_entry)
    seen = set()
    for it in range(1, 40):
        za = np.zeros(env.num_agents, dtype=np.int64)
        za[slots.rows] = slots.act(obs)
        step = env.step(np.zeros((16, 2), dtype=np.int64), za)
        done = step.terminated | step.truncated
        for e in slots.envs:
            if not done[e]:
                assert slots.env_entry[int(e)] == entry_at_start[int(e)]
        slots.episodes_ended(done, step.outcome, lambda o: 1.0)
        entry_at_start = dict(slots.env_entry)
        slots.end_iteration(it)
        seen.update(slots.resident())
        obs = step.next
    assert len(seen) >= 3, "leases rotate the resident entries"
    assert sum(s.games for s in pool.scores["human"].values()) > 0


def test_tiny_league_trains_snapshots_and_resumes(tiny_cfg, tmp_path, monkeypatch):
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(tiny_cfg))
    starters = {}
    for side in ("human", "zombie"):
        actor = HumanActor(hidden=16) if side == "human" else ZombieActor(hidden=16)
        path = tmp_path / f"{side}_seed.pt"
        save_snapshot(path, actor, side, 0, "s0_small")
        starters[side] = path
    cfg = {
        "env_config": str(env_path),
        "zombie_counts": [2],
        "num_envs": 8,
        "total_steps": 3 * 16 * 8,
        "num_threads": 1,
        "device": "cpu",
        "eval_every": 3,
        "eval_episodes": 2,
        "checkpoint_every": 2,
        "seeds": [
            {"side": "human", "path": str(starters["human"])},
            {"side": "zombie", "path": str(starters["zombie"])},
        ],
        "league": {"snapshot_every_steps": 2 * 16 * 8, "guard_window": 1, "lease_iters": 1},
        "human": {"rollout": 16, "bptt": 8, "minibatches": 2, "epochs": 1, "hidden": 16},
        "zombie": {"rollout": 16, "bptt": 8, "minibatches": 2, "epochs": 1, "hidden": 16},
    }
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ZAI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("ZAI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ZAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ZAI_SERVER_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["league", "--config", str(cfg_path), "--run-id", "lg"])
    main()
    league_pt = tmp_path / "ckpt" / "league" / "lg" / "league.pt"
    state = torch.load(league_pt, weights_only=False)
    assert state["iteration"] == 3
    snaps = [e for e in state["pool"]["entries"]["zombie"] if e["source"] == "league"]
    assert len(snaps) == 2, "one at the start, one after two iterations"
    assert (tmp_path / "data" / "leagues" / "lg" / "pool.json").is_file()
    assert (tmp_path / "data" / "models" / "lg-human.onnx").is_file()
    # Resume for one more iteration: the pool state carries over.
    cfg["total_steps"] = 4 * 16 * 8
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        sys,
        "argv",
        ["league", "--config", str(cfg_path), "--run-id", "lg", "--resume", str(league_pt)],
    )
    main()
    state2 = torch.load(league_pt, weights_only=False)
    assert state2["iteration"] == 4
    assert state2["pool"]["snapshots"]["zombie"] >= state["pool"]["snapshots"]["zombie"]


@pytest.mark.parametrize("name", ["league_cpu.yaml", "league_s0small_gpu.yaml"])
def test_league_configs_load(name):
    from zai_env.config import config_dir as env_dir
    from zai_rl.train_league import LeagueConfig

    cfg = LeagueConfig.load(env_dir().parent / "train" / name, {})
    assert cfg.human.num_envs == len(
        [
            g
            for g in league_groups(cfg.num_envs, cfg.zombie_counts, cfg.league.shares)
            for _ in range(g.num_envs)
            if g.kind in "AB"
        ]
    )
