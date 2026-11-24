import json

import numpy as np
import zai_sim

from arena.export_replays import record_episode, replay_id_prefix, write_replay
from zai_env.config import load_config
from zai_rl.bots import ScriptedHuman, ScriptedZombies
from zai_rl.models import HumanActor
from zai_rl.policy import RecurrentHumanPolicy


def short_config():
    return load_config(
        overrides={
            "map_size_m": 150.0,
            "house_count": 3,
            "zombie_count": 3,
            "food_count": 20,
            "episode_cap_s": 30.0,
        },
        env_overrides=False,
    )


def resimulate(replay):
    """What the server's `Replay::verify` does: re-run the action log from the seed."""
    env = zai_sim.Env(json.dumps(replay["config"]), replay["seed"])
    for a in replay["actions"]:
        env.step(a[0], a[1], a[2:])
    return f"{env.state_hash():016x}"


def test_recorded_episode_re_simulates_to_the_final_hash():
    cfg = short_config()
    seed = zai_sim.episode_seed(7, 0, 0)
    ep = record_episode(
        cfg, seed, ScriptedHuman(cfg, 1, seed=1), ScriptedZombies(cfg, 1, 3, seed=2)
    )
    assert ep["outcome"] in ("caught", "starved", "truncated")
    assert all(len(a) == 2 + 3 for a in ep["actions"])
    assert resimulate(ep) == ep["final_hash"]
    # The server deserializes the config into the Rust `Config`.
    zai_sim.validate_config(json.dumps(ep["config"]))


def test_policy_episode_is_written_as_a_loadable_replay(tmp_path):
    cfg = short_config()
    policy = RecurrentHumanPolicy(HumanActor(hidden=32), num_envs=1)
    policy.reset(np.ones(1, dtype=bool))
    ep = record_episode(cfg, 3, policy, ScriptedZombies(cfg, 1, 3, seed=3))
    replay = {
        "id": "run-latest-e0",
        "match_id": "run-latest",
        "episode": 0,
        "created_at": 0,
        "human": "policy",
        "zombies": "scripted",
        **ep,
    }
    path = write_replay(tmp_path / "replays", replay)
    loaded = json.loads(path.read_text())
    assert resimulate(loaded) == loaded["final_hash"]


def test_replay_ids_are_server_safe():
    assert replay_id_prefix("human-s0-gpu1-latest") == "human-s0-gpu1-latest"
    assert replay_id_prefix("../a b.pt") == "---a-b-pt"
