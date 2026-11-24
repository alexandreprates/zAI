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


def test_training_replays_use_fixed_seeds_and_iteration_ids(tmp_path):
    from arena.export_replays import record_training_replays

    cfg = short_config()

    def human(seed):
        return ScriptedHuman(cfg, 1, seed=seed), "scripted"

    def zombies(seed):
        return ScriptedZombies(cfg, 1, 3, seed=seed), "scripted"

    long_run = "zombie-s0-a-very-long-run-identifier-that-goes-past-the-limit"
    first = record_training_replays(tmp_path, long_run, 1, cfg, human, zombies, episodes=2)
    later = record_training_replays(tmp_path, long_run, 25, cfg, human, zombies, episodes=2)
    ids = [r["id"] for r in first + later]
    assert all(len(i) <= 64 and replay_id_prefix(i) == i for i in ids)
    assert ids[0].endswith("-it000001-e0") and ids[3].endswith("-it000025-e1")
    replays = [json.loads((tmp_path / f"{i}.json").read_text()) for i in ids]
    assert {r["match_id"] for r in replays} == {long_run}
    assert [r["episode"] for r in replays] == [1, 1, 25, 25]
    # Same scenario at every iteration: episode k always uses the same seed.
    assert replays[0]["seed"] == replays[2]["seed"] != replays[1]["seed"]
    assert all(resimulate(r) == r["final_hash"] for r in replays)
