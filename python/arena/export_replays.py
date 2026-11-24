"""Records episodes of a trained human checkpoint as server replays.

Live server matches only run the Rust bots until the gRPC agent API lands (M7). Replays,
however, are just `(config, seed, action log)` re-simulated by the server, so episodes played
here with the Python policy can be written into the server's `replays/` directory and watched
in the viewer like any other replay.

    python -m arena.export_replays --checkpoint /checkpoints/human/<run>/latest.pt --episodes 10
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zai_sim

from zai_env.config import load_config
from zai_env.vec import OUTCOMES
from zai_rl.bots import RandomZombies, ScriptedZombies
from zai_rl.policy import RecurrentHumanPolicy
from zai_rl.train import resolve

log = logging.getLogger(__name__)


def record_episode(config: dict[str, Any], seed: int, human: Any, zombies: Any) -> dict[str, Any]:
    """Plays one episode with batched (`num_envs=1`) policies and returns the replay fields
    that depend on the episode: config, seed, actions, outcome and final state hash."""
    env = zai_sim.Env(json.dumps(config), seed)
    actions: list[list[int]] = []
    while OUTCOMES[env.outcome] == "running":
        h = human.act(env.human_obs()[None], env.human_mask()[None])[0]
        z = zombies.act(env.zombie_obs()[None])[0]
        movement, interact = int(h[0]), int(h[1])
        zs = [int(a) for a in z]
        actions.append([movement, interact, *zs])
        env.step(movement, interact, zs)
    return {
        "config": json.loads(env.config_json),
        "seed": seed,
        "actions": actions,
        "outcome": OUTCOMES[env.outcome],
        "final_hash": f"{env.state_hash():016x}",
    }


def replay_id_prefix(name: str) -> str:
    """Server replay ids allow only `[A-Za-z0-9_-]` and at most 64 characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)[:56]


def write_replay(out_dir: Path, replay: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{replay['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(replay))
    tmp.replace(path)
    return path


def default_out_dir() -> Path:
    return Path(os.environ.get("ZAI_DATA_DIR", "/data")) / "replays"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a human checkpoint's episodes as replays.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", default="configs/env/s0.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--zombies", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--deterministic", action="store_true", help="argmax actions")
    parser.add_argument("--name", help="match id shown in the viewer (default: <run>-<ckpt>)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $ZAI_DATA_DIR/replays")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.set_num_threads(1)
    config = load_config(resolve(args.env_config), env_overrides=False)
    policy = RecurrentHumanPolicy.from_checkpoint(
        args.checkpoint, num_envs=1, deterministic=args.deterministic
    )
    torch.manual_seed(args.seed)
    name = replay_id_prefix(args.name or f"{args.checkpoint.parent.name}-{args.checkpoint.stem}")
    out_dir = args.out_dir or default_out_dir()

    counts = dict.fromkeys(OUTCOMES[1:], 0)
    for ep in range(args.episodes):
        seed = zai_sim.episode_seed(args.seed, 0, ep)
        policy.reset(np.ones(1, dtype=bool))
        n = config["zombie_count"]
        zombies = (
            ScriptedZombies(config, 1, n, seed=seed)
            if args.zombies == "scripted"
            else RandomZombies(1, n, seed=seed)
        )
        episode = record_episode(config, seed, policy, zombies)
        replay = {
            "id": f"{name}-e{ep}",
            "match_id": name,
            "episode": ep,
            "created_at": int(time.time()),
            "human": "policy",
            "zombies": args.zombies,
            **episode,
        }
        path = write_replay(out_dir, replay)
        counts[episode["outcome"]] += 1
        log.info(
            "episode %d: %s after %.0f s -> %s",
            ep,
            episode["outcome"],
            len(episode["actions"]) * config["dt_s"] * config["action_repeat"],
            path,
        )
    log.info("done: %s", ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
