"""Records episodes of trained checkpoints (human, zombies or both) as server replays.

Live server matches only run the Rust bots until the gRPC agent API lands (M7). Replays,
however, are just `(config, seed, action log)` re-simulated by the server, so episodes played
here with the Python policy can be written into the server's `replays/` directory and watched
in the viewer like any other replay.

    python -m arena.export_replays --checkpoint /checkpoints/human/<run>/latest.pt --episodes 10
    python -m arena.export_replays --zombie-checkpoint /checkpoints/zombie/<run>/latest.pt

A side without a checkpoint is played by a bot (`--human` / `--zombies`, scripted by default).
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
from zai_rl.bots import RandomHuman, RandomZombies, ScriptedHuman, ScriptedZombies
from zai_rl.policy import RecurrentHumanPolicy, RecurrentZombiePolicy
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


def checkpoint_name(path: Path) -> str:
    return f"{path.parent.name}-{path.stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record checkpoint episodes as viewer replays.")
    parser.add_argument(
        "--checkpoint", "--human-checkpoint", dest="human_checkpoint", type=Path, default=None
    )
    parser.add_argument("--zombie-checkpoint", type=Path, default=None)
    parser.add_argument("--env-config", default="configs/env/s0.yaml")
    parser.add_argument("--zombie-count", type=int, help="override the env config's count")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--human", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--zombies", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--deterministic", action="store_true", help="argmax actions")
    parser.add_argument("--name", help="match id shown in the viewer (default: from checkpoints)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $ZAI_DATA_DIR/replays")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.human_checkpoint is None and args.zombie_checkpoint is None:
        parser.error("give --checkpoint (human) and/or --zombie-checkpoint")

    torch.set_num_threads(1)
    overrides = {"zombie_count": args.zombie_count} if args.zombie_count else None
    config = load_config(resolve(args.env_config), overrides, env_overrides=False)
    n = config["zombie_count"]
    human_policy = (
        RecurrentHumanPolicy.from_checkpoint(
            args.human_checkpoint, num_envs=1, deterministic=args.deterministic
        )
        if args.human_checkpoint
        else None
    )
    zombie_policy = (
        RecurrentZombiePolicy.from_checkpoint(
            args.zombie_checkpoint, num_envs=1, num_zombies=n, deterministic=args.deterministic
        )
        if args.zombie_checkpoint
        else None
    )
    torch.manual_seed(args.seed)
    parts = [checkpoint_name(p) for p in (args.human_checkpoint, args.zombie_checkpoint) if p]
    name = replay_id_prefix(args.name or "-vs-".join(parts))
    out_dir = args.out_dir or default_out_dir()
    human_kind = "policy" if human_policy else args.human
    zombie_kind = "policy" if zombie_policy else args.zombies

    counts = dict.fromkeys(OUTCOMES[1:], 0)
    for ep in range(args.episodes):
        seed = zai_sim.episode_seed(args.seed, 0, ep)
        fresh = np.ones(1, dtype=bool)
        if human_policy:
            human_policy.reset(fresh)
            human = human_policy
        elif args.human == "scripted":
            human = ScriptedHuman(config, 1, seed=seed)
        else:
            human = RandomHuman(1, seed=seed)
        if zombie_policy:
            zombie_policy.reset(fresh)
            zombies = zombie_policy
        elif args.zombies == "scripted":
            zombies = ScriptedZombies(config, 1, n, seed=seed)
        else:
            zombies = RandomZombies(1, n, seed=seed)
        episode = record_episode(config, seed, human, zombies)
        replay = {
            "id": f"{name}-e{ep}",
            "match_id": name,
            "episode": ep,
            "created_at": int(time.time()),
            "human": human_kind,
            "zombies": zombie_kind,
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
