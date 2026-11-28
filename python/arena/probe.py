"""Exploitability probe of a league (M6.4, docs/ARCHITECTURE.md section 4.8.6).

For a side, the probe fine-tunes the league learner's latest checkpoint against the other
side's frozen latest checkpoint for `steps` env steps with the single-side trainer
(`zai_rl.train` or `zai_rl.train_zombie`, `opponent: policy` + `init_checkpoint`), then plays
the probe and the learner against that opponent on the arena's fixed seeds. The gap (probe
score minus learner score; human score for the human side, zombie score for the zombies) says
how exploitable the frozen opponent is: a large gap means the league learner of the other side
has a weakness a short best-response finds. Results are appended to
`<league dir>/probes.jsonl` and posted as `probe/<side>_gap` (model `arena`).

    python -m arena.probe --run-id ID --side both --steps 10000000
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from zai_env.config import deep_merge, load_config
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED

from .matches import play_detailed

log = logging.getLogger("arena.probe")


def side_score(side: str, outcome: np.ndarray) -> float:
    if side == "human":
        good = outcome == OUTCOME_TRUNCATED
    else:
        good = outcome == OUTCOME_CAUGHT
    return float((good + 0.5 * (outcome == OUTCOME_STARVED)).mean())


def probe_config(
    league: dict[str, Any], side: str, opponent: Path, init: Path, steps: int, threads: int
) -> dict[str, Any]:
    """Single-side trainer config fine-tuning `init` against the frozen `opponent`."""
    hp = dict(league["human" if side == "human" else "zombie"])
    hp["lr"] = 1e-4
    hp["ent_start"], hp["ent_end"] = 0.003, 0.001
    hp["num_envs"] = league["num_envs"]
    cfg = {
        "env_config": league["env_config"],
        "zombie_counts": league["zombie_counts"],
        "opponent": "policy",
        "init_checkpoint": str(init),
        "total_steps": steps,
        "seed": league["seed"] + 991,
        "num_threads": threads,
        "torch_threads": league.get("torch_threads"),
        "device": league["device"],
        "eval_every": 10_000,
        "checkpoint_every": 10_000,
        "replay_every": 0,
    }
    if side == "human":
        cfg["zombie_checkpoint"] = str(opponent)
        cfg["ppo"] = hp
    else:
        cfg["human_checkpoint"] = str(opponent)
        cfg["baseline_episodes"] = 8
        cfg["mappo"] = hp
    return cfg


def evaluate(
    side: str,
    mine: Path,
    opponent: Path,
    env_cfg: dict[str, Any],
    counts: list[int],
    episodes: int,
    device: str,
) -> float:
    from zai_rl.policy import (
        RecurrentHumanPolicy,
        RecurrentZombiePolicy,
        load_human_actor,
        load_zombie_actor,
    )

    h_path, z_path = (mine, opponent) if side == "human" else (opponent, mine)
    ha, za = load_human_actor(h_path, device), load_zombie_actor(z_path, device)
    scores = []
    for n in counts:
        cfg = deep_merge(env_cfg, {"zombie_count": n})
        ep = play_detailed(
            cfg,
            lambda c, e, s: RecurrentHumanPolicy(ha, e, device, config=c),
            lambda c, e, k, s: RecurrentZombiePolicy(za, e, k, device, config=c),
            episodes,
            4242 + n,
        )
        scores.append(side_score(side, ep.outcome))
    return float(np.mean(scores))


def main() -> None:
    from zai_rl.metrics import MetricsLogger
    from zai_rl.train import resolve

    parser = argparse.ArgumentParser(description="Exploitability probe of a league run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--side", choices=["human", "zombie", "both"], default="both")
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=10)
    # Diagnostics (docs/ARCHITECTURE.md 4.8.9, D3): fine-tune with another env config (e.g.
    # without the zombie shaping) or from another checkpoint than the learner's latest.pt.
    parser.add_argument("--env-config", help="training env config instead of the league's")
    parser.add_argument("--init", type=Path, help="checkpoint the probe starts from")
    parser.add_argument("--tag", default="", help="suffix of the probe id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints"))
    league_dir = root / "league" / args.run_id
    state = torch.load(league_dir / "league.pt", map_location="cpu", weights_only=False)
    league = state["config"]
    env_cfg = load_config(resolve(league["env_config"]), env_overrides=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = MetricsLogger(
        args.run_id,
        "arena",
        os.environ.get("ZAI_SERVER_URL"),
        Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
    )
    sides = ["human", "zombie"] if args.side == "both" else [args.side]
    for side in sides:
        mine = root / side / args.run_id / "latest.pt"
        opponent = root / ("zombie" if side == "human" else "human") / args.run_id / "latest.pt"
        probe_id = f"{args.run_id}-probe-{side}-{state['steps'] // 1_000_000}m{args.tag}"
        init = args.init or mine
        cfg = probe_config(league, side, opponent, init, args.steps, args.num_threads)
        if args.env_config:
            cfg["env_config"] = args.env_config
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
        module = "zai_rl.train" if side == "human" else "zai_rl.train_zombie"
        t0 = time.time()
        log.info(
            "probe %s: fine-tuning %s against %s for %d steps", probe_id, init, opponent, args.steps
        )
        subprocess.run(
            [sys.executable, "-m", module, "--config", f.name, "--run-id", probe_id],
            check=True,
            env={**os.environ, "ZAI_DATA_DIR": ""},
        )
        probe = root / side / probe_id / "latest.pt"
        counts = league["zombie_counts"]
        learner_score = evaluate(side, mine, opponent, env_cfg, counts, args.episodes, device)
        probe_score = evaluate(side, probe, opponent, env_cfg, counts, args.episodes, device)
        row = {
            "side": side,
            "probe": probe_id,
            "init": str(init),
            "env_config": cfg["env_config"],
            "path": str(probe),
            "league_steps": state["steps"],
            "learner_score": learner_score,
            "probe_score": probe_score,
            "gap": probe_score - learner_score,
            "seconds": time.time() - t0,
            "time": time.time(),
        }
        with (league_dir / "probes.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        metrics.log(state["steps"], {f"probe/{side}_gap": row["gap"]})
        log.info(
            "probe %s: learner %.3f, probe %.3f, gap %.1f pts",
            side,
            learner_score,
            probe_score,
            100 * row["gap"],
        )
    metrics.close()


if __name__ == "__main__":
    main()
