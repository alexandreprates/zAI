"""Zombie training entrypoint (M5): recurrent MAPPO against a fixed human bot.

    python -m zai_rl.train_zombie --config configs/train/zombie_s0.yaml [--total-steps N]

Same outputs as the human trainer (`zai_rl.train`): stdout, TensorBoard (`ZAI_RUNS_DIR`), the
zai-server metrics API (`ZAI_SERVER_URL`, model `zombie`) and checkpoints under
`ZAI_CHECKPOINT_DIR/zombie/<run_id>/`. `total_steps` counts env steps (decisions per match), as
for the human; zombie agent steps are about `mean(zombie_counts)` times more.

Evaluation plays each zombie count separately on fixed seeds and reports the catch rate per
count and pooled, with a Wilson 95% interval, next to the scripted-zombie baseline measured at
startup on the same seeds (M5 acceptance: pooled catch rate above the baseline with
non-overlapping intervals, docs/ARCHITECTURE.md section 4.6).
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from arena.export_model import export_training_model
from arena.export_replays import record_training_replays, training_replay_dir
from arena.matches import MatchStats, play
from zai_env.config import deep_merge, load_config

from .algos.common import EpisodeStats, save_checkpoint
from .algos.mappo_zombie import MAPPOConfig, ZombieMAPPO
from .bots import RandomHuman, RandomZombies, ScriptedHuman, ScriptedZombies
from .envs import HumanFactory, ZombieFactory, ZombieTrainingEnv
from .metrics import MetricsLogger
from .policy import RecurrentHumanPolicy, RecurrentZombiePolicy
from .train import OUTCOME_KEYS, resolve

log = logging.getLogger("zai_rl.train_zombie")


@dataclass
class ZombieTrainConfig:
    run_id: str | None = None
    env_config: str = "configs/env/s0.yaml"
    zombie_counts: list[int] = field(default_factory=lambda: [2, 3, 4])
    opponent: str = "scripted"  # human bot used for training: scripted | random
    total_steps: int = 20_000_000
    seed: int = 0
    num_threads: int | None = None
    torch_threads: int | None = None
    device: str = "auto"
    eval_every: int = 10
    eval_episodes: int = 128  # per zombie count
    baseline_episodes: int = 256  # per zombie count, measured once at startup
    eval_human_checkpoint: str | None = None  # frozen learned human, extra eval opponent
    checkpoint_every: int = 20
    # At iteration 1 and every `replay_every` iterations (0 disables), record `replay_episodes`
    # episodes of the current policy into the server replays (needs ZAI_DATA_DIR), same seeds each
    # time, so the viewer shows how play changes over the run.
    replay_every: int = 0
    replay_episodes: int = 2
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)

    @classmethod
    def load(cls, path: Path | None, overrides: dict[str, Any]) -> "ZombieTrainConfig":
        raw: dict[str, Any] = {}
        if path is not None:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        if raw.get("opponent", "scripted") not in ("scripted", "random"):
            raise ValueError("opponent must be 'scripted' or 'random'")
        mappo = MAPPOConfig(**raw.pop("mappo", {}))
        return cls(**raw, mappo=mappo)


def human_bot(kind: str) -> HumanFactory:
    if kind == "random":
        return lambda c, e, s: RandomHuman(e, seed=s)
    return lambda c, e, s: ScriptedHuman(c, e, seed=s)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion `p` observed over `n` trials."""
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def catch_rates(
    env_cfg: dict[str, Any],
    counts: list[int],
    human_factory: HumanFactory,
    zombie_factory: ZombieFactory,
    episodes: int,
    seed: int,
    prefix: str,
) -> dict[str, float]:
    """Plays `episodes` per zombie count on fixed seeds; returns per-count and pooled rates."""
    out: dict[str, float] = {}
    results: list[MatchStats] = []
    for n in counts:
        cfg = deep_merge(env_cfg, {"zombie_count": n})
        r = play(cfg, human_factory, zombie_factory, episodes=episodes, seed=seed + n)
        results.append(r)
        out[f"{prefix}/caught_z{n}"] = r.caught
    total = sum(r.episodes for r in results)
    for key in ("caught", "starved", "survived", "mean_length_s"):
        out[f"{prefix}/{key}"] = sum(getattr(r, key) * r.episodes for r in results) / total
    lo, hi = wilson(out[f"{prefix}/caught"], total)
    out[f"{prefix}/caught_lo"], out[f"{prefix}/caught_hi"] = lo, hi
    return out


def learned_zombies(algo: ZombieMAPPO) -> ZombieFactory:
    return lambda c, e, n, s: RecurrentZombiePolicy(algo.actor, e, n, algo.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the zombie team with recurrent MAPPO.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/zombie_s0.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ZombieTrainConfig.load(
        resolve(str(args.config)),
        {"run_id": args.run_id, "total_steps": args.total_steps, "device": args.device},
    )
    run_id = cfg.run_id or time.strftime("zombie-s0-%Y%m%d-%H%M%S")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env_cfg = load_config(resolve(cfg.env_config), env_overrides=False)
    decision_s = env_cfg["dt_s"] * env_cfg["action_repeat"]
    env = ZombieTrainingEnv(
        env_cfg,
        cfg.zombie_counts,
        cfg.mappo.num_envs,
        human_bot(cfg.opponent),
        seed=cfg.seed,
        num_threads=cfg.num_threads,
    )
    algo = ZombieMAPPO(cfg.mappo, device)
    iteration, steps = 0, 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        algo.load_state_dict(sd)
        iteration, steps = sd["iteration"], sd["steps"]
        run_id = sd.get("run_id", run_id)
        log.info("resumed %s at iteration %d (%d steps)", args.resume, iteration, steps)

    ckpt_dir = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints")) / "zombie" / run_id
    metrics = MetricsLogger(
        run_id,
        "zombie",
        server_url=os.environ.get("ZAI_SERVER_URL"),
        tensorboard_dir=Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
    )
    params = sum(p.numel() for p in algo.actor.parameters())
    log.info(
        "run %s on %s; actor %.2fM params; %d envs, %d zombie agents",
        run_id,
        device,
        params / 1e6,
        env.num_envs,
        env.num_agents,
    )

    eval_seed = cfg.seed + 777
    opponent = human_bot(cfg.opponent)
    evals: list[tuple[str, HumanFactory]] = [("eval", opponent)]
    if cfg.eval_human_checkpoint:
        frozen = RecurrentHumanPolicy.from_checkpoint(
            resolve(cfg.eval_human_checkpoint), num_envs=1, device=str(device)
        ).actor
        evals.append(
            ("eval_vs_learned_human", lambda c, e, s: RecurrentHumanPolicy(frozen, e, device))
        )

    def measure(prefix: str, zombies: ZombieFactory, episodes: int) -> dict[str, float]:
        return catch_rates(
            env_cfg, cfg.zombie_counts, opponent, zombies, episodes, eval_seed, prefix
        )

    baseline = measure(
        "baseline", lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s), cfg.baseline_episodes
    )
    floor = measure(
        "baseline_random", lambda c, e, n, s: RandomZombies(e, n, seed=s), cfg.eval_episodes
    )
    log.info(
        "baseline (scripted zombies vs %s human): caught %.1f%% [%.1f, %.1f] (%s); random %.1f%%",
        cfg.opponent,
        100 * baseline["baseline/caught"],
        100 * baseline["baseline/caught_lo"],
        100 * baseline["baseline/caught_hi"],
        ", ".join(
            f"z{n} {100 * baseline[f'baseline/caught_z{n}']:.1f}%" for n in cfg.zombie_counts
        ),
        100 * floor["baseline_random/caught"],
    )

    replay_dir = training_replay_dir()
    batch = env.reset()
    h = algo.actor.initial_state(env.num_agents, device)
    start = torch.ones(env.num_agents, dtype=torch.bool, device=device)
    ep_return = np.zeros(env.num_agents, dtype=np.float64)
    per_iter = cfg.mappo.rollout * env.num_envs
    started = time.time()
    while steps < cfg.total_steps:
        t0 = time.time()
        ep = EpisodeStats()
        rollout, batch, h, start = algo.collect(env, batch, h, start, ep_return, ep)
        t1 = time.time()
        stats = algo.update(rollout, steps / cfg.total_steps)
        t2 = time.time()
        iteration += 1
        steps += per_iter

        scalars = dict(stats)
        scalars["perf/steps_per_s"] = per_iter / (t2 - t0)
        scalars["perf/agent_steps_per_s"] = cfg.mappo.rollout * env.num_agents / (t2 - t0)
        scalars["perf/rollout_s"] = t1 - t0
        scalars["perf/update_s"] = t2 - t1
        scalars["train/reward_mean"] = rollout["rewards"].mean().item()
        scalars["train/nearest_zombie_m"] = rollout["nearest_zombie_m"].item()
        if ep.returns:
            outcomes = np.array(ep.outcomes)
            scalars["episode/return"] = float(np.mean(ep.returns))
            scalars["episode/length_s"] = float(np.mean(ep.lengths)) * decision_s
            scalars["episode/count"] = len(ep.returns)
            for code, name in OUTCOME_KEYS.items():
                scalars[f"episode/{name}"] = float((outcomes == code).mean())

        if iteration % cfg.eval_every == 0:
            for prefix, human in evals:
                scalars.update(
                    catch_rates(
                        env_cfg,
                        cfg.zombie_counts,
                        human,
                        learned_zombies(algo),
                        cfg.eval_episodes,
                        eval_seed,
                        prefix,
                    )
                )
            scalars["eval/baseline_caught"] = baseline["baseline/caught"]
            scalars["eval/baseline_caught_hi"] = baseline["baseline/caught_hi"]
            log.info(
                "eval: caught %.1f%% [%.1f, %.1f] (baseline %.1f%% [%.1f, %.1f]), starved %.1f%%%s",
                100 * scalars["eval/caught"],
                100 * scalars["eval/caught_lo"],
                100 * scalars["eval/caught_hi"],
                100 * baseline["baseline/caught"],
                100 * baseline["baseline/caught_lo"],
                100 * baseline["baseline/caught_hi"],
                100 * scalars["eval/starved"],
                (
                    f"; vs learned human {100 * scalars['eval_vs_learned_human/caught']:.1f}%"
                    if "eval_vs_learned_human/caught" in scalars
                    else ""
                ),
            )

        if (
            replay_dir
            and cfg.replay_every
            and (iteration == 1 or iteration % cfg.replay_every == 0)
        ):
            n = env_cfg["zombie_count"]
            recorded = record_training_replays(
                replay_dir,
                run_id,
                iteration,
                env_cfg,
                lambda seed: (opponent(env_cfg, 1, seed), cfg.opponent),
                lambda seed, n=n: (RecurrentZombiePolicy(algo.actor, 1, n, device), "policy"),
                cfg.replay_episodes,
            )
            log.info("replays: %s", ", ".join(f"{r['id']} {r['outcome']}" for r in recorded))

        metrics.log(steps, scalars)
        log.info(
            "it %d steps %d | return %.3f len %.0fs caught %.2f nearest %.0fm | %.0f steps/s",
            iteration,
            steps,
            scalars.get("episode/return", float("nan")),
            scalars.get("episode/length_s", float("nan")),
            scalars.get("episode/caught", float("nan")),
            scalars["train/nearest_zombie_m"],
            scalars["perf/steps_per_s"],
        )
        if iteration % cfg.checkpoint_every == 0 or steps >= cfg.total_steps:
            extra = {
                "iteration": iteration,
                "steps": steps,
                "run_id": run_id,
                "env_config": env_cfg,
            }
            save_checkpoint(ckpt_dir / f"it{iteration:06d}.pt", algo, extra)
            save_checkpoint(ckpt_dir / "latest.pt", algo, extra)
            # Live matches in the viewer play the latest exported model of each run.
            export_training_model(algo.actor, run_id, env_cfg, iteration, steps)

    metrics.close()
    log.info("done: %d steps in %.0f s", steps, time.time() - started)


if __name__ == "__main__":
    main()
