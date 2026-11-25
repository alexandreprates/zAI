"""Human training entrypoint (M4): recurrent PPO against fixed zombie bots.

    python -m zai_rl.train --config configs/train/human_s0.yaml [--total-steps N] [--run-id ID]

With `opponent: policy` the zombies are a frozen trained policy (`zombie_checkpoint`), and
`init_checkpoint` starts from a trained human's weights (fine-tuning: fresh optimizer and
counters, unlike `--resume`).

Metrics go to stdout, TensorBoard (`ZAI_RUNS_DIR`, default ./runs) and, when `ZAI_SERVER_URL`
is set, to the zai-server metrics API (visible in the viewer). Checkpoints are written to
`ZAI_CHECKPOINT_DIR` (default ./checkpoints) under `human/<run_id>/`.
"""

import argparse
import logging
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
from arena.matches import play
from zai_env.config import deep_merge, load_config

from .algos.ppo_human import EpisodeStats, HumanPPO, PPOConfig, save_checkpoint
from .bots import ScriptedHuman, ScriptedZombies
from .envs import HumanTrainingEnv
from .metrics import MetricsLogger
from .policy import RecurrentHumanPolicy, RecurrentZombiePolicy

log = logging.getLogger("zai_rl.train")
REPO = Path(__file__).resolve().parents[2]
OUTCOME_KEYS = {1: "caught", 2: "starved", 3: "survived"}


@dataclass
class TrainConfig:
    run_id: str | None = None
    env_config: str = "configs/env/s0.yaml"
    zombie_counts: list[int] = field(default_factory=lambda: [2, 3, 4])
    # Zombie opponent: `scripted` bots or a frozen trained `policy` (`zombie_checkpoint`).
    opponent: str = "scripted"
    zombie_checkpoint: str | None = None
    # Human checkpoint whose actor, critic and value statistics initialize the run.
    init_checkpoint: str | None = None
    total_steps: int = 20_000_000
    seed: int = 0
    num_threads: int | None = None
    torch_threads: int | None = None
    device: str = "auto"
    eval_every: int = 10
    eval_episodes: int = 64
    checkpoint_every: int = 20
    # At iteration 1 and every `replay_every` iterations (0 disables), record `replay_episodes`
    # episodes of the current policy into the server replays (needs ZAI_DATA_DIR).
    replay_every: int = 0
    replay_episodes: int = 2
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @classmethod
    def load(cls, path: Path | None, overrides: dict[str, Any]) -> "TrainConfig":
        raw: dict[str, Any] = {}
        if path is not None:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        ppo = PPOConfig(**raw.pop("ppo", {}))
        return cls(**raw, ppo=ppo)


def resolve(path: str) -> Path:
    """Relative paths resolve against the working directory, then the source checkout."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO / p


def scripted_zombies(c, e, n, s):
    return ScriptedZombies(c, e, n, seed=s)


def zombie_opponent(cfg: TrainConfig, device: torch.device):
    """Zombie factory `(config, num_envs, num_zombies, seed) -> policy` for `cfg.opponent`."""
    if cfg.opponent == "scripted":
        return scripted_zombies
    if cfg.opponent != "policy" or not cfg.zombie_checkpoint:
        raise ValueError("opponent must be 'scripted', or 'policy' with a zombie_checkpoint")
    path = resolve(cfg.zombie_checkpoint)
    return lambda c, e, n, s: RecurrentZombiePolicy.from_checkpoint(path, e, n, str(device))


def evaluate(
    env_cfg: dict[str, Any],
    algo: HumanPPO,
    episodes: int,
    seed: int,
    zombies=scripted_zombies,
    prefix: str = "eval",
) -> dict[str, float]:
    stats = play(
        env_cfg,
        lambda c, e, s: RecurrentHumanPolicy(algo.actor, e, algo.device),
        zombies,
        episodes=episodes,
        seed=seed,
    )
    return {
        f"{prefix}/survived": stats.survived,
        f"{prefix}/caught": stats.caught,
        f"{prefix}/starved": stats.starved,
        f"{prefix}/mean_length_s": stats.mean_length_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the human policy with recurrent PPO.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/human_s0.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = TrainConfig.load(
        resolve(str(args.config)),
        {"run_id": args.run_id, "total_steps": args.total_steps, "device": args.device},
    )
    run_id = cfg.run_id or time.strftime("human-s0-%Y%m%d-%H%M%S")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Curriculum stage configs fix their own counts; ZAI_ZOMBIE_COUNT / ZAI_FOOD_COUNT apply to
    # live matches and full-spec runs, not to these stages.
    env_cfg = load_config(resolve(cfg.env_config), env_overrides=False)
    decision_s = env_cfg["dt_s"] * env_cfg["action_repeat"]
    opponent = zombie_opponent(cfg, device)
    env = HumanTrainingEnv(
        env_cfg,
        cfg.zombie_counts,
        cfg.ppo.num_envs,
        opponent,
        seed=cfg.seed,
        num_threads=cfg.num_threads,
    )
    algo = HumanPPO(cfg.ppo, device)
    iteration, steps = 0, 0
    if cfg.init_checkpoint and not args.resume:
        sd = torch.load(resolve(cfg.init_checkpoint), map_location=device, weights_only=False)
        algo.actor.load_state_dict(sd["actor"])
        algo.critic.load_state_dict(sd["critic"])
        algo.value_rms.load_state_dict(sd["value_rms"])
        log.info("initialized from %s (%s steps)", cfg.init_checkpoint, sd.get("steps"))
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        algo.load_state_dict(sd)
        iteration, steps = sd["iteration"], sd["steps"]
        run_id = sd.get("run_id", run_id)
        log.info("resumed %s at iteration %d (%d steps)", args.resume, iteration, steps)

    ckpt_dir = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints")) / "human" / run_id
    metrics = MetricsLogger(
        run_id,
        "human",
        server_url=os.environ.get("ZAI_SERVER_URL"),
        tensorboard_dir=Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
    )
    params = sum(p.numel() for p in algo.actor.parameters())
    log.info(
        "run %s on %s; actor %.2fM params; %d envs", run_id, device, params / 1e6, env.num_envs
    )

    baseline = play(
        env_cfg,
        lambda c, e, s: ScriptedHuman(c, e, seed=s),
        opponent,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 777,
    )
    log.info(
        "baseline (scripted human vs %s zombies): survived %.1f%%",
        cfg.opponent,
        100 * baseline.survived,
    )
    if iteration == 0 and cfg.init_checkpoint:
        # The starting point of a fine-tune, at step 0 of the eval curves.
        ev = evaluate(env_cfg, algo, cfg.eval_episodes, cfg.seed + 777, opponent)
        ev["eval/baseline_survived"] = baseline.survived
        metrics.log(0, ev)
        log.info("initial eval: survived %.1f%%", 100 * ev["eval/survived"])

    replay_dir = training_replay_dir()
    batch = env.reset()
    h = algo.actor.initial_state(env.num_envs, device)
    start = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    ep_return = np.zeros(env.num_envs, dtype=np.float64)
    per_iter = cfg.ppo.rollout * env.num_envs
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
        scalars["perf/rollout_s"] = t1 - t0
        scalars["perf/update_s"] = t2 - t1
        scalars["train/reward_mean"] = rollout["rewards"].mean().item()
        if ep.returns:
            outcomes = np.array(ep.outcomes)
            scalars["episode/return"] = float(np.mean(ep.returns))
            scalars["episode/length_s"] = float(np.mean(ep.lengths)) * decision_s
            scalars["episode/count"] = len(ep.returns)
            for code, name in OUTCOME_KEYS.items():
                scalars[f"episode/{name}"] = float((outcomes == code).mean())

        if iteration % cfg.eval_every == 0:
            ev = evaluate(env_cfg, algo, cfg.eval_episodes, cfg.seed + 777, opponent)
            if cfg.opponent != "scripted":
                # Watch for forgetting how to beat the bots it was trained against first.
                scalars.update(
                    evaluate(
                        env_cfg, algo, cfg.eval_episodes, cfg.seed + 777, prefix="eval_scripted"
                    )
                )
            scalars.update(ev)
            scalars["eval/baseline_survived"] = baseline.survived
            log.info(
                "eval: survived %.1f%% (baseline %.1f%%), caught %.1f%%, starved %.1f%%",
                100 * ev["eval/survived"],
                100 * baseline.survived,
                100 * ev["eval/caught"],
                100 * ev["eval/starved"],
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
                lambda seed: (RecurrentHumanPolicy(algo.actor, 1, device), "policy"),
                lambda seed, n=n: (
                    opponent(deep_merge(env_cfg, {"zombie_count": n}), 1, n, seed),
                    "scripted" if cfg.opponent == "scripted" else "policy",
                ),
                cfg.replay_episodes,
            )
            log.info("replays: %s", ", ".join(f"{r['id']} {r['outcome']}" for r in recorded))

        metrics.log(steps, scalars)
        log.info(
            "it %d steps %d | return %.3f len %.0fs survived %.2f | %.0f steps/s",
            iteration,
            steps,
            scalars.get("episode/return", float("nan")),
            scalars.get("episode/length_s", float("nan")),
            scalars.get("episode/survived", float("nan")),
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
