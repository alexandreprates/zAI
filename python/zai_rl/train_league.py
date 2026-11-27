"""League co-training (M6.3, docs/ARCHITECTURE.md sections 4.3 and 4.8.4-4.8.5).

One process trains both learners in the same envs, split into groups: A (50%, latest human vs
latest zombies, both learn), B (25%, the human learner vs frozen zombies from the pool), C
(25%, the zombie learner vs a frozen human from the pool), each split across the stage's
zombie counts. Frozen opponents come from a snapshot pool sampled with PFSP
(`league.pool`), batched per resident slot with leases (`league.slots`); the scripted bot of
each side is pinned to `scripted_share` of that side's B/C episodes. Both learners snapshot
into the pool every `snapshot_every_steps`. An imbalance guard gives the losing side an extra
epoch and halves the winning side's learning rate while group A's zombie score stays outside
[0.15, 0.85].

Outputs: learner checkpoints where the single-side trainers write them
(`/checkpoints/{human,zombie}/<run_id>/`), league state and pool under
`/checkpoints/league/<run_id>/` (`league.pt`, `pool.json`, `pool/<side>/<id>.pt`), a copy of
`pool.json` in `<ZAI_DATA_DIR>/leagues/<run_id>/`, and metrics for models `human` and
`zombie` under the run id.

    python -m zai_rl.train_league --config configs/train/league_cpu.yaml
"""

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from arena.export_model import export_training_model
from arena.tactics import tactic_metrics
from zai_env.config import load_config
from zai_env.layout import OBS_VERSION
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED

from .algos.common import EpisodeStats, require_obs_version, save_checkpoint
from .algos.mappo_zombie import MAPPOConfig, ZombieMAPPO, ZombieRollout
from .algos.ppo_human import HumanPPO, HumanRollout, PPOConfig
from .bots import ScriptedHuman, ScriptedZombies
from .envs import MatchTrainingEnv, league_groups
from .league.pool import Entry, Pool, other
from .league.slots import OpponentSlots
from .metrics import MetricsLogger
from .policy import RecurrentHumanPolicy, RecurrentZombiePolicy
from .profiling import rollout_split, timer
from .train import OUTCOME_KEYS, resolve
from .train_zombie import catch_rates

log = logging.getLogger("zai_rl.train_league")


@dataclass
class LeagueParams:
    shares: tuple[float, float, float] = (0.5, 0.25, 0.25)  # groups A, B, C
    slots: int = 4  # resident opponent slots per side, slot 0 is the scripted bot
    scripted_share: float = 0.2
    lease_iters: int = 5
    snapshot_every_steps: int = 10_000_000
    pool_cap: int = 50
    decay: float = 0.98
    pfsp: str = "hard"  # hard: (1 - p)^2, variance: p (1 - p)
    uniform: float = 0.1
    guard_window: int = 10  # iterations
    guard_high: float = 0.85
    guard_low: float = 0.15
    guard_windows: int = 3  # consecutive windows before the guard engages
    guard_release: tuple[float, float] = (0.3, 0.7)
    starvation_alarm: float = 0.5


@dataclass
class Seed:
    side: str
    path: str
    id: str | None = None
    source: str = "distilled"


@dataclass
class LeagueConfig:
    run_id: str | None = None
    env_config: str = "configs/env/s0_small.yaml"
    stage: str = "s0_small"
    zombie_counts: list[int] = field(default_factory=lambda: [3, 6, 9])
    num_envs: int = 1024
    total_steps: int = 100_000_000
    seed: int = 0
    num_threads: int | None = None
    torch_threads: int | None = None
    device: str = "auto"
    human_init: str | None = None  # layout v2 checkpoints the learners start from
    zombie_init: str | None = None
    seeds: list[Seed] = field(default_factory=list)  # extra pool entries
    eval_every: int = 20  # iterations; anchor evaluation against the scripted bots (0: off)
    eval_episodes: int = 64  # per zombie count
    checkpoint_every: int = 20
    league: LeagueParams = field(default_factory=LeagueParams)
    human: PPOConfig = field(default_factory=PPOConfig)
    zombie: MAPPOConfig = field(default_factory=MAPPOConfig)

    @classmethod
    def load(cls, path: Path | None, overrides: dict[str, Any]) -> "LeagueConfig":
        raw: dict[str, Any] = {}
        if path is not None:
            raw = yaml.safe_load(path.read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown league config keys: {sorted(unknown)}")
        lg = raw.pop("league", {})
        for k in ("shares", "guard_release"):
            if k in lg:
                lg[k] = tuple(lg[k])
        bad = set(lg) - {f.name for f in fields(LeagueParams)}
        if bad:
            raise ValueError(f"unknown league.* keys: {sorted(bad)}")
        groups = league_groups(
            raw.get("num_envs", 1024),
            raw.get("zombie_counts", [3, 6, 9]),
            lg.get("shares", LeagueParams.shares),
        )
        rows_h = sum(g.num_envs for g in groups if g.kind in "AB")
        human = PPOConfig(**{**raw.pop("human", {}), "num_envs": rows_h})
        zombie = MAPPOConfig(**{**raw.pop("zombie", {}), "num_envs": rows_h})
        seeds = [Seed(**s) for s in raw.pop("seeds", [])]
        return cls(**raw, league=LeagueParams(**lg), human=human, zombie=zombie, seeds=seeds)


def human_score(outcome: int) -> float:
    """Human learner's score against a frozen zombie team."""
    return {OUTCOME_TRUNCATED: 1.0, OUTCOME_STARVED: 0.5}.get(outcome, 0.0)


def zombie_score(outcome: int) -> float:
    return {OUTCOME_CAUGHT: 1.0, OUTCOME_STARVED: 0.5}.get(outcome, 0.0)


def save_snapshot(path: Path, actor: torch.nn.Module, side: str, steps: int, stage: str) -> None:
    """An actor-only file that `policy.load_{human,zombie}_actor` can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = "ppo_config" if side == "human" else "mappo_config"
    sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "actor": sd,
            key: {"hidden": actor.hidden_size},
            "obs_version": actor.obs_version,
            "side": side,
            "steps": steps,
            "stage": stage,
        },
        tmp,
    )
    tmp.replace(path)


class Guard:
    """Imbalance guard and starvation alarm over windows of group A episodes (4.8.5)."""

    def __init__(self, p: LeagueParams):
        self.p = p
        self.state = 0  # +1: zombies dominate, -1: the human dominates
        self.high = self.low = self.starving = 0
        self.alarm = False

    def update(self, outcomes: np.ndarray) -> float:
        """Folds in one window of group A outcomes; returns the window's zombie score."""
        if len(outcomes) == 0:
            return float("nan")
        caught = (outcomes == OUTCOME_CAUGHT).mean()
        starved = (outcomes == OUTCOME_STARVED).mean()
        s_z = float(caught + 0.5 * starved)
        self.high = self.high + 1 if s_z > self.p.guard_high else 0
        self.low = self.low + 1 if s_z < self.p.guard_low else 0
        if self.high >= self.p.guard_windows:
            self.state = 1
        elif self.low >= self.p.guard_windows:
            self.state = -1
        elif self.p.guard_release[0] <= s_z <= self.p.guard_release[1]:
            self.state = 0
        self.starving = self.starving + 1 if starved > self.p.starvation_alarm else 0
        self.alarm = self.starving >= self.p.guard_windows
        return s_z

    def overrides(self, side: str, epochs: int) -> tuple[int, float]:
        """`(epochs, lr_scale)` of `side`'s next update."""
        losing = (self.state == 1 and side == "human") or (self.state == -1 and side == "zombie")
        winning = (self.state == 1 and side == "zombie") or (self.state == -1 and side == "human")
        return epochs + int(losing), 0.5 if winning else 1.0


class League:
    def __init__(self, cfg: LeagueConfig, run_id: str, device: torch.device):
        self.cfg, self.run_id, self.device = cfg, run_id, device
        self.env_cfg = load_config(resolve(cfg.env_config), env_overrides=False)
        p = cfg.league
        self.groups = league_groups(cfg.num_envs, cfg.zombie_counts, p.shares)
        self.env = MatchTrainingEnv(self.env_cfg, self.groups, cfg.seed, cfg.num_threads)
        self.human = HumanPPO(cfg.human, device)
        self.zombie = ZombieMAPPO(cfg.zombie, device)
        self.dir = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints")) / "league" / run_id
        data = os.environ.get("ZAI_DATA_DIR")
        self.public = Path(data) / "leagues" / run_id if data else None
        self.rng = np.random.default_rng(cfg.seed + 5)
        self.pool = Pool(p.pool_cap, p.decay, p.pfsp, p.uniform)
        self.guard = Guard(p)
        self.steps = self.iteration = 0
        self.last_snapshot = 0

    # --- setup --------------------------------------------------------------------------
    def init_learners(self) -> None:
        for side, algo, path in (
            ("human", self.human, self.cfg.human_init),
            ("zombie", self.zombie, self.cfg.zombie_init),
        ):
            if not path:
                continue
            sd = torch.load(resolve(path), map_location=self.device, weights_only=False)
            require_obs_version(sd, OBS_VERSION, f"{side}_init {path}")
            algo.actor.load_state_dict(sd["actor"])
            algo.critic.load_state_dict(sd["critic"])
            algo.value_rms.load_state_dict(sd["value_rms"])
        # Seeds: the starters themselves, then the extra entries.
        seeds = [
            Seed(s, p, f"{s}-starter", "distilled")
            for s, p in (("human", self.cfg.human_init), ("zombie", self.cfg.zombie_init))
            if p
        ]
        for i, s in enumerate(seeds + list(self.cfg.seeds)):
            sd = torch.load(resolve(s.path), map_location="cpu", weights_only=False)
            require_obs_version(sd, OBS_VERSION, f"pool seed {s.path}")
            entry_id = s.id or f"{s.side}-seed{i}"
            self.pool.add(
                Entry(
                    entry_id,
                    s.side,
                    str(resolve(s.path)),
                    int(sd.get("steps", 0)),
                    self.cfg.stage,
                    s.source,
                    protected=True,
                )
            )

    def snapshot(self) -> None:
        for side, algo in (("human", self.human), ("zombie", self.zombie)):
            entry_id = self.pool.next_id(side)
            path = self.dir / "pool" / side / f"{entry_id}.pt"
            save_snapshot(path, algo.actor, side, self.steps, self.cfg.stage)
            self.pool.add(Entry(entry_id, side, str(path), self.steps, self.cfg.stage, "league"))
        self.last_snapshot = self.steps
        log.info(
            "snapshot at %d steps: pool %d humans, %d zombies",
            self.steps,
            len(self.pool.entries["human"]),
            len(self.pool.entries["zombie"]),
        )

    # --- persistence --------------------------------------------------------------------
    def save(self) -> None:
        extra = {
            "iteration": self.iteration,
            "steps": self.steps,
            "run_id": self.run_id,
            "env_config": self.env_cfg,
            "league": True,
        }
        root = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints"))
        for side, algo in (("human", self.human), ("zombie", self.zombie)):
            save_checkpoint(root / side / self.run_id / "latest.pt", algo, extra)
            export_training_model(
                algo.actor, f"{self.run_id}-{side}", self.env_cfg, self.iteration, self.steps
            )
        state = {
            "iteration": self.iteration,
            "steps": self.steps,
            "last_snapshot": self.last_snapshot,
            "pool": self.pool.state_dict(),
            "guard": vars(self.guard) | {"p": None},
            "rng": self.rng.bit_generator.state,
            "config": asdict(self.cfg),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "league.tmp"
        torch.save(state, tmp)
        tmp.replace(self.dir / "league.pt")
        self.write_pool()

    def resume(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        root = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints"))
        for side, algo in (("human", self.human), ("zombie", self.zombie)):
            sd = torch.load(
                root / side / self.run_id / "latest.pt",
                map_location=self.device,
                weights_only=False,
            )
            algo.load_state_dict(sd)
        self.iteration, self.steps = state["iteration"], state["steps"]
        self.last_snapshot = state["last_snapshot"]
        self.pool = Pool.from_state_dict(state["pool"])
        for k, v in state["guard"].items():
            if k != "p":
                setattr(self.guard, k, v)
        self.rng.bit_generator.state = state["rng"]

    def write_pool(self) -> None:
        self.pool.write_summary(self.dir / "pool.json")
        if self.public is not None:
            self.pool.write_summary(self.public / "pool.json")

    # --- evaluation ---------------------------------------------------------------------
    def anchors(self) -> dict[str, float]:
        """Latest learners against the scripted bots on fixed seeds (the gates' anchors)."""
        dev, n = self.device, self.cfg.eval_episodes
        human = self.human.actor
        zomb = self.zombie.actor
        out = {}
        r = catch_rates(
            self.env_cfg,
            self.cfg.zombie_counts,
            lambda c, e, s: RecurrentHumanPolicy(human, e, dev),
            lambda c, e, k, s: ScriptedZombies(c, e, k, seed=s),
            n,
            777,
            "x",
        )
        out["anchor/survived"] = r["x/survived"]
        out["anchor/human_caught"] = r["x/caught"]
        r = catch_rates(
            self.env_cfg,
            self.cfg.zombie_counts,
            lambda c, e, s: ScriptedHuman(c, e, seed=s),
            lambda c, e, k, s: RecurrentZombiePolicy(zomb, e, k, dev),
            n,
            777,
            "x",
        )
        out["anchor/caught"] = r["x/caught"]
        out["anchor/caught_lo"], out["anchor/caught_hi"] = r["x/caught_lo"], r["x/caught_hi"]
        return out

    # --- training -----------------------------------------------------------------------
    def run(self) -> None:
        cfg, env, p = self.cfg, self.env, self.cfg.league
        dev = self.device
        hlog = MetricsLogger(
            self.run_id,
            "human",
            os.environ.get("ZAI_SERVER_URL"),
            Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
        )
        zlog = MetricsLogger(
            self.run_id,
            "zombie",
            os.environ.get("ZAI_SERVER_URL"),
            Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
        )
        if self.steps == 0 and not (cfg.human_init and cfg.zombie_init):
            # Learners without a starter checkpoint enter the pool at their starting point.
            self.snapshot()
        opp_z = OpponentSlots(
            "zombie",
            env,
            self.pool,
            dev,
            self.rng,
            p.slots,
            p.scripted_share,
            p.lease_iters,
            cfg.seed + 101,
        )
        opp_h = OpponentSlots(
            "human",
            env,
            self.pool,
            dev,
            self.rng,
            p.slots,
            p.scripted_share,
            p.lease_iters,
            cfg.seed + 202,
        )
        eh, az = len(env.human_rows), len(env.zombie_agents)
        hro = HumanRollout(
            self.human,
            self.human.actor.initial_state(eh, dev),
            torch.ones(eh, dtype=torch.bool, device=dev),
            np.zeros(eh),
        )
        zro = ZombieRollout(
            self.zombie,
            env.zombie_agent_env,
            self.zombie.actor.initial_state(az, dev),
            torch.ones(az, dtype=torch.bool, device=dev),
            np.zeros(az),
        )
        obs = env.reset()
        per_iter = cfg.human.rollout * env.num_envs
        decision_s = self.env_cfg["dt_s"] * self.env_cfg["action_repeat"]
        window: deque = deque()  # group A (outcome, length_s, stats) of the guard window
        a_rows = env.kind == "A"
        started = time.time()
        timer.take()
        log.info(
            "league %s: %d envs (%s), %d human rows, %d zombie agents",
            self.run_id,
            env.num_envs,
            ", ".join(f"{g.kind}{g.zombie_count}x{g.num_envs}" for g in self.groups),
            eh,
            az,
        )
        while self.steps < cfg.total_steps:
            t0 = time.time()
            hstats, zstats = EpisodeStats(), EpisodeStats()
            hro.begin()
            zro.begin()
            for _ in range(cfg.human.rollout):
                ha = np.zeros((env.num_envs, 2), dtype=np.int64)
                za = np.zeros(env.num_agents, dtype=np.int64)
                ha[env.human_rows] = hro.act(
                    obs.human_obs[env.human_rows],
                    obs.human_mask[env.human_rows],
                    obs.state[env.human_rows],
                )
                za[env.zombie_agents] = zro.act(
                    obs.zombie_obs[env.zombie_agents], obs.state[env.zombie_envs]
                )
                with timer("opponent"):
                    ha[opp_h.rows] = opp_h.act(obs)
                    za[opp_z.rows] = opp_z.act(obs)
                step = env.step(ha, za)
                hro.record(env.human_view(step), hstats)
                zro.record(env.zombie_view(step), zstats)
                done = step.terminated | step.truncated
                with timer("opponent"):
                    opp_z.episodes_ended(done, step.outcome, human_score)
                    opp_h.episodes_ended(done, step.outcome, zombie_score)
                for e in np.nonzero(done & a_rows)[0]:
                    window.append(
                        (
                            int(step.outcome[e]),
                            float(step.episode_length[e]) * decision_s,
                            step.final_stats[e].copy(),
                        )
                    )
                obs = step.next
            t1 = time.time()
            hr = hro.finish(obs.state[env.human_rows])
            zr = zro.finish(obs.zombie_obs[env.zombie_agents], obs.state[env.zombie_envs])
            progress = self.steps / cfg.total_steps
            he, hs = self.guard.overrides("human", cfg.human.epochs)
            ze, zs = self.guard.overrides("zombie", cfg.zombie.epochs)
            hstat = self.human.update(hr, progress, he, hs)
            zstat = self.zombie.update(zr, progress, ze, zs)
            t2 = time.time()
            self.iteration += 1
            self.steps += per_iter
            self.pool.end_iteration()
            opp_z.end_iteration(self.iteration)
            opp_h.end_iteration(self.iteration)

            hs_ = dict(hstat)
            zs_ = dict(zstat)
            perf = {
                "perf/steps_per_s": per_iter / (t2 - t0),
                "perf/rollout_s": t1 - t0,
                "perf/update_s": t2 - t1,
            }
            perf.update(timer.take())
            rollout_split(perf)
            hs_.update(perf)
            zs_["train/nearest_zombie_m"] = zr["nearest_zombie_m"].item()
            for stats, out in ((hstats, hs_), (zstats, zs_)):
                if stats.outcomes:
                    o = np.array(stats.outcomes)
                    out["episode/count"] = len(o)
                    out["episode/return"] = float(np.mean(stats.returns))
                    for code, name in OUTCOME_KEYS.items():
                        out[f"episode/{name}"] = float((o == code).mean())
            if self.iteration % p.guard_window == 0:
                outcomes = np.array([w[0] for w in window])
                s_z = self.guard.update(outcomes)
                league = {
                    "league/s_z": s_z,
                    "league/guard": self.guard.state,
                    "league/starvation_alarm": float(self.guard.alarm),
                }
                if len(outcomes):
                    for code, name in OUTCOME_KEYS.items():
                        league[f"episode_A/{name}"] = float((outcomes == code).mean())
                    tm = tactic_metrics(
                        np.stack([w[2] for w in window]), outcomes, np.array([w[1] for w in window])
                    )
                    league.update({f"tactics_train/{k}": v for k, v in tm.items()})
                window.clear()
                for side, out in (("human", hs_), ("zombie", zs_)):
                    w = self.pool.weights(side)
                    out["pool/size"] = len(self.pool.entries[other(side)])
                    out["pool/pfsp_entropy"] = float(
                        -sum(x * np.log(x) for x in w.values() if x > 0)
                    )
                    ps = [self.pool.p(side, e) for e in w]
                    if ps:
                        out["pool/score_min"], out["pool/score_mean"] = min(ps), float(np.mean(ps))
                hs_.update(league)
                if self.guard.alarm:
                    log.warning(
                        "starvation alarm: group A ends in starvation > %.0f%%",
                        100 * p.starvation_alarm,
                    )
            if cfg.eval_every and self.iteration % cfg.eval_every == 0:
                t_eval = time.time()
                ev = self.anchors()
                hs_.update(ev)
                hs_["perf/eval_s"] = time.time() - t_eval
            if self.steps - self.last_snapshot >= p.snapshot_every_steps:
                self.snapshot()
            hlog.log(self.steps, hs_)
            zlog.log(self.steps, zs_)
            log.info(
                "it %d steps %d | human survived %.2f | zombies caught %.2f | s_z %s guard %d"
                " | resident z %s h %s | %.0f steps/s",
                self.iteration,
                self.steps,
                hs_.get("episode/survived", float("nan")),
                zs_.get("episode/caught", float("nan")),
                f"{hs_['league/s_z']:.2f}" if "league/s_z" in hs_ else "-",
                self.guard.state,
                opp_z.resident(),
                opp_h.resident(),
                perf["perf/steps_per_s"],
            )
            if self.iteration % cfg.checkpoint_every == 0 or self.steps >= cfg.total_steps:
                self.save()
        hlog.close()
        zlog.close()
        log.info("done: %d steps in %.0f s", self.steps, time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="League co-training of both sides (M6).")
    parser.add_argument("--config", type=Path, default=Path("configs/train/league_cpu.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path, help="league.pt to resume from")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = LeagueConfig.load(
        resolve(str(args.config)),
        {"run_id": args.run_id, "total_steps": args.total_steps, "device": args.device},
    )
    run_id = cfg.run_id or time.strftime("league-%Y%m%d-%H%M%S")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    league = League(cfg, run_id, device)
    if args.resume:
        league.resume(args.resume)
    else:
        league.init_learners()
    league.run()


if __name__ == "__main__":
    main()
