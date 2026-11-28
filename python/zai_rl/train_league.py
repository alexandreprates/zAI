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
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from arena.export_model import export_training_model
from arena.export_replays import record_training_replays, training_replay_dir
from arena.tactics import tactic_metrics
from zai_env.config import load_config
from zai_env.layout import OBS_VERSION
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED

from .algos.common import EpisodeStats, require_obs_version, save_checkpoint
from .algos.mappo_zombie import MAPPOConfig, ZombieMAPPO, ZombieRollout
from .algos.ppo_human import HumanPPO, HumanRollout, PPOConfig
from .bots import ScriptedHuman, ScriptedZombies
from .envs import MatchTrainingEnv, league_groups
from .league.gates import GateInputs, GateSpec, GateTracker, margins, passes
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
    # scale: the winning side updates at half the lr; freeze: it skips its updates while the
    # guard is engaged (docs/ARCHITECTURE.md section 4.8.9). The losing side gets +1 epoch.
    guard_mode: str = "scale"
    starvation_alarm: float = 0.5


@dataclass
class Seed:
    side: str
    path: str
    id: str | None = None
    source: str = "distilled"


@dataclass
class StageSpec:
    """One curriculum stage (4.8.1): trained for at least `min_steps` and at most `max_steps`
    env steps; with a `gate`, the league advances to the next stage when it passes and stops at
    `max_steps` when it does not."""

    name: str
    env_config: str
    zombie_counts: list[int]
    min_steps: int = 0
    max_steps: int = 100_000_000
    gate: GateSpec | None = None


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
    # On CUDA, replay the learners' and resident opponents' decision steps as CUDA graphs
    # (zai_rl.models.graphs): the rollout is bound by kernel launches otherwise.
    cuda_graphs: bool = True
    human_init: str | None = None  # layout v2 checkpoints the learners start from
    zombie_init: str | None = None
    seeds: list[Seed] = field(default_factory=list)  # extra pool entries
    eval_every: int = 20  # iterations; anchor evaluation against the scripted bots (0: off)
    eval_episodes: int = 64  # per zombie count
    checkpoint_every: int = 20
    # At iteration 1 and every `replay_every` iterations (0 disables), record `replay_episodes`
    # learner-vs-learner episodes on fixed seeds with the stage's largest zombie count, as
    # viewer replays under `match_id = run_id` (needs `ZAI_DATA_DIR`).
    replay_every: int = 0
    replay_episodes: int = 2
    # Curriculum (M6.5): without `stages`, one stage from `stage`, `env_config`,
    # `zombie_counts` and `total_steps`, without a gate.
    stages: list[StageSpec] = field(default_factory=list)
    gate_every: int = 20  # iterations between gate checks, after a stage's minimum budget
    gate_episodes: int = 128  # per zombie count, for the anchors and the baselines
    # Run both exploitability probes at a gate check once every other condition passes, unless
    # the latest probe of the stage is younger than `probe_max_age_steps`.
    probe_at_gates: bool = False
    probe_max_age_steps: int = 50_000_000
    probe_steps: int = 10_000_000
    probe_every_steps: int = 0  # also every this many env steps (0: off)
    forgetting_age_steps: int = 40_000_000
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
        if lg.get("guard_mode", "scale") not in ("scale", "freeze"):
            raise ValueError("league.guard_mode must be 'scale' or 'freeze'")
        groups = league_groups(
            raw.get("num_envs", 1024),
            raw.get("zombie_counts", [3, 6, 9]),
            lg.get("shares", LeagueParams.shares),
        )
        rows_h = sum(g.num_envs for g in groups if g.kind in "AB")
        human = PPOConfig(**{**raw.pop("human", {}), "num_envs": rows_h})
        zombie = MAPPOConfig(**{**raw.pop("zombie", {}), "num_envs": rows_h})
        seeds = [Seed(**s) for s in raw.pop("seeds", [])]
        stages = []
        for st in raw.pop("stages", []):
            gate = st.pop("gate", None)
            if gate is not None:
                if "balance" in gate:
                    gate["balance"] = tuple(gate["balance"])
                gate = GateSpec(**gate)
            stages.append(StageSpec(**st, gate=gate))
        return cls(
            **raw,
            league=LeagueParams(**lg),
            human=human,
            zombie=zombie,
            seeds=seeds,
            stages=stages,
        )

    def stage_list(self) -> list[StageSpec]:
        if self.stages:
            return self.stages
        return [StageSpec(self.stage, self.env_config, self.zombie_counts, 0, self.total_steps)]


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
        self.frozen = {"human": 0, "zombie": 0}  # iterations each side skipped its update

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

    def overrides(self, side: str, epochs: int) -> tuple[int, float, bool]:
        """`(epochs, lr_scale, frozen)` of `side`'s next update; a frozen side skips it."""
        losing = (self.state == 1 and side == "human") or (self.state == -1 and side == "zombie")
        winning = (self.state == 1 and side == "zombie") or (self.state == -1 and side == "human")
        if self.p.guard_mode == "freeze":
            return epochs + int(losing), 1.0, winning
        return epochs + int(losing), 0.5 if winning else 1.0, False


class League:
    def __init__(self, cfg: LeagueConfig, run_id: str, device: torch.device):
        self.cfg, self.run_id, self.device = cfg, run_id, device
        p = cfg.league
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
        self.updated_since_snapshot = {"human": True, "zombie": True}
        self.last_probe = 0
        self.stage_index = 0
        self.stage_start = 0
        self.starters: dict[str, str] = {}  # side -> pool entry the stage started from
        self.baseline: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []  # finished stages
        self.last_s_z = float("nan")
        self.stopped: str | None = None
        self.tracker: GateTracker | None = None
        self._build_stage()

    # --- stages -------------------------------------------------------------------------
    @property
    def stage(self) -> StageSpec:
        return self.cfg.stage_list()[self.stage_index]

    def _build_stage(self) -> None:
        st = self.stage
        self.env_cfg = load_config(resolve(st.env_config), env_overrides=False)
        self.groups = league_groups(self.cfg.num_envs, st.zombie_counts, self.cfg.league.shares)
        self.env = MatchTrainingEnv(
            self.env_cfg, self.groups, self.cfg.seed + 7919 * self.stage_index, self.cfg.num_threads
        )
        self.tracker = GateTracker(st.gate) if st.gate is not None else None

    def measure_baselines(self) -> None:
        """Scripted human vs scripted zombies on this stage, on the gates' seeds."""
        r = catch_rates(
            self.env_cfg,
            self.stage.zombie_counts,
            lambda c, e, s: ScriptedHuman(c, e, seed=s),
            lambda c, e, k, s: ScriptedZombies(c, e, k, seed=s),
            self.cfg.gate_episodes,
            777,
            "x",
        )
        self.baseline = {
            "survived": r["x/survived"],
            "caught": r["x/caught"],
            "caught_hi": r["x/caught_hi"],
        }
        log.info(
            "stage %s baselines (scripted vs scripted): survived %.1f%%, caught %.1f%% [.., %.1f]",
            self.stage.name,
            100 * self.baseline["survived"],
            100 * self.baseline["caught"],
            100 * self.baseline["caught_hi"],
        )

    def advance(self) -> None:
        """Next stage: the learners keep their weights and value normalization (fresh
        optimizers), the pool restarts from the scripted bots and this stage's finals, the
        score statistics and the guard reset, and the baselines are measured again."""
        finished = self.stage.name
        finals = {}
        for side, algo in (("human", self.human), ("zombie", self.zombie)):
            entry_id = f"{side}-{finished}-final"
            path = self.dir / "pool" / side / f"{entry_id}.pt"
            save_snapshot(path, algo.actor, side, self.steps, finished)
            finals[side] = Entry(
                entry_id, side, str(path), self.steps, finished, "stage_final", protected=True
            )
        self.history.append(
            {"stage": finished, "start": self.stage_start, "end": self.steps, "result": "passed"}
        )
        p = self.cfg.league
        self.pool = Pool(p.pool_cap, p.decay, p.pfsp, p.uniform)
        for e in finals.values():
            self.pool.add(e)
        self.starters = {side: e.id for side, e in finals.items()}
        frozen = self.guard.frozen
        self.guard = Guard(p)
        self.guard.frozen = frozen  # frozen_frac_* covers the whole run
        self.human.optimizer = torch.optim.Adam(
            list(self.human.actor.parameters()) + list(self.human.critic.parameters()),
            lr=self.cfg.human.lr,
            eps=1e-5,
        )
        self.zombie.optimizer = torch.optim.Adam(
            list(self.zombie.actor.parameters()) + list(self.zombie.critic.parameters()),
            lr=self.cfg.zombie.lr,
            eps=1e-5,
        )
        self.stage_index += 1
        self.stage_start = self.last_snapshot = self.last_probe = self.steps
        self._build_stage()
        self.measure_baselines()
        log.info("advanced from %s to %s at %d steps", finished, self.stage.name, self.steps)

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
                    self.stage.name,
                    s.source,
                    protected=True,
                )
            )
        if not (self.cfg.human_init and self.cfg.zombie_init):
            # Learners without a starter checkpoint enter the pool at their starting point.
            self.snapshot()
        for side in ("human", "zombie"):
            starter = f"{side}-starter"
            if any(e.id == starter for e in self.pool.entries[side]):
                self.starters[side] = starter
            else:
                first = [e for e in self.pool.entries[side] if e.source == "league"]
                if first:
                    self.starters[side] = first[0].id
        self.measure_baselines()

    def snapshot(self) -> None:
        """Adds both learners to the pool, except a side that has not updated since its last
        snapshot (frozen by the guard), which would only duplicate it."""
        for side, algo in (("human", self.human), ("zombie", self.zombie)):
            if not self.updated_since_snapshot[side]:
                continue
            self.updated_since_snapshot[side] = False
            entry_id = self.pool.next_id(side)
            path = self.dir / "pool" / side / f"{entry_id}.pt"
            save_snapshot(path, algo.actor, side, self.steps, self.stage.name)
            self.pool.add(Entry(entry_id, side, str(path), self.steps, self.stage.name, "league"))
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
            "stage": self.stage.name,
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
            "updated_since_snapshot": self.updated_since_snapshot,
            "last_probe": self.last_probe,
            "stage_index": self.stage_index,
            "stage_start": self.stage_start,
            "starters": self.starters,
            "baseline": self.baseline,
            "history": self.history,
            "streak": self.tracker.streak if self.tracker else 0,
            "stopped": self.stopped,
            "pool": self.pool.state_dict(),
            "guard": {k: v for k, v in vars(self.guard).items() if k != "p"},
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
        self.updated_since_snapshot = state.get(
            "updated_since_snapshot", {"human": True, "zombie": True}
        )
        self.last_probe = state.get("last_probe", self.steps)
        if state.get("stage_index", 0) != self.stage_index:
            self.stage_index = state["stage_index"]
            self._build_stage()
        self.stage_start = state.get("stage_start", 0)
        self.starters = state.get("starters", {})
        self.baseline = state.get("baseline", {})
        self.history = state.get("history", [])
        if self.tracker is not None:
            self.tracker.streak = state.get("streak", 0)
        self.pool = Pool.from_state_dict(state["pool"])
        for k, v in state["guard"].items():
            setattr(self.guard, k, v)
        self.guard.frozen = dict(state["guard"].get("frozen", {"human": 0, "zombie": 0}))
        self.rng.bit_generator.state = state["rng"]
        if not self.baseline:
            self.measure_baselines()

    def write_pool(self) -> None:
        st = self.stage
        extra = {
            "stage": {
                "name": st.name,
                "index": self.stage_index,
                "env_config": st.env_config,
                "zombie_counts": st.zombie_counts,
                "start_steps": self.stage_start,
                "starters": self.starters,
            },
            "steps": self.steps,
            "history": self.history,
            "stopped": self.stopped,
            "gate": self.tracker.last if self.tracker else {},
        }
        self.pool.write_summary(self.dir / "pool.json", extra)
        if self.public is not None:
            self.pool.write_summary(self.public / "pool.json", extra)

    # --- evaluation ---------------------------------------------------------------------
    def anchors(self, episodes: int | None = None) -> dict[str, float]:
        """Latest learners against the scripted bots on fixed seeds (the gates' anchors)."""
        dev, n = self.device, episodes or self.cfg.eval_episodes
        human = self.human.actor
        zomb = self.zombie.actor
        out = {}
        r = catch_rates(
            self.env_cfg,
            self.stage.zombie_counts,
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
            self.stage.zombie_counts,
            lambda c, e, s: ScriptedHuman(c, e, seed=s),
            lambda c, e, k, s: RecurrentZombiePolicy(zomb, e, k, dev),
            n,
            777,
            "x",
        )
        out["anchor/caught"] = r["x/caught"]
        out["anchor/caught_lo"], out["anchor/caught_hi"] = r["x/caught_lo"], r["x/caught_hi"]
        for k in ("survived", "caught"):
            if k in self.baseline:
                out[f"anchor/baseline_{k}"] = self.baseline[k]
        return out

    def record_replays(self) -> bool:
        """Records the current learners against each other as viewer replays."""
        out_dir = training_replay_dir()
        if out_dir is None:
            return False
        n = max(self.stage.zombie_counts)
        show_cfg = {**self.env_cfg, "zombie_count": n}
        dev, human, zomb = self.device, self.human.actor, self.zombie.actor
        recorded = record_training_replays(
            out_dir,
            self.run_id,
            self.iteration,
            show_cfg,
            lambda seed: (RecurrentHumanPolicy(human, 1, dev), "policy"),
            lambda seed: (RecurrentZombiePolicy(zomb, 1, n, dev), "policy"),
            self.cfg.replay_episodes,
        )
        log.info("replays: %s", ", ".join(f"{r['id']} {r['outcome']}" for r in recorded))
        return True

    def run_probes(self) -> dict[str, float]:
        """Exploitability probes of both sides at the current point (blocking, `arena.probe`)."""
        self.save()
        cmd = [
            sys.executable,
            "-m",
            "arena.probe",
            "--run-id",
            self.run_id,
            "--side",
            "both",
            "--steps",
            str(self.cfg.probe_steps),
            "--episodes",
            str(self.cfg.gate_episodes),
        ]
        subprocess.run(cmd, check=True)
        self.last_probe = self.steps
        return self.read_probes()

    def read_probes(self) -> dict[str, float]:
        """Latest probe gap per side since this stage started."""
        out: dict[str, float] = {}
        path = self.dir / "probes.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r["league_steps"] >= self.stage_start:
                    out[r["side"]] = r["gap"]
        return out

    def read_elo(self) -> tuple[dict[str, tuple], dict[str, tuple]]:
        """`(latest, starter)` Elo `(rating, lo, hi)` per side from this stage's arena report."""
        try:
            arena = json.loads((self.dir / "arena.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}, {}
        if arena.get("stage") != self.stage.name:
            return {}, {}
        by_id = {e["id"]: (e["elo"], e["elo_lo"], e["elo_hi"]) for e in arena["entries"]}
        latest = {s: by_id[i] for s, i in arena["latest"].items() if i in by_id}
        starter = {s: by_id[i] for s, i in self.starters.items() if i in by_id}
        return latest, starter

    def forgetting(self) -> dict[str, float]:
        """Each learner's training score against this stage's snapshots older than 40 M steps."""
        out = {}
        for side in ("human", "zombie"):
            old = [
                e.id
                for e in self.pool.entries[other(side)]
                if e.source == "league"
                and e.stage == self.stage.name
                and e.steps <= self.steps - self.cfg.forgetting_age_steps
            ]
            out[side] = float(np.mean([self.pool.p(side, i) for i in old])) if old else float("nan")
        return out

    def check_gate(self) -> dict[str, float]:
        """Evaluates the stage gate; returns the `gate/*` and anchor scalars. Probes run only
        when every other condition passes (they take about 20 min per side)."""
        assert self.tracker is not None
        anchors = self.anchors(self.cfg.gate_episodes)
        elo, elo_starter = self.read_elo()
        x = GateInputs(
            survived=anchors["anchor/survived"],
            caught=anchors["anchor/caught"],
            caught_lo=anchors["anchor/caught_lo"],
            baseline_survived=self.baseline["survived"],
            baseline_caught=self.baseline["caught"],
            baseline_caught_hi=self.baseline["caught_hi"],
            s_z=self.last_s_z,
            forgetting=self.forgetting(),
            elo=elo,
            elo_starter=elo_starter,
            probe_gap=self.read_probes(),
        )
        m = margins(self.tracker.spec, x)
        others_pass = passes({k: v for k, v in m.items() if not k.startswith("probe_")})
        stale = self.last_probe < self.stage_start or (
            self.steps - self.last_probe >= self.cfg.probe_max_age_steps
        )
        if self.cfg.probe_at_gates and others_pass and stale:
            x.probe_gap = self.run_probes()
        self.tracker.update(x)
        log.info(
            "gate %s: streak %d, failing %s",
            self.stage.name,
            self.tracker.streak,
            self.tracker.failing() or "none",
        )
        return anchors | self.tracker.scalars()

    # --- training -----------------------------------------------------------------------
    def run(self) -> None:
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
        started = time.time()
        while self.stopped is None and self.steps < self.cfg.total_steps:
            outcome = self.train_stage(hlog, zlog)
            if outcome == "passed" and self.stage_index + 1 < len(self.cfg.stage_list()):
                self.advance()
                self.save()
            elif outcome != "running":
                self.stopped = outcome
                self.history.append(
                    {
                        "stage": self.stage.name,
                        "start": self.stage_start,
                        "end": self.steps,
                        "result": outcome,
                    }
                )
                self.save()
        hlog.close()
        zlog.close()
        log.info(
            "done: %d steps in %.0f s (%s)",
            self.steps,
            time.time() - started,
            self.stopped or "budget reached",
        )

    def train_stage(self, hlog: MetricsLogger, zlog: MetricsLogger) -> str:
        """Trains on the current stage until its gate passes (`passed`), its maximum budget is
        used up (`gate_failed` with a gate, `completed` without), or `total_steps` ends
        (`running`)."""
        cfg, env, p, st = self.cfg, self.env, self.cfg.league, self.stage
        dev = self.device
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
            graph=cfg.cuda_graphs,
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
            graph=cfg.cuda_graphs,
        )
        eh, az = len(env.human_rows), len(env.zombie_agents)
        hro = HumanRollout(
            self.human,
            self.human.actor.initial_state(eh, dev),
            torch.ones(eh, dtype=torch.bool, device=dev),
            np.zeros(eh),
            graph=cfg.cuda_graphs,
        )
        zro = ZombieRollout(
            self.zombie,
            env.zombie_agent_env,
            self.zombie.actor.initial_state(az, dev),
            torch.ones(az, dtype=torch.bool, device=dev),
            np.zeros(az),
            graph=cfg.cuda_graphs,
        )
        obs = env.reset()
        per_iter = cfg.human.rollout * env.num_envs
        decision_s = self.env_cfg["dt_s"] * self.env_cfg["action_repeat"]
        window: deque = deque()  # group A (outcome, length_s, stats) of the guard window
        a_rows = env.kind == "A"
        timer.take()
        log.info(
            "league %s, stage %s: %d envs (%s), %d human rows, %d zombie agents",
            self.run_id,
            st.name,
            env.num_envs,
            ", ".join(f"{g.kind}{g.zombie_count}x{g.num_envs}" for g in self.groups),
            eh,
            az,
        )
        self.write_pool()
        while self.steps < cfg.total_steps:
            in_stage = self.steps - self.stage_start
            if in_stage >= st.max_steps:
                if self.tracker is None:
                    return "completed"
                log.warning(
                    "stage %s: gate not passed within %d steps; failing: %s",
                    st.name,
                    st.max_steps,
                    self.tracker.failing(),
                )
                return "gate_failed"
            t0 = time.time()
            hstats, zstats = EpisodeStats(), EpisodeStats()
            hro.begin()
            zro.begin()
            for _ in range(cfg.human.rollout):
                ha = np.zeros((env.num_envs, 2), dtype=np.int64)
                za = np.zeros(env.num_agents, dtype=np.int64)
                # Both learners' forward passes are queued first; the opponents' transfer
                # then waits for them too, so the learners' own transfers below are cheap.
                ha_t = hro.act_device(
                    obs.human_obs[env.human_sel],
                    obs.human_mask[env.human_sel],
                    obs.state[env.human_sel],
                )
                za_t = zro.act_device(
                    obs.zombie_obs[env.zombie_agent_sel], obs.state[env.zombie_env_sel]
                )
                with timer("opponent"):
                    ha[opp_h.rows] = opp_h.act(obs)
                    za[opp_z.rows] = opp_z.act(obs)
                with timer("policy"):
                    ha[env.human_sel] = ha_t.cpu().numpy()
                    za[env.zombie_agent_sel] = za_t.cpu().numpy()
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
            hr = hro.finish(obs.state[env.human_sel])
            zr = zro.finish(obs.zombie_obs[env.zombie_agent_sel], obs.state[env.zombie_env_sel])
            # Learning-rate schedules run over each stage's maximum budget.
            progress = min(1.0, in_stage / st.max_steps)
            stats_ = {}
            for side, algo, ro, epochs in (
                ("human", self.human, hr, cfg.human.epochs),
                ("zombie", self.zombie, zr, cfg.zombie.epochs),
            ):
                ep, scale, frozen = self.guard.overrides(side, epochs)
                if frozen:
                    self.guard.frozen[side] += 1
                    stats_[side] = {"train/updated": 0.0}
                else:
                    self.updated_since_snapshot[side] = True
                    stats_[side] = {**algo.update(ro, progress, ep, scale), "train/updated": 1.0}
                stats_[side][f"league/frozen_{side}"] = float(frozen)
            hstat, zstat = stats_["human"], stats_["zombie"]
            t2 = time.time()
            self.iteration += 1
            self.steps += per_iter
            self.pool.end_iteration()
            opp_z.end_iteration(self.iteration)
            opp_h.end_iteration(self.iteration)

            hs_ = dict(hstat)
            zs_ = dict(zstat)
            for side, out in (("human", hs_), ("zombie", zs_)):
                out[f"league/frozen_frac_{side}"] = self.guard.frozen[side] / self.iteration
            perf = {
                "perf/steps_per_s": per_iter / (t2 - t0),
                "perf/rollout_s": t1 - t0,
                "perf/update_s": t2 - t1,
            }
            perf.update(timer.take())
            rollout_split(perf)
            hs_.update(perf)
            hs_["league/stage"] = float(self.stage_index)
            zs_["train/nearest_zombie_m"] = zr["nearest_zombie_m"].item()
            for stats, out in ((hstats, hs_), (zstats, zs_)):
                if stats.outcomes:
                    o = np.array(stats.outcomes)
                    out["episode/count"] = len(o)
                    out["episode/return"] = float(np.mean(stats.returns))
                    for code, name in OUTCOME_KEYS.items():
                        out[f"episode/{name}"] = float((o == code).mean())
            if self.iteration % p.guard_window == 0:
                hs_.update(self._window_scalars(window))
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
            budget_end = (
                self.steps >= cfg.total_steps or self.steps - self.stage_start >= st.max_steps
            )
            if self.steps - self.last_snapshot >= p.snapshot_every_steps or (
                budget_end and self.steps > self.last_snapshot
            ):
                self.snapshot()
            passed = False
            gate_due = (
                self.tracker is not None
                and self.steps - self.stage_start >= st.min_steps
                and self.iteration % cfg.gate_every == 0
            )
            t_eval = time.time()
            if gate_due:
                hs_.update(self.check_gate())
                passed = self.tracker.streak >= self.tracker.spec.consecutive
            elif cfg.eval_every and self.iteration % cfg.eval_every == 0:
                hs_.update(self.anchors())
            if cfg.probe_every_steps and self.steps - self.last_probe >= cfg.probe_every_steps:
                for side, gap in self.run_probes().items():
                    hs_[f"probe/{side}_gap"] = gap
            if time.time() - t_eval > 1e-3:
                hs_["perf/eval_s"] = time.time() - t_eval
            if cfg.replay_every and (self.iteration == 1 or self.iteration % cfg.replay_every == 0):
                t_replay = time.time()
                if self.record_replays():
                    hs_["perf/replay_s"] = time.time() - t_replay
            hlog.log(self.steps, hs_)
            zlog.log(self.steps, zs_)
            log.info(
                "it %d steps %d | %s | human survived %.2f | zombies caught %.2f | s_z %s"
                " guard %d | resident z %s h %s | %.0f steps/s",
                self.iteration,
                self.steps,
                st.name,
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
            if passed:
                log.info("stage %s: gate passed at %d steps", st.name, self.steps)
                return "passed"
        return "running"

    def _window_scalars(self, window: deque) -> dict[str, float]:
        """Guard update and group A outcome and tactic metrics of one window."""
        p = self.cfg.league
        outcomes = np.array([w[0] for w in window])
        s_z = self.guard.update(outcomes)
        if np.isfinite(s_z):
            self.last_s_z = s_z
        out = {
            "league/s_z": s_z,
            "league/guard": self.guard.state,
            "league/starvation_alarm": float(self.guard.alarm),
        }
        if len(outcomes):
            for code, name in OUTCOME_KEYS.items():
                out[f"episode_A/{name}"] = float((outcomes == code).mean())
            tm = tactic_metrics(
                np.stack([w[2] for w in window]), outcomes, np.array([w[1] for w in window])
            )
            out.update({f"tactics_train/{k}": v for k, v in tm.items()})
        if self.guard.alarm:
            log.warning(
                "starvation alarm: group A ends in starvation > %.0f%%", 100 * p.starvation_alarm
            )
        return out


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
