"""Snapshot pool with prioritized fictitious self-play (docs/ARCHITECTURE.md section 4.8.5).

Entries are frozen policies of one side: league snapshots, distilled starters, probe
exploiters, and the scripted bot (no file). The learner of the other side keeps a decayed score
against each entry from its training episodes (win 1, starvation 0.5, loss 0; `n <- decay * n
+ games`, `w <- decay * w + score` once per iteration, estimate `p = (w + 1) / (n + 2)`), and
opponents are sampled with weight `f(p)` mixed with a uniform share:

- `hard`: `(1 - p)^2`, the AlphaStar weighting, which keeps sampling opponents the learner
  never beats (the forgetting case the pool is meant to catch);
- `variance`: `p (1 - p)`, section 4.3's weighting, kept for ablation.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SIDES = ("human", "zombie")
SCRIPTED = "scripted"


def other(side: str) -> str:
    return "zombie" if side == "human" else "human"


@dataclass
class Entry:
    id: str
    side: str
    path: str | None  # actor file; None for the scripted bot
    steps: int = 0  # learner env steps when the snapshot was taken
    stage: str = ""
    source: str = "league"  # league | distilled | scripted | probe
    protected: bool = False  # never evicted
    created: float = field(default_factory=time.time)

    @property
    def scripted(self) -> bool:
        return self.path is None


@dataclass
class Score:
    n: float = 0.0
    w: float = 0.0
    pending_n: float = 0.0
    pending_w: float = 0.0
    games: int = 0  # undecayed total, for reports

    @property
    def p(self) -> float:
        return (self.w + 1.0) / (self.n + 2.0)


class Pool:
    def __init__(
        self,
        cap: int = 50,
        decay: float = 0.98,
        pfsp: str = "hard",
        uniform: float = 0.1,
        protect_every: int = 10,
    ):
        if pfsp not in ("hard", "variance"):
            raise ValueError("pfsp must be 'hard' or 'variance'")
        self.cap, self.decay, self.pfsp, self.uniform = cap, decay, pfsp, uniform
        self.protect_every = protect_every
        self.entries: dict[str, list[Entry]] = {s: [] for s in SIDES}
        self.evicted: dict[str, list[Entry]] = {s: [] for s in SIDES}
        # scores[learner side][opponent entry id]
        self.scores: dict[str, dict[str, Score]] = {s: {} for s in SIDES}
        self.snapshots: dict[str, int] = {s: 0 for s in SIDES}
        for side in SIDES:
            self.add(
                Entry(
                    id=f"{side}-{SCRIPTED}", side=side, path=None, source=SCRIPTED, protected=True
                )
            )

    # --- entries ------------------------------------------------------------------------
    def get(self, entry_id: str) -> Entry:
        for side in SIDES:
            for e in self.entries[side]:
                if e.id == entry_id:
                    return e
        raise KeyError(entry_id)

    def scripted(self, side: str) -> Entry:
        return next(e for e in self.entries[side] if e.scripted)

    def add(self, entry: Entry) -> Entry:
        if any(e.id == entry.id for s in SIDES for e in self.entries[s]):
            raise ValueError(f"duplicate pool entry {entry.id}")
        if entry.source == "league":
            self.snapshots[entry.side] += 1
            if self.snapshots[entry.side] % self.protect_every == 0:
                entry.protected = True
        self.entries[entry.side].append(entry)
        self._evict(entry.side)
        return entry

    def next_id(self, side: str) -> str:
        return f"{side[0]}{self.snapshots[side] + 1:04d}"

    def _evict(self, side: str) -> None:
        live = self.entries[side]
        while len(live) > self.cap:
            victims = [e for e in live if not e.protected]
            if not victims:
                return
            oldest = min(victims, key=lambda e: e.created)
            live.remove(oldest)
            self.evicted[side].append(oldest)

    # --- scores -------------------------------------------------------------------------
    def record(self, learner: str, entry_id: str, score: float) -> None:
        """One training episode of `learner` against entry `entry_id` (score in [0, 1])."""
        s = self.scores[learner].setdefault(entry_id, Score())
        s.pending_n += 1.0
        s.pending_w += score
        s.games += 1

    def end_iteration(self) -> None:
        """Applies the decay and folds in this iteration's games."""
        for by_entry in self.scores.values():
            for s in by_entry.values():
                s.n = self.decay * s.n + s.pending_n
                s.w = self.decay * s.w + s.pending_w
                s.pending_n = s.pending_w = 0.0

    def p(self, learner: str, entry_id: str) -> float:
        s = self.scores[learner].get(entry_id)
        return s.p if s is not None else 0.5

    def weights(self, learner: str, include_scripted: bool = False) -> dict[str, float]:
        """Sampling weights over the opponent side's entries for `learner` (sum to 1)."""
        cands = [e for e in self.entries[other(learner)] if include_scripted or not e.scripted]
        if not cands:
            return {}
        f = np.array([self._f(self.p(learner, e.id)) for e in cands])
        f = f / f.sum() if f.sum() > 0 else np.full(len(cands), 1.0 / len(cands))
        w = (1.0 - self.uniform) * f + self.uniform / len(cands)
        return {e.id: float(x) for e, x in zip(cands, w, strict=True)}

    def _f(self, p: float) -> float:
        return (1.0 - p) ** 2 if self.pfsp == "hard" else p * (1.0 - p)

    def sample(self, learner: str, rng: np.random.Generator) -> Entry | None:
        """A non-scripted opponent entry for `learner`, or None if the pool has none."""
        w = self.weights(learner)
        if not w:
            return None
        ids = list(w)
        return self.get(ids[int(rng.choice(len(ids), p=np.array(list(w.values()))))])

    # --- persistence --------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "cap": self.cap,
                "decay": self.decay,
                "pfsp": self.pfsp,
                "uniform": self.uniform,
                "protect_every": self.protect_every,
            },
            "entries": {s: [asdict(e) for e in v] for s, v in self.entries.items()},
            "evicted": {s: [asdict(e) for e in v] for s, v in self.evicted.items()},
            "scores": {s: {k: asdict(v) for k, v in d.items()} for s, d in self.scores.items()},
            "snapshots": dict(self.snapshots),
        }

    @classmethod
    def from_state_dict(cls, sd: dict[str, Any]) -> "Pool":
        pool = cls(**sd["config"])
        pool.entries = {s: [Entry(**e) for e in v] for s, v in sd["entries"].items()}
        pool.evicted = {s: [Entry(**e) for e in v] for s, v in sd["evicted"].items()}
        pool.scores = {s: {k: Score(**v) for k, v in d.items()} for s, d in sd["scores"].items()}
        pool.snapshots = dict(sd["snapshots"])
        return pool

    def summary(self) -> dict[str, Any]:
        """`pool.json`: entries with their scores and current sampling weights."""
        out: dict[str, Any] = {"updated": time.time(), "pfsp": self.pfsp, "sides": {}}
        for side in SIDES:
            learner = other(side)
            weights = self.weights(learner)
            out["sides"][side] = [
                {
                    **asdict(e),
                    "p_learner": self.p(learner, e.id),
                    "games": self.scores[learner].get(e.id, Score()).games,
                    "weight": weights.get(e.id, 0.0),
                }
                for e in self.entries[side]
            ]
        return out

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.summary(), indent=1))
        tmp.replace(path)
