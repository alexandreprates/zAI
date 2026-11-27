"""League arena (M6.4, docs/ARCHITECTURE.md section 4.8.6).

Runs next to `zai_rl.train_league` and watches its `pool.json`. Every human of the Elo set
plays every zombie team of the Elo set once, on fixed seeds (`episodes` per training zombie
count); results are appended to `crossplay.jsonl` and never recomputed. The Elo set of a side
is its scripted and random bots, its seed entries (distilled starters, probes) and its last
`elo_last` league snapshots. After each round the arena fits Elo ratings over the whole table
(`arena.elo`, anchored at the scripted zombies), writes `arena.json` (ratings, cross-play
matrix, tactic metrics of the latest pair), copies it to `<ZAI_DATA_DIR>/leagues/<run_id>/`
for the viewer, and posts `arena` metrics under the run id at the latest snapshot's step.

    python -m arena.league --run-id ID [--once]
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zai_env.config import deep_merge, load_config
from zai_env.vec import OUTCOME_STARVED, OUTCOME_TRUNCATED

from .elo import Matchup, bootstrap, fit
from .matches import Episodes, play_detailed
from .stats import wilson
from .tactics import tactic_metrics

log = logging.getLogger("arena.league")

SCRIPTED_ZOMBIES = "zombie-scripted"
RANDOM = {"human": "human-random", "zombie": "zombie-random"}


@dataclass
class ArenaConfig:
    run_id: str
    league_dir: Path
    env_config: str
    zombie_counts: list[int]
    episodes: int = 128  # per zombie count and matchup
    elo_last: int = 10
    old_steps: int = 40_000_000  # snapshots this much older than the latest count as "old"
    bootstrap_samples: int = 200
    poll_s: float = 60.0
    num_threads: int = 3
    device: str = "cpu"
    seed: int = 4242
    data_dir: Path | None = field(default=None)


def human_score(outcome: np.ndarray) -> np.ndarray:
    return np.select([outcome == OUTCOME_TRUNCATED, outcome == OUTCOME_STARVED], [1.0, 0.5], 0.0)


class Arena:
    def __init__(self, cfg: ArenaConfig, metrics: Any = None):
        self.cfg = cfg
        self.env_cfg = load_config(Path(cfg.env_config), env_overrides=False)
        self.device = torch.device(cfg.device)
        self.metrics = metrics
        self.results: dict[tuple[str, str], dict[str, Any]] = {}
        self.actors: dict[str, Any] = {}
        self.posted_for: tuple[str, str] | None = None
        path = cfg.league_dir / "crossplay.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.results[(r["human"], r["zombie"])] = r

    # --- entries ------------------------------------------------------------------------
    def elo_set(self, pool: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        out = {}
        for side in ("human", "zombie"):
            entries = pool["sides"][side]
            fixed = [e for e in entries if e["source"] != "league"]
            league = sorted(
                (e for e in entries if e["source"] == "league"), key=lambda e: e["steps"]
            )
            out[side] = fixed + league[-self.cfg.elo_last :]
            out[side].append(
                {"id": RANDOM[side], "side": side, "path": None, "source": "random", "steps": 0}
            )
        return out

    def _factory(self, entry: dict[str, Any]):
        from zai_rl.bots import RandomHuman, RandomZombies, ScriptedHuman, ScriptedZombies
        from zai_rl.policy import (
            RecurrentHumanPolicy,
            RecurrentZombiePolicy,
            load_human_actor,
            load_zombie_actor,
        )

        side, dev = entry["side"], self.device
        if entry["source"] == "random":
            if side == "human":
                return lambda c, e, s: RandomHuman(e, seed=s)
            return lambda c, e, n, s: RandomZombies(e, n, seed=s)
        if entry.get("path") is None:
            if side == "human":
                return lambda c, e, s: ScriptedHuman(c, e, seed=s)
            return lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s)
        if entry["id"] not in self.actors:
            load = load_human_actor if side == "human" else load_zombie_actor
            self.actors[entry["id"]] = load(entry["path"], dev)
        actor = self.actors[entry["id"]]
        if side == "human":
            return lambda c, e, s: RecurrentHumanPolicy(actor, e, dev, config=c)
        return lambda c, e, n, s: RecurrentZombiePolicy(actor, e, n, dev, config=c)

    # --- matches ------------------------------------------------------------------------
    def play(
        self, human: dict[str, Any], zombie: dict[str, Any]
    ) -> tuple[dict[str, Any], Episodes]:
        """One matchup over every zombie count, on the arena's fixed seeds."""
        parts = []
        for n in self.cfg.zombie_counts:
            cfg = deep_merge(self.env_cfg, {"zombie_count": n})
            parts.append(
                play_detailed(
                    cfg,
                    self._factory(human),
                    self._factory(zombie),
                    self.cfg.episodes,
                    self.cfg.seed + n,
                    self.cfg.num_threads,
                )
            )
        ep = Episodes(
            outcome=np.concatenate([p.outcome for p in parts]),
            length_s=np.concatenate([p.length_s for p in parts]),
            stats=np.concatenate([p.stats for p in parts]),
        )
        s = ep.summary()
        row = {
            "human": human["id"],
            "zombie": zombie["id"],
            "episodes": s.episodes,
            "score": float(human_score(ep.outcome).mean()),
            "caught": s.caught,
            "starved": s.starved,
            "survived": s.survived,
            "mean_length_s": s.mean_length_s,
            "caught_z": {
                str(n): p.summary().caught
                for n, p in zip(self.cfg.zombie_counts, parts, strict=True)
            },
            "tactics": tactic_metrics(ep.stats, ep.outcome, ep.length_s),
            "played": time.time(),
        }
        return row, ep

    def round(self, pool: dict[str, Any]) -> int:
        """Plays every missing matchup of the Elo set; returns how many were played."""
        sets = self.elo_set(pool)
        played = 0
        path = self.cfg.league_dir / "crossplay.jsonl"
        for h in sets["human"]:
            for z in sets["zombie"]:
                key = (h["id"], z["id"])
                if key in self.results:
                    continue
                t0 = time.time()
                row, _ = self.play(h, z)
                self.results[key] = row
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                played += 1
                log.info(
                    "%s vs %s: human score %.3f (caught %.2f) in %.0f s",
                    *key,
                    row["score"],
                    row["caught"],
                    time.time() - t0,
                )
        return played

    # --- ratings and reports ------------------------------------------------------------
    def report(self, pool: dict[str, Any]) -> dict[str, Any]:
        sets = self.elo_set(pool)
        rows = list(self.results.values())
        matchups = [Matchup(r["human"], r["zombie"], r["episodes"], r["score"]) for r in rows]
        ratings = fit(matchups, SCRIPTED_ZOMBIES)
        ci = bootstrap(matchups, SCRIPTED_ZOMBIES, self.cfg.bootstrap_samples)
        entries = []
        for side in ("human", "zombie"):
            for e in pool["sides"][side] + [{"id": RANDOM[side], "source": "random", "steps": 0}]:
                if e["id"] in ratings:
                    lo, hi = ci.get(e["id"], (ratings[e["id"]],) * 2)
                    entries.append(
                        {
                            "id": e["id"],
                            "side": side,
                            "source": e["source"],
                            "steps": e.get("steps", 0),
                            "elo": ratings[e["id"]],
                            "elo_lo": lo,
                            "elo_hi": hi,
                        }
                    )
        latest = {
            s: max(
                (e for e in sets[s] if e["source"] == "league"),
                key=lambda e: e["steps"],
                default=None,
            )
            for s in ("human", "zombie")
        }
        return {
            "run_id": self.cfg.run_id,
            "updated": time.time(),
            "anchor": SCRIPTED_ZOMBIES,
            "entries": entries,
            "crossplay": [
                {
                    k: r[k]
                    for k in (
                        "human",
                        "zombie",
                        "episodes",
                        "score",
                        "caught",
                        "starved",
                        "survived",
                    )
                }
                for r in rows
            ],
            "latest": {s: (e["id"] if e else None) for s, e in latest.items()},
            "tactics": self._latest_tactics(latest),
        }

    def _latest_tactics(self, latest: dict[str, Any]) -> dict[str, Any]:
        h, z = latest["human"], latest["zombie"]
        out: dict[str, Any] = {}
        if h and z and (h["id"], z["id"]) in self.results:
            out["latest"] = self.results[(h["id"], z["id"])]["tactics"]
        base = self.results.get(("human-scripted", SCRIPTED_ZOMBIES))
        if base:
            out["baseline"] = base["tactics"]
        return out

    def scalars(self, report: dict[str, Any], pool: dict[str, Any]) -> tuple[int, dict[str, float]]:
        """`arena` metrics for the latest snapshot pair, and their step."""
        by_id = {e["id"]: e for e in report["entries"]}
        lh, lz = report["latest"]["human"], report["latest"]["zombie"]
        out: dict[str, float] = {}
        step = 0
        for side, eid in (("human", lh), ("zombie", lz)):
            if eid and eid in by_id:
                e = by_id[eid]
                out[f"elo/{side}"], out[f"elo/{side}_lo"], out[f"elo/{side}_hi"] = (
                    e["elo"],
                    e["elo_lo"],
                    e["elo_hi"],
                )
                step = max(step, int(e["steps"]))
        if "human-scripted" in by_id:
            out["elo/baseline_human"] = by_id["human-scripted"]["elo"]
        res = self.results
        base = res.get(("human-scripted", SCRIPTED_ZOMBIES))
        if lh and (lh, SCRIPTED_ZOMBIES) in res:
            r = res[(lh, SCRIPTED_ZOMBIES)]
            out["anchor/survived"] = r["survived"]
            out["anchor/survived_lo"], out["anchor/survived_hi"] = wilson(
                r["survived"], r["episodes"]
            )
            if base:
                out["anchor/baseline_survived"] = base["survived"]
        if lz and ("human-scripted", lz) in res:
            r = res[("human-scripted", lz)]
            out["anchor/caught"] = r["caught"]
            out["anchor/caught_lo"], out["anchor/caught_hi"] = wilson(r["caught"], r["episodes"])
            for n, v in r["caught_z"].items():
                out[f"anchor/caught_z{n}"] = v
            if base:
                out["anchor/baseline_caught"] = base["caught"]
        if lh and lz:
            # Latest snapshots against the other side's snapshots older than `old_steps`.
            steps = {e["id"]: e["steps"] for s in pool["sides"].values() for e in s}
            for side, me, other_side in (("human", lh, "zombie"), ("zombie", lz, "human")):
                old = [
                    e["id"]
                    for e in pool["sides"][other_side]
                    if e["source"] == "league" and e["steps"] <= steps[me] - self.cfg.old_steps
                ]
                rows = [
                    res[(me, o) if side == "human" else (o, me)]
                    for o in old
                    if ((me, o) if side == "human" else (o, me)) in res
                ]
                if rows:
                    s = np.mean([r["score"] for r in rows])
                    out[f"xplay/{side}_score_vs_old"] = float(s if side == "human" else 1 - s)
        t = report["tactics"]
        for k, v in t.get("latest", {}).items():
            out[f"tactics/{k}"] = v
        for k, v in t.get("baseline", {}).items():
            side, name = k.split("/", 1)  # tactics/<side>/baseline_<name> (charts.ts)
            out[f"tactics/{side}/baseline_{name}"] = v
        return step, out

    def publish(self, pool: dict[str, Any]) -> dict[str, Any]:
        report = self.report(pool)
        for d in filter(None, [self.cfg.league_dir, self.cfg.data_dir]):
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "arena.json.tmp"
            tmp.write_text(json.dumps(report))
            tmp.replace(d / "arena.json")
        pair = (report["latest"]["human"], report["latest"]["zombie"])
        if self.metrics is not None and pair != self.posted_for:
            step, scalars = self.scalars(report, pool)
            self.metrics.log(step, scalars)
            self.posted_for = pair
        return report


def read_pool(league_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((league_dir / "pool.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    from zai_rl.metrics import MetricsLogger
    from zai_rl.train import resolve

    parser = argparse.ArgumentParser(description="League arena: cross-play Elo and tactics.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--league-dir", type=Path)
    parser.add_argument("--env-config", default="configs/env/s0_small.yaml")
    parser.add_argument("--zombie-counts", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="play one round and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(2)
    root = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints"))
    data = os.environ.get("ZAI_DATA_DIR")
    cfg = ArenaConfig(
        run_id=args.run_id,
        league_dir=args.league_dir or root / "league" / args.run_id,
        env_config=str(resolve(args.env_config)),
        zombie_counts=args.zombie_counts,
        episodes=args.episodes,
        poll_s=args.poll_s,
        num_threads=args.num_threads,
        device=args.device,
        data_dir=Path(data) / "leagues" / args.run_id if data else None,
    )
    metrics = MetricsLogger(
        args.run_id,
        "arena",
        os.environ.get("ZAI_SERVER_URL"),
        Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
    )
    arena = Arena(cfg, metrics)
    while True:
        pool = read_pool(cfg.league_dir)
        if pool is None:
            log.info("waiting for %s", cfg.league_dir / "pool.json")
        elif arena.round(pool) or not (cfg.league_dir / "arena.json").is_file():
            report = arena.publish(pool)
            ratings = ", ".join(f"{e['id']} {e['elo']:.0f}" for e in report["entries"])
            log.info("ratings: %s", ratings)
        if args.once:
            break
        time.sleep(cfg.poll_s)
    metrics.close()


if __name__ == "__main__":
    main()
