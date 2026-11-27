"""Curriculum gates of the league (M6.5, docs/ARCHITECTURE.md section 4.8.1).

A gate is a set of conditions checked after a stage's minimum budget; it passes when every
condition holds at `consecutive` evaluations in a row. Each condition reports a margin (>= 0
passes) so the curves show how far a run is from advancing; a condition whose input is missing
(no arena rating or probe yet) fails with margin NaN. The thresholds are relative to the scripted
baselines measured at the start of each stage, except where the stage sets absolute minimums.
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateSpec:
    survived_min: float = 0.0  # human anchor: latest human vs scripted zombies
    survived_over_baseline: float = 0.0  # ... at least baseline + this
    caught_min: float = 0.0  # zombie anchor: latest zombies vs scripted human
    caught_times_baseline: float = 0.0  # ... at least this times the baseline
    caught_over_baseline: float = 0.0  # ... at least baseline + this
    caught_ci_above_baseline: bool = False  # Wilson lower bound above the baseline's upper
    balance: tuple[float, float] = (0.2, 0.8)  # group A zombie score s_z
    forgetting_min: float = 0.55  # each side's score against this stage's old snapshots
    elo_gain: float | None = 100.0  # latest over the stage starter, intervals apart (None: off)
    probe_gap_max: float | None = 0.2  # exploitability probe, both sides (None: off)
    consecutive: int = 2


@dataclass
class GateInputs:
    survived: float
    caught: float
    caught_lo: float
    baseline_survived: float
    baseline_caught: float
    baseline_caught_hi: float
    s_z: float
    forgetting: dict[str, float] = field(default_factory=dict)  # side -> score (NaN: none)
    elo: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # side -> latest
    elo_starter: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    probe_gap: dict[str, float] = field(default_factory=dict)


def margins(spec: GateSpec, x: GateInputs) -> dict[str, float]:
    """Margin of every condition (> 0 passes, NaN: input missing)."""
    out: dict[str, float] = {}
    need_s = max(spec.survived_min, x.baseline_survived + spec.survived_over_baseline)
    out["survived"] = x.survived - need_s
    need_c = max(
        spec.caught_min,
        spec.caught_times_baseline * x.baseline_caught,
        x.baseline_caught + spec.caught_over_baseline,
    )
    out["caught"] = x.caught - need_c
    if spec.caught_ci_above_baseline:
        out["caught_ci"] = x.caught_lo - x.baseline_caught_hi
    lo, hi = spec.balance
    out["balance"] = min(x.s_z - lo, hi - x.s_z) if math.isfinite(x.s_z) else math.nan
    for side in ("human", "zombie"):
        f = x.forgetting.get(side, math.nan)
        # No old snapshots yet: nothing to forget.
        out[f"forgetting_{side}"] = f - spec.forgetting_min if math.isfinite(f) else 1.0
        if spec.elo_gain is not None:
            if side in x.elo and side in x.elo_starter:
                (e, lo_e, _), (s, _, hi_s) = x.elo[side], x.elo_starter[side]
                out[f"elo_{side}"] = min(e - s - spec.elo_gain, lo_e - hi_s)
            else:
                out[f"elo_{side}"] = math.nan
        if spec.probe_gap_max is not None:
            g = x.probe_gap.get(side, math.nan)
            out[f"probe_{side}"] = spec.probe_gap_max - g if math.isfinite(g) else math.nan
    return out


def passes(m: dict[str, float]) -> bool:
    return all(math.isfinite(v) and v >= 0 for v in m.values())


class GateTracker:
    """Counts consecutive passing evaluations of one stage's gate."""

    def __init__(self, spec: GateSpec):
        self.spec = spec
        self.streak = 0
        self.last: dict[str, float] = {}

    def update(self, x: GateInputs) -> bool:
        self.last = margins(self.spec, x)
        self.streak = self.streak + 1 if passes(self.last) else 0
        return self.streak >= self.spec.consecutive

    def failing(self) -> list[str]:
        return [k for k, v in self.last.items() if not (math.isfinite(v) and v >= 0)]

    def scalars(self) -> dict[str, Any]:
        out = {f"gate/{k}": v for k, v in self.last.items()}
        out["gate/passed"] = float(self.streak >= self.spec.consecutive)
        out["gate/streak"] = float(self.streak)
        return out
