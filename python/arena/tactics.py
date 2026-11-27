"""Tactic metrics from the per-episode counters (M6.2, docs/ARCHITECTURE.md section 4.8.6).

`tactic_metrics` turns the rows of `VecStep.final_stats` (or `Episodes.stats`) of a set of
finished episodes into the P1 metrics of section 4.8.6, keyed `<side>/<metric>`. Rates are
pooled over episodes (sum of counts over sum of decisions or minutes), so long episodes weigh
more, as they do in play time. Catch geometry is averaged over caught episodes only.
"""

import numpy as np

from zai_env.layout import STATS
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED

ENCIRCLE_GAP_DEG = 120.0  # crates/zai-sim/src/stats.rs


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def tactic_metrics(
    stats: np.ndarray, outcome: np.ndarray, length_s: np.ndarray
) -> dict[str, float]:
    """P1 tactic metrics of `n` finished episodes: `stats [n, STATS_DIM]`, outcome codes `[n]`
    and lengths in seconds `[n]`."""
    g = lambda name: STATS.get(stats, name).astype(np.float64)  # noqa: E731
    out: dict[str, float] = {}
    n = len(outcome)
    if n == 0:
        return out
    decisions = g("decisions").sum()
    minutes = length_s.sum() / 60.0

    for name, code in (
        ("caught", OUTCOME_CAUGHT),
        ("starved", OUTCOME_STARVED),
        ("survived", OUTCOME_TRUNCATED),
    ):
        sel = length_s[outcome == code]
        if len(sel):
            p10, p50, p90 = np.percentile(sel, [10, 50, 90])
            out[f"human/length_{name}_p10_s"] = float(p10)
            out[f"human/length_{name}_median_s"] = float(p50)
            out[f"human/length_{name}_p90_s"] = float(p90)
    p10, p50, p90 = np.percentile(length_s, [10, 50, 90])
    out.update({"human/length_p10_s": p10, "human/length_median_s": p50, "human/length_p90_s": p90})

    out["human/pickups_per_episode"] = g("pickups").sum() / n
    out["human/eaten_per_episode"] = g("eaten").sum() / n
    out["human/mean_carried"] = _ratio(g("carried_sum").sum(), decisions)
    out["human/heavy_fraction"] = _ratio(g("heavy_decisions").sum(), decisions)
    out["human/walk_noises_per_min"] = _ratio(g("walk_noises").sum(), minutes)
    out["human/pickups_per_min"] = _ratio(g("pickups").sum(), minutes)
    out["human/door_closes_per_min"] = _ratio(g("door_closes").sum(), minutes)
    out["human/risky_pickups_per_episode"] = g("risky_pickups").sum() / n
    out["human/risky_pickup_fraction"] = _ratio(g("risky_pickups").sum(), g("pickups").sum())
    out["human/door_closes_per_episode"] = g("door_closes").sum() / n
    out["human/defensive_closes_per_episode"] = g("defensive_closes").sum() / n
    out["human/inside_fraction"] = _ratio(g("inside_decisions").sum(), decisions)

    caught = outcome == OUTCOME_CAUGHT
    if caught.any():
        gap = g("catch_gap_deg")[caught]
        out["zombie/catch_gap_deg"] = float(gap.mean())
        out["zombie/encircled_fraction"] = float((gap < ENCIRCLE_GAP_DEG).mean())
        out["zombie/catch_sectors"] = float(g("catch_sectors")[caught].mean())
        out["zombie/catch_close_zombies"] = float(g("catch_close_zombies")[caught].mean())
        out["zombie/time_to_catch_s"] = float(length_s[caught].mean())
    out["zombie/quiet_dispersion_m"] = _ratio(
        g("quiet_dispersion_sum").sum(), g("quiet_decisions").sum()
    )
    return {k: float(v) for k, v in out.items()}
