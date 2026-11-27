"""Elo ratings from a bipartite cross-play table (docs/ARCHITECTURE.md section 4.8.6).

Humans only play zombies, so ratings are fitted jointly by Bradley-Terry maximum likelihood on
episode scores (human win 1, starvation 0.5, loss 0) on the Elo scale: the expected human
score against a zombie team is `1 / (1 + 10^((r_z - r_h) / 400))`. The bipartite table fixes
ratings only up to one shared offset, so a single anchor is pinned (the scripted zombies at
1000), and a weak Gaussian prior (sigma 400, centered on the anchor rating) keeps entries with
few or one-sided games finite. Intervals come from bootstrap resamples of each matchup's
episodes.
"""

from dataclasses import dataclass

import numpy as np

SCALE = 400.0 / np.log(10.0)  # Elo points per natural-log odds unit


@dataclass(frozen=True)
class Matchup:
    human: str
    zombie: str
    episodes: int
    score: float  # mean human score over the episodes


def fit(
    matchups: list[Matchup],
    anchor: str,
    anchor_rating: float = 1000.0,
    prior_sigma: float = 400.0,
    iters: int = 200,
) -> dict[str, float]:
    """Ratings of every human and zombie entry in `matchups` (ids must be unique across sides)."""
    ids = sorted({m.human for m in matchups} | {m.zombie for m in matchups} | {anchor})
    index = {k: i for i, k in enumerate(ids)}
    h = np.array([index[m.human] for m in matchups])
    z = np.array([index[m.zombie] for m in matchups])
    n = np.array([m.episodes for m in matchups], dtype=np.float64)
    w = np.array([m.score for m in matchups], dtype=np.float64) * n
    a = index[anchor]
    r = np.zeros(len(ids))  # log-odds units, relative to the anchor
    prior = (SCALE / prior_sigma) ** 2  # precision in log-odds units
    for _ in range(iters):  # Newton steps on the penalized log-likelihood
        p = 1.0 / (1.0 + np.exp(-(r[h] - r[z])))
        resid = w - n * p
        grad = np.zeros(len(ids))
        np.add.at(grad, h, resid)
        np.add.at(grad, z, -resid)
        grad -= prior * r
        hw = n * p * (1 - p)
        hess = np.zeros((len(ids), len(ids)))
        np.add.at(hess, (h, h), -hw)
        np.add.at(hess, (z, z), -hw)
        np.add.at(hess, (h, z), hw)
        np.add.at(hess, (z, h), hw)
        hess -= prior * np.eye(len(ids))
        keep = np.arange(len(ids)) != a  # the anchor stays at 0
        step = np.zeros(len(ids))
        step[keep] = np.linalg.solve(hess[np.ix_(keep, keep)], -grad[keep])
        r += step
        if np.abs(step).max() < 1e-9:
            break
    return {k: float(anchor_rating + SCALE * r[index[k]]) for k in ids}


def bootstrap(
    matchups: list[Matchup],
    anchor: str,
    samples: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of the ratings over resamples of each matchup's episodes (scores are
    resampled as a binomial over win/loss with draws split, a close approximation)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(samples):
        res = [
            Matchup(
                m.human,
                m.zombie,
                m.episodes,
                rng.binomial(m.episodes, min(max(m.score, 0.0), 1.0)) / m.episodes,
            )
            for m in matchups
        ]
        for k, v in fit(res, anchor).items():
            draws.setdefault(k, []).append(v)
    q = (1 - level) / 2
    return {k: (float(np.quantile(v, q)), float(np.quantile(v, 1 - q))) for k, v in draws.items()}
