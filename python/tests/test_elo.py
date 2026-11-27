import numpy as np

from arena.elo import Matchup, bootstrap, fit


def synthetic(true: dict[str, float], humans, zombies, episodes, rng):
    out = []
    for h in humans:
        for z in zombies:
            p = 1 / (1 + 10 ** ((true[z] - true[h]) / 400))
            out.append(Matchup(h, z, episodes, rng.binomial(episodes, p) / episodes))
    return out


def test_fit_recovers_bradley_terry_ratings_within_thirty_points():
    rng = np.random.default_rng(0)
    true = {"h1": 900, "h2": 1100, "h3": 1250, "zs": 1000, "z1": 1150, "z2": 800}
    ms = synthetic(true, ["h1", "h2", "h3"], ["zs", "z1", "z2"], 2000, rng)
    r = fit(ms, anchor="zs", prior_sigma=4000)
    assert r["zs"] == 1000.0
    for k, v in true.items():
        assert abs(r[k] - v) < 30, (k, r[k], v)


def test_bootstrap_intervals_cover_the_fit_and_widen_with_fewer_games():
    rng = np.random.default_rng(1)
    true = {"h1": 950, "zs": 1000, "z1": 1100}
    big = synthetic(true, ["h1"], ["zs", "z1"], 2000, rng)
    small = [Matchup(m.human, m.zombie, 50, m.score) for m in big]
    ci_big = bootstrap(big, "zs", samples=100)
    ci_small = bootstrap(small, "zs", samples=100)
    r = fit(big, "zs")
    assert ci_big["h1"][0] <= r["h1"] <= ci_big["h1"][1]
    width = lambda ci: ci["h1"][1] - ci["h1"][0]  # noqa: E731
    assert width(ci_small) > 2 * width(ci_big)


def test_one_sided_results_stay_finite_with_the_prior():
    r = fit([Matchup("h", "zs", 100, 1.0)], anchor="zs")
    assert 1000 < r["h"] < 3000
