import numpy as np

from arena.matches import play, play_detailed
from arena.tactics import tactic_metrics
from zai_env.layout import STATS
from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_TRUNCATED
from zai_rl.bots import ScriptedHuman, ScriptedZombies


def test_metrics_pool_counts_and_average_catch_geometry_over_catches():
    stats = np.zeros((2, len(STATS.names)), dtype=np.float32)
    row = dict(
        decisions=120,
        pickups=3,
        risky_pickups=1,
        carried_sum=60,
        walk_noises=60,
        catch_gap_deg=90,
        catch_sectors=3,
    )
    for k, v in row.items():
        stats[0, STATS.index[k]] = v
    stats[1, STATS.index["decisions"]] = 120
    stats[1, STATS.index["catch_gap_deg"]] = 359  # not caught: ignored
    m = tactic_metrics(stats, np.array([OUTCOME_CAUGHT, OUTCOME_TRUNCATED]), np.array([60.0, 60.0]))
    assert m["human/pickups_per_episode"] == 1.5
    assert m["human/mean_carried"] == 0.25
    assert m["human/walk_noises_per_min"] == 30.0
    assert m["human/risky_pickup_fraction"] == 1 / 3
    assert m["zombie/catch_gap_deg"] == 90.0 and m["zombie/encircled_fraction"] == 1.0


def test_detailed_play_returns_the_counters_of_each_episode(small_config):
    h = lambda c, e, s: ScriptedHuman(c, e, seed=s)  # noqa: E731
    z = lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s)  # noqa: E731
    ep = play_detailed(small_config, h, z, episodes=8, seed=3)
    decision_s = small_config["dt_s"] * small_config["action_repeat"]
    np.testing.assert_allclose(STATS.get(ep.stats, "decisions") * decision_s, ep.length_s)
    assert ep.summary() == play(small_config, h, z, episodes=8, seed=3)
    m = tactic_metrics(ep.stats, ep.outcome, ep.length_s)
    assert m["human/walk_noises_per_min"] > 0
