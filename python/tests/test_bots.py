import numpy as np

from arena.matches import play
from zai_env.config import load_config
from zai_rl.bots import (
    RandomHuman,
    RandomZombies,
    ScriptedHuman,
    ScriptedZombies,
    direction_to_action,
)


def s0_config():
    return load_config(
        overrides={
            "map_size_m": 150.0,
            "house_count": 3,
            "zombie_count": 4,
            "food_count": 20,
            "episode_cap_s": 300.0,
        },
        env_overrides=False,
    )


def scripted_human(c, e, s):
    return ScriptedHuman(c, e, seed=s)


def random_human(c, e, s):
    return RandomHuman(e, seed=s)


def scripted_zombies(c, e, n, s):
    return ScriptedZombies(c, e, n, seed=s)


def random_zombies(c, e, n, s):
    return RandomZombies(e, n, seed=s)


def test_direction_to_action():
    dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
    dy = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(direction_to_action(dx, dy), [0, 4, 8, 12, 16, 2])


def test_scripted_human_outlives_random_human():
    cfg = s0_config()
    scripted = play(cfg, scripted_human, scripted_zombies, episodes=48, seed=1)
    random = play(cfg, random_human, scripted_zombies, episodes=48, seed=1)
    assert scripted.mean_length_s > random.mean_length_s
    assert scripted.starved < random.starved


def test_scripted_zombies_catch_more_than_random_zombies():
    # Scripted zombies rarely run down a careful human (see the balance sweep), so this
    # compares the hunters against a noisy random prey spawned nearby.
    cfg = load_config(
        overrides={
            "map_size_m": 150.0,
            "house_count": 3,
            "zombie_count": 20,
            "food_count": 20,
            "episode_cap_s": 85.0,
            "min_spawn_distance_m": 10.0,
        },
        env_overrides=False,
    )
    scripted = play(cfg, random_human, scripted_zombies, episodes=64, seed=1)
    random = play(cfg, random_human, random_zombies, episodes=64, seed=1)
    assert scripted.caught > 2 * random.caught
    assert scripted.caught > 0.05
