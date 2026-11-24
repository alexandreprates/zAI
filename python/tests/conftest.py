import pytest

from zai_env.config import load_config


@pytest.fixture
def small_config():
    """S0-like configuration: small map, few zombies, short episodes so resets happen."""
    return load_config(
        overrides={
            "map_size_m": 150.0,
            "house_count": 3,
            "zombie_count": 4,
            "food_count": 20,
            "episode_cap_s": 30.0,
            "min_spawn_distance_m": 20.0,
        },
        env_overrides=False,
    )
