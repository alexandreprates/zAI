import pytest
import yaml

from zai_env.config import config_dir, default_config, load_config


def test_default_yaml_matches_rust_defaults():
    with open(config_dir() / "default.yaml") as f:
        assert load_config(config_dir() / "default.yaml", env_overrides=False) == default_config()
        assert yaml.safe_load(f).keys() == default_config().keys()


def test_overrides_are_merged_recursively():
    cfg = load_config(overrides={"human": {"vision_radius_m": 50.0}}, env_overrides=False)
    assert cfg["human"]["vision_radius_m"] == 50.0
    assert cfg["human"]["hunger_interval_s"] == 90.0


def test_environment_variables_override_counts(monkeypatch):
    monkeypatch.setenv("ZAI_ZOMBIE_COUNT", "7")
    monkeypatch.setenv("ZAI_FOOD_COUNT", "99")
    cfg = load_config()
    assert (cfg["zombie_count"], cfg["food_count"]) == (7, 99)
    assert load_config(env_overrides=False)["zombie_count"] == 20


@pytest.mark.parametrize(
    "override",
    [{"zombie_count": 0}, {"zombie_count": 65}, {"unknown_key": 1}, {"human": {"typo": 1}}],
)
def test_invalid_configs_are_rejected(override):
    with pytest.raises(ValueError):
        load_config(overrides=override, env_overrides=False)
