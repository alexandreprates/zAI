"""Environment configuration loading.

The schema lives in Rust (`zai_sim::Config`); Python only merges YAML files and overrides
and asks the extension to validate the result, so there is a single source of truth.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml
import zai_sim


def config_dir() -> Path:
    """Directory with the environment YAML files (`ZAI_CONFIG_DIR` overrides it)."""
    if env := os.environ.get("ZAI_CONFIG_DIR"):
        return Path(env)
    return Path(__file__).resolve().parents[2] / "configs" / "env"


def default_config() -> dict[str, Any]:
    """PRD defaults, as defined by the Rust `Config::default()`."""
    return json.loads(zai_sim.default_config())


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Returns `base` updated recursively with `override` (inputs are not modified)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env_overrides: bool = True,
) -> dict[str, Any]:
    """Builds a validated config: defaults <- YAML file <- `overrides` <- environment variables.

    Raises `ValueError` if the result is invalid or contains unknown keys.
    """
    cfg = default_config()
    if path is not None:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    raw = json.dumps(cfg)
    raw = zai_sim.apply_env_overrides(raw) if env_overrides else zai_sim.validate_config(raw)
    return json.loads(raw)
