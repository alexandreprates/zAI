"""Observation parity of the gRPC path with training (docs/ARCHITECTURE.md 4.9.5, step 2);
the checks live in `zai_agent.parity`, shared with `make grpc-smoke`."""

import pytest

from zai_agent.parity import play_and_verify
from zai_env.config import config_dir, load_config

CONFIGS = {
    "s0_small": load_config(config_dir() / "s0_small.yaml", env_overrides=False),
    "s0": load_config(config_dir() / "s0.yaml", env_overrides=False),
    # The PRD defaults (500 m, 20 zombies) with a short cap.
    "default": load_config(overrides={"episode_cap_s": 40.0}, env_overrides=False),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("deterministic", [False, True])
def test_grpc_observations_equal_the_training_env(zai_server, name, deterministic):
    target, http = zai_server
    assert play_and_verify(target, CONFIGS[name], deterministic, http=http) > 0
