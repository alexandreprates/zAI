"""Policies a remote side can play, with a batch of one environment (see `client.Client.play`).

A spec is `scripted`, `random` or a checkpoint path (`.pt`, any layout: v1 checkpoints play
through `zai_env.legacy` with the episode's config).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zai_rl.bots import RandomHuman, RandomZombies, ScriptedHuman, ScriptedZombies
from zai_rl.policy import RecurrentHumanPolicy, RecurrentZombiePolicy

PolicyFactory = Callable[[dict[str, Any], int], Any]


def human_policy(
    spec: str, seed: int = 0, deterministic: bool = False, device: str = "cpu"
) -> PolicyFactory:
    """Factory `(config, num_agents) -> policy` of the human side."""

    def make(config: dict[str, Any], _n: int) -> Any:
        if spec == "scripted":
            return ScriptedHuman(config, 1, seed=seed)
        if spec == "random":
            return RandomHuman(1, seed=seed)
        return RecurrentHumanPolicy.from_checkpoint(
            Path(spec), 1, device, deterministic, config=config
        )

    return make


def zombie_policy(
    spec: str, seed: int = 0, deterministic: bool = False, device: str = "cpu"
) -> PolicyFactory:
    """Factory `(config, num_zombies) -> policy` of the zombie side."""

    def make(config: dict[str, Any], n: int) -> Any:
        if spec == "scripted":
            return ScriptedZombies(config, 1, n, seed=seed)
        if spec == "random":
            return RandomZombies(1, n, seed=seed)
        return RecurrentZombiePolicy.from_checkpoint(
            Path(spec), 1, n, device, deterministic, config=config
        )

    return make
