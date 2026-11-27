"""Trained policies wrapped with the bot interface (`act` + `reset`) for evaluation.

Checkpoints trained on observation layout v1 (`obs_version` missing or 1) play in the v2 env
through `zai_env.legacy`: give the env `config` so their observations can be rebuilt.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zai_env.layout import OBS_VERSION
from zai_env.legacy import adapt_v1_human, adapt_v1_zombie

from .algos.common import checkpoint_obs_version
from .models import HumanActor, ZombieActor


def _adapter(
    actor: HumanActor | ZombieActor,
    config: dict[str, Any] | None,
    adapt: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    if actor.obs_version == OBS_VERSION:
        return lambda obs: obs
    if config is None:
        raise ValueError(
            f"a layout v{actor.obs_version} policy needs the env config to adapt observations"
        )
    return lambda obs: adapt(obs, config)


def load_human_actor(path: Path, device: torch.device | str = "cpu") -> HumanActor:
    """Frozen human actor of a checkpoint, on the checkpoint's observation layout."""
    sd = torch.load(path, map_location=device, weights_only=False)
    actor = HumanActor(hidden=sd["ppo_config"]["hidden"], obs_version=checkpoint_obs_version(sd))
    actor.load_state_dict(sd["actor"])
    return actor.to(device).eval()


class RecurrentHumanPolicy:
    """Batched human policy keeping one GRU state per environment."""

    def __init__(
        self,
        actor: HumanActor,
        num_envs: int,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        config: dict[str, Any] | None = None,
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.h = actor.initial_state(num_envs, self.device)
        self.adapt = _adapter(actor, config, adapt_v1_human)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        num_envs: int,
        device: str = "cpu",
        deterministic: bool = False,
        config: dict[str, Any] | None = None,
    ) -> "RecurrentHumanPolicy":
        return cls(load_human_actor(path, device), num_envs, device, deterministic, config)

    def reset(self, ended: np.ndarray) -> None:
        """Clears the memory of environments whose episode just ended."""
        if ended.any():
            self.h[torch.as_tensor(ended, device=self.device)] = 0.0

    @torch.no_grad()
    def act(self, human_obs: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(self.adapt(human_obs), device=self.device)
        mask = torch.as_tensor(human_mask, device=self.device)
        move, interact, self.h = self.actor.step(obs, self.h, mask)
        if self.deterministic:
            a = torch.stack([move.probs.argmax(-1), interact.probs.argmax(-1)], dim=1)
        else:
            a = torch.stack([move.sample(), interact.sample()], dim=1)
        return a.cpu().numpy().astype(np.int64)


class RecurrentZombiePolicy:
    """Batched zombie team policy: one GRU state per zombie of each environment."""

    def __init__(
        self,
        actor: ZombieActor,
        num_envs: int,
        num_zombies: int,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        config: dict[str, Any] | None = None,
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.shape = (num_envs, num_zombies)
        self.h = actor.initial_state(num_envs * num_zombies, self.device)
        self.adapt = _adapter(actor, config, adapt_v1_zombie)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        num_envs: int,
        num_zombies: int,
        device: str = "cpu",
        deterministic: bool = False,
        config: dict[str, Any] | None = None,
    ) -> "RecurrentZombiePolicy":
        sd = torch.load(path, map_location=device, weights_only=False)
        actor = ZombieActor(
            hidden=sd["mappo_config"]["hidden"], obs_version=checkpoint_obs_version(sd)
        ).to(device)
        actor.load_state_dict(sd["actor"])
        actor.eval()
        return cls(actor, num_envs, num_zombies, device, deterministic, config)

    def reset(self, ended: np.ndarray) -> None:
        """Clears the memory of every zombie of the environments whose episode just ended."""
        if ended.any():
            rows = torch.as_tensor(np.repeat(ended, self.shape[1]), device=self.device)
            self.h[rows] = 0.0

    @torch.no_grad()
    def act(self, zombie_obs: np.ndarray) -> np.ndarray:
        zombie_obs = self.adapt(zombie_obs)
        obs = torch.as_tensor(zombie_obs, device=self.device).reshape(-1, zombie_obs.shape[-1])
        dist, self.h = self.actor.step(obs, self.h)
        a = dist.probs.argmax(-1) if self.deterministic else dist.sample()
        return a.reshape(self.shape).cpu().numpy().astype(np.int64)
