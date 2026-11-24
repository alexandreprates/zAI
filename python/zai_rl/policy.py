"""Trained policies wrapped with the bot interface (`act` + `reset`) for evaluation."""

from pathlib import Path

import numpy as np
import torch

from .models import HumanActor, ZombieActor


class RecurrentHumanPolicy:
    """Batched human policy keeping one GRU state per environment."""

    def __init__(
        self,
        actor: HumanActor,
        num_envs: int,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.h = actor.initial_state(num_envs, self.device)

    @classmethod
    def from_checkpoint(
        cls, path: Path, num_envs: int, device: str = "cpu", deterministic: bool = False
    ) -> "RecurrentHumanPolicy":
        sd = torch.load(path, map_location=device, weights_only=False)
        actor = HumanActor(hidden=sd["ppo_config"]["hidden"]).to(device)
        actor.load_state_dict(sd["actor"])
        actor.eval()
        return cls(actor, num_envs, device, deterministic)

    def reset(self, ended: np.ndarray) -> None:
        """Clears the memory of environments whose episode just ended."""
        if ended.any():
            self.h[torch.as_tensor(ended, device=self.device)] = 0.0

    @torch.no_grad()
    def act(self, human_obs: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(human_obs, device=self.device)
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
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.shape = (num_envs, num_zombies)
        self.h = actor.initial_state(num_envs * num_zombies, self.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        num_envs: int,
        num_zombies: int,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> "RecurrentZombiePolicy":
        sd = torch.load(path, map_location=device, weights_only=False)
        actor = ZombieActor(hidden=sd["mappo_config"]["hidden"]).to(device)
        actor.load_state_dict(sd["actor"])
        actor.eval()
        return cls(actor, num_envs, num_zombies, device, deterministic)

    def reset(self, ended: np.ndarray) -> None:
        """Clears the memory of every zombie of the environments whose episode just ended."""
        if ended.any():
            rows = torch.as_tensor(np.repeat(ended, self.shape[1]), device=self.device)
            self.h[rows] = 0.0

    @torch.no_grad()
    def act(self, zombie_obs: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(zombie_obs, device=self.device).reshape(-1, zombie_obs.shape[-1])
        dist, self.h = self.actor.step(obs, self.h)
        a = dist.probs.argmax(-1) if self.deterministic else dist.sample()
        return a.reshape(self.shape).cpu().numpy().astype(np.int64)
