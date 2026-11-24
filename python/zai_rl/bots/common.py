"""Helpers shared by the scripted and random policies."""

import numpy as np
import zai_sim

STAY = zai_sim.MOVE_ACTIONS - 1
_SECTOR = np.pi / 8  # 16 directions, 22.5 degrees apart


def direction_to_action(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Nearest of the 16 movement actions for each direction; `STAY` for zero vectors."""
    angle = np.arctan2(dy, dx)
    action = np.rint(angle / _SECTOR).astype(np.int64) % 16
    return np.where((dx == 0) & (dy == 0), STAY, action)


class RandomHuman:
    """Uniform random movement; interaction sampled uniformly among valid ones."""

    def __init__(self, num_envs: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_envs = num_envs

    def act(self, human_obs: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
        move = self.rng.integers(0, zai_sim.MOVE_ACTIONS, self.num_envs)
        interact_mask = human_mask[:, zai_sim.MOVE_ACTIONS :]
        scores = self.rng.random(interact_mask.shape) * interact_mask
        return np.stack([move, scores.argmax(axis=1)], axis=1).astype(np.int64)


class RandomZombies:
    """Uniform random movement for every zombie."""

    def __init__(self, num_envs: int, num_zombies: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.shape = (num_envs, num_zombies)

    def act(self, zombie_obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, zai_sim.MOVE_ACTIONS, self.shape, dtype=np.int64)
