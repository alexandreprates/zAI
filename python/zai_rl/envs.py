"""Training environments: several `ZaiVecEnv` groups (one per zombie count) seen by the human
learner as one batch, with the zombies driven by a fixed opponent policy."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from zai_env.config import deep_merge
from zai_env.vec import ZaiVecEnv

ZombieFactory = Callable[[dict[str, Any], int, int, int], Any]


@dataclass
class HumanBatch:
    obs: np.ndarray  # [E, HUMAN_OBS_DIM]
    mask: np.ndarray  # [E, 20] bool
    state: np.ndarray  # [E, STATE_DIM]


@dataclass
class HumanStep:
    next: HumanBatch
    reward: np.ndarray  # [E]
    terminated: np.ndarray  # [E] bool
    truncated: np.ndarray  # [E] bool
    outcome: np.ndarray  # [E] uint8
    episode_length: np.ndarray  # [E]
    final_state: np.ndarray  # [E, STATE_DIM], rows valid where truncated


class HumanTrainingEnv:
    """Concatenates one vec env per zombie count; zombies act with `zombie_factory` policies.

    The numbers of envs per group are as equal as possible and sum to `num_envs`.
    """

    def __init__(
        self,
        config: dict[str, Any],
        zombie_counts: list[int],
        num_envs: int,
        zombie_factory: ZombieFactory,
        seed: int = 0,
        num_threads: int | None = None,
    ):
        if num_envs < len(zombie_counts):
            raise ValueError("num_envs must be at least the number of zombie groups")
        sizes = [num_envs // len(zombie_counts)] * len(zombie_counts)
        for i in range(num_envs - sum(sizes)):
            sizes[i] += 1
        self.groups = []
        for i, (n, size) in enumerate(zip(zombie_counts, sizes, strict=True)):
            cfg = deep_merge(config, {"zombie_count": n})
            venv = ZaiVecEnv(cfg, num_envs=size, seed=seed + 1000 * i, num_threads=num_threads)
            self.groups.append((venv, zombie_factory(cfg, size, n, seed + 1000 * i + 1)))
        self.num_envs = num_envs
        self._zombie_obs: list[np.ndarray] = []

    def reset(self) -> HumanBatch:
        obs = [venv.reset() for venv, _ in self.groups]
        self._zombie_obs = [o.zombie_obs for o in obs]
        return HumanBatch(
            obs=np.concatenate([o.human_obs for o in obs]),
            mask=np.concatenate([o.human_mask for o in obs]),
            state=np.concatenate([o.state for o in obs]),
        )

    def step(self, human_actions: np.ndarray) -> HumanStep:
        parts = []
        start = 0
        for (venv, zombies), zobs in zip(self.groups, self._zombie_obs, strict=True):
            end = start + venv.num_envs
            parts.append(venv.step(human_actions[start:end], zombies.act(zobs)))
            start = end
        self._zombie_obs = [p.obs.zombie_obs for p in parts]

        def cat(f):
            return np.concatenate([f(p) for p in parts])

        return HumanStep(
            next=HumanBatch(
                obs=cat(lambda p: p.obs.human_obs),
                mask=cat(lambda p: p.obs.human_mask),
                state=cat(lambda p: p.obs.state),
            ),
            reward=cat(lambda p: p.human_reward),
            terminated=cat(lambda p: p.terminated),
            truncated=cat(lambda p: p.truncated),
            outcome=cat(lambda p: p.outcome),
            episode_length=cat(lambda p: p.episode_length),
            final_state=cat(lambda p: p.final_state),
        )
