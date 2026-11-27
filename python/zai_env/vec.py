"""Typed wrapper over the batched Rust environment (`zai_sim.VecEnv`)."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import zai_sim

from .config import load_config

OUTCOMES = ("running", "caught", "starved", "truncated")
OUTCOME_CAUGHT = 1
OUTCOME_STARVED = 2
OUTCOME_TRUNCATED = 3


@dataclass(frozen=True)
class VecObs:
    human_obs: np.ndarray  # float32 [E, HUMAN_OBS_DIM]
    human_mask: np.ndarray  # bool [E, MOVE_ACTIONS + INTERACT_ACTIONS]
    zombie_obs: np.ndarray  # float32 [E, N, ZOMBIE_OBS_DIM]
    zombie_valid: np.ndarray  # bool [E, N]
    state: np.ndarray  # float32 [E, STATE_DIM]


@dataclass(frozen=True)
class VecStep:
    obs: VecObs
    human_reward: np.ndarray  # float32 [E]
    zombie_reward: np.ndarray  # float32 [E, N]: team reward plus per-zombie distance shaping
    terminated: np.ndarray  # bool [E]
    truncated: np.ndarray  # bool [E]
    outcome: np.ndarray  # uint8 [E], index into OUTCOMES
    episode_length: np.ndarray  # uint32 [E], decisions in the (finished) episode
    final_human_obs: np.ndarray  # rows valid where terminated | truncated
    final_zombie_obs: np.ndarray
    final_state: np.ndarray
    final_stats: np.ndarray  # float32 [E, STATS_DIM]: tactic counters (zai_env.layout.STATS)


class ZaiVecEnv:
    """`num_envs` matches stepped in parallel with automatic reset.

    The arrays returned by `reset()` / `step()` are reused and overwritten by the next call;
    copy anything that must outlive it.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        num_envs: int = 64,
        seed: int = 0,
        num_threads: int | None = None,
    ):
        self.config = config if config is not None else load_config()
        self._env = zai_sim.VecEnv(json.dumps(self.config), num_envs, seed, num_threads)
        self.num_envs = num_envs
        self.num_zombies = self._env.num_zombies

    def reset(self, seed: int | None = None) -> VecObs:
        return VecObs(**self._env.reset(seed))

    def step(self, human_actions: np.ndarray, zombie_actions: np.ndarray) -> VecStep:
        obs, hr, zr, term, trunc, info = self._env.step(
            np.ascontiguousarray(human_actions, dtype=np.int64),
            np.ascontiguousarray(zombie_actions, dtype=np.int64),
        )
        return VecStep(
            obs=VecObs(**obs),
            human_reward=hr,
            zombie_reward=zr,
            terminated=term,
            truncated=trunc,
            **info,
        )

    def current_seeds(self) -> list[int]:
        return self._env.current_seeds()

    def state_hashes(self) -> list[int]:
        return self._env.state_hashes()
