"""Training environments: several `ZaiVecEnv` groups (one per zombie count) seen by one learner
as a single batch, with the other side driven by a fixed opponent policy.

- `HumanTrainingEnv`: the learner is the human (`[E]` rows), zombies are the opponent.
- `ZombieTrainingEnv`: the learner is the zombie team. Every group has a fixed zombie count, so
  the zombies of all groups form a flat agent axis `[A]` (`A = sum_g E_g * N_g`) with a static
  `agent_env [A]` index into the env axis `[E]`; agent `off_g + e * N_g + n` is zombie `n` of
  env `e` of group `g`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from zai_env.config import deep_merge
from zai_env.vec import ZaiVecEnv

from .profiling import timer

ZombieFactory = Callable[[dict[str, Any], int, int, int], Any]
HumanFactory = Callable[[dict[str, Any], int, int], Any]


def per_group(factory: Any, groups: int) -> list[Any]:
    """One opponent factory per group: a list is used as is, a single factory is shared."""
    if isinstance(factory, list):
        if len(factory) != groups:
            raise ValueError("one opponent factory per zombie-count group")
        return factory
    return [factory] * groups


def group_sizes(num_envs: int, groups: int) -> list[int]:
    """Envs per group: as equal as possible, summing to `num_envs`."""
    if num_envs < groups:
        raise ValueError("num_envs must be at least the number of zombie groups")
    sizes = [num_envs // groups] * groups
    for i in range(num_envs - sum(sizes)):
        sizes[i] += 1
    return sizes


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
        zombie_factory: ZombieFactory | list[ZombieFactory],
        seed: int = 0,
        num_threads: int | None = None,
    ):
        sizes = group_sizes(num_envs, len(zombie_counts))
        factories = per_group(zombie_factory, len(zombie_counts))
        self.groups = []
        self.group_slices = []
        start = 0
        for i, (n, size) in enumerate(zip(zombie_counts, sizes, strict=True)):
            cfg = deep_merge(config, {"zombie_count": n})
            venv = ZaiVecEnv(cfg, num_envs=size, seed=seed + 1000 * i, num_threads=num_threads)
            self.groups.append((venv, factories[i](cfg, size, n, seed + 1000 * i + 1)))
            self.group_slices.append(slice(start, start + size))
            start += size
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
            with timer("opponent"):
                zombie_actions = zombies.act(zobs)
            with timer("sim"):
                step = venv.step(human_actions[start:end], zombie_actions)
            # Recurrent opponents (trained zombie policies) clear the memory of finished envs.
            if hasattr(zombies, "reset"):
                with timer("opponent"):
                    zombies.reset(step.terminated | step.truncated)
            parts.append(step)
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


@dataclass
class ZombieBatch:
    obs: np.ndarray  # [A, ZOMBIE_OBS_DIM]
    state: np.ndarray  # [E, STATE_DIM]


@dataclass
class ZombieStep:
    next: ZombieBatch
    reward: np.ndarray  # [A]
    terminated: np.ndarray  # [E] bool
    truncated: np.ndarray  # [E] bool
    outcome: np.ndarray  # [E] uint8
    episode_length: np.ndarray  # [E]
    final_obs: np.ndarray  # [A, ZOMBIE_OBS_DIM], rows valid where the agent's env truncated
    final_state: np.ndarray  # [E, STATE_DIM], rows valid where truncated


class ZombieTrainingEnv:
    """One vec env per zombie count; the human acts with `human_factory` policies."""

    def __init__(
        self,
        config: dict[str, Any],
        zombie_counts: list[int],
        num_envs: int,
        human_factory: HumanFactory | list[HumanFactory],
        seed: int = 0,
        num_threads: int | None = None,
    ):
        sizes = group_sizes(num_envs, len(zombie_counts))
        factories = per_group(human_factory, len(zombie_counts))
        self.groups = []
        agent_env = []
        env_offset = 0
        self.group_slices = []
        for i, (n, size) in enumerate(zip(zombie_counts, sizes, strict=True)):
            cfg = deep_merge(config, {"zombie_count": n})
            venv = ZaiVecEnv(cfg, num_envs=size, seed=seed + 1000 * i, num_threads=num_threads)
            self.groups.append((venv, factories[i](cfg, size, seed + 1000 * i + 1), n))
            self.group_slices.append(slice(env_offset, env_offset + size))
            agent_env.append(np.repeat(np.arange(size) + env_offset, n))
            env_offset += size
        self.num_envs = num_envs
        self.agent_env = np.concatenate(agent_env)
        self.num_agents = len(self.agent_env)
        self._human: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self) -> ZombieBatch:
        obs = [venv.reset() for venv, _, _ in self.groups]
        self._human = [(o.human_obs.copy(), o.human_mask.copy()) for o in obs]
        return ZombieBatch(
            obs=np.concatenate([o.zombie_obs.reshape(-1, o.zombie_obs.shape[-1]) for o in obs]),
            state=np.concatenate([o.state for o in obs]),
        )

    def step(self, zombie_actions: np.ndarray) -> ZombieStep:
        parts = []
        start = 0
        for (venv, human, n), (hobs, hmask) in zip(self.groups, self._human, strict=True):
            end = start + venv.num_envs * n
            za = zombie_actions[start:end].reshape(venv.num_envs, n)
            with timer("opponent"):
                human_actions = human.act(hobs, hmask)
            with timer("sim"):
                step = venv.step(human_actions, za)
            if hasattr(human, "reset"):
                with timer("opponent"):
                    human.reset(step.terminated | step.truncated)
            parts.append(step)
            start = end
        self._human = [(p.obs.human_obs.copy(), p.obs.human_mask.copy()) for p in parts]

        def cat(f):
            return np.concatenate([f(p) for p in parts])

        def flat(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        return ZombieStep(
            next=ZombieBatch(
                obs=cat(lambda p: flat(p.obs.zombie_obs)),
                state=cat(lambda p: p.obs.state),
            ),
            reward=cat(lambda p: p.zombie_reward.reshape(-1)),
            terminated=cat(lambda p: p.terminated),
            truncated=cat(lambda p: p.truncated),
            outcome=cat(lambda p: p.outcome),
            episode_length=cat(lambda p: p.episode_length),
            final_obs=cat(lambda p: flat(p.final_zombie_obs)),
            final_state=cat(lambda p: p.final_state),
        )
