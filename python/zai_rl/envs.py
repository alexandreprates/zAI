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


@dataclass(frozen=True)
class GroupSpec:
    """One group of the league (docs/ARCHITECTURE.md section 4.8.4): in kind A both learners
    play, in B the human learner plays frozen zombies, in C the zombie learner plays a frozen
    human."""

    kind: str  # A | B | C
    zombie_count: int
    num_envs: int


def league_groups(
    num_envs: int, zombie_counts: list[int], shares: tuple[float, float, float]
) -> list[GroupSpec]:
    """Groups A/B/C with `shares` of `num_envs`, each split equally across `zombie_counts`.

    The groups are ordered B, A, C, so the human learner's envs (A and B) and the zombie
    learner's envs (A and C) are both contiguous and `MatchTrainingEnv` selects them by slicing.
    """
    per_kind = [int(round(num_envs * s)) for s in shares]
    per_kind[0] = num_envs - per_kind[1] - per_kind[2]
    groups = []
    for kind, n in sorted(zip("ABC", per_kind, strict=True), key=lambda kn: "BAC".index(kn[0])):
        if n == 0:
            continue
        for count, size in zip(zombie_counts, group_sizes(n, len(zombie_counts)), strict=True):
            groups.append(GroupSpec(kind, count, size))
    return groups


@dataclass
class MatchObs:
    human_obs: np.ndarray  # [E, HUMAN_OBS_DIM]
    human_mask: np.ndarray  # [E, 20] bool
    state: np.ndarray  # [E, STATE_DIM]
    zombie_obs: np.ndarray  # [A, ZOMBIE_OBS_DIM], every zombie of every env


@dataclass
class MatchStep:
    next: MatchObs
    human_reward: np.ndarray  # [E]
    zombie_reward: np.ndarray  # [A]
    terminated: np.ndarray  # [E] bool
    truncated: np.ndarray  # [E] bool
    outcome: np.ndarray  # [E] uint8
    episode_length: np.ndarray  # [E]
    final_zombie_obs: np.ndarray  # [A], rows valid where the agent's env truncated
    final_state: np.ndarray  # [E], rows valid where truncated
    final_stats: np.ndarray  # [E, STATS_DIM], rows valid where the episode ended


class MatchTrainingEnv:
    """All league groups as one batch, with both sides' actions supplied by the caller.

    Envs `[E]` are ordered by group; every zombie of every env sits on a flat axis `[A]` with
    `agent_env [A]` (env of each agent). `kind [E]` is the group kind of each env. Learner and
    opponent rows are selected with the index arrays below; `human_view` / `zombie_view` build
    the `HumanStep` / `ZombieStep` the learners' rollouts record.
    """

    def __init__(
        self,
        config: dict[str, Any],
        groups: list[GroupSpec],
        seed: int = 0,
        num_threads: int | None = None,
    ):
        self.specs = groups
        self.venvs: list[ZaiVecEnv] = []
        self.group_cfgs: list[dict[str, Any]] = []
        kind, agent_env, env_slices, agent_slices = [], [], [], []
        e0 = a0 = 0
        for i, g in enumerate(groups):
            cfg = deep_merge(config, {"zombie_count": g.zombie_count})
            self.group_cfgs.append(cfg)
            self.venvs.append(
                ZaiVecEnv(cfg, num_envs=g.num_envs, seed=seed + 1000 * i, num_threads=num_threads)
            )
            kind += [g.kind] * g.num_envs
            agent_env.append(np.repeat(np.arange(g.num_envs) + e0, g.zombie_count))
            env_slices.append(slice(e0, e0 + g.num_envs))
            agent_slices.append(slice(a0, a0 + g.num_envs * g.zombie_count))
            e0 += g.num_envs
            a0 += g.num_envs * g.zombie_count
        self.num_envs, self.num_agents = e0, a0
        self.kind = np.array(kind)
        self.agent_env = np.concatenate(agent_env)
        self.env_slices, self.agent_slices = env_slices, agent_slices
        # Learner rows.
        self.human_rows = np.nonzero(np.isin(self.kind, ["A", "B"]))[0]
        self.zombie_envs = np.nonzero(np.isin(self.kind, ["A", "C"]))[0]
        self.zombie_agents = np.nonzero(np.isin(self.kind[self.agent_env], ["A", "C"]))[0]
        local = np.full(self.num_envs, -1)
        local[self.zombie_envs] = np.arange(len(self.zombie_envs))
        self.zombie_agent_env = local[self.agent_env[self.zombie_agents]]  # into zombie_envs
        # Opponent rows: frozen humans in C, frozen zombies in B.
        self.opp_human_rows = np.nonzero(self.kind == "C")[0]
        self.opp_zombie_agents = np.nonzero(self.kind[self.agent_env] == "B")[0]
        # The same learner rows as slices when contiguous (groups ordered B, A, C), so that
        # selecting them gives views instead of copies.
        self.human_sel = _selector(self.human_rows)
        self.zombie_env_sel = _selector(self.zombie_envs)
        self.zombie_agent_sel = _selector(self.zombie_agents)

    def _cat(self, parts: list[Any], f) -> np.ndarray:
        return np.concatenate([f(p) for p in parts])

    def _cat_if(self, parts: list[Any], f, rows: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """`_cat`, or `fallback` when no part has a valid row (`rows`: per-env flags)."""
        return self._cat(parts, f) if rows.any() else fallback

    def _obs(self, parts: list[Any]) -> MatchObs:
        return MatchObs(
            human_obs=self._cat(parts, lambda o: o.human_obs),
            human_mask=self._cat(parts, lambda o: o.human_mask),
            state=self._cat(parts, lambda o: o.state),
            zombie_obs=self._cat(parts, lambda o: o.zombie_obs.reshape(-1, o.zombie_obs.shape[-1])),
        )

    def reset(self) -> MatchObs:
        return self._obs([v.reset() for v in self.venvs])

    def step(self, human_actions: np.ndarray, zombie_actions: np.ndarray) -> MatchStep:
        parts = []
        with timer("sim"):
            for v, es, as_ in zip(self.venvs, self.env_slices, self.agent_slices, strict=True):
                za = zombie_actions[as_].reshape(v.num_envs, v.num_zombies)
                parts.append(v.step(human_actions[es], za))

        def flat(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        nxt = self._obs([p.obs for p in parts])
        terminated = self._cat(parts, lambda p: p.terminated)
        truncated = self._cat(parts, lambda p: p.truncated)
        # The final_* rows are only read where an episode truncated (ended), which most steps
        # have none of: skip those copies then (the next observation stands in).
        return MatchStep(
            next=nxt,
            human_reward=self._cat(parts, lambda p: p.human_reward),
            zombie_reward=self._cat(parts, lambda p: p.zombie_reward.reshape(-1)),
            terminated=terminated,
            truncated=truncated,
            outcome=self._cat(parts, lambda p: p.outcome),
            episode_length=self._cat(parts, lambda p: p.episode_length),
            final_zombie_obs=self._cat_if(
                parts, lambda p: flat(p.final_zombie_obs), truncated, nxt.zombie_obs
            ),
            final_state=self._cat_if(parts, lambda p: p.final_state, truncated, nxt.state),
            final_stats=self._cat_if(
                parts,
                lambda p: p.final_stats,
                terminated | truncated,
                np.zeros((self.num_envs, parts[0].final_stats.shape[1]), np.float32),
            ),
        )

    def human_view(self, step: MatchStep) -> HumanStep:
        r = self.human_sel
        return HumanStep(
            next=HumanBatch(step.next.human_obs[r], step.next.human_mask[r], step.next.state[r]),
            reward=step.human_reward[r],
            terminated=step.terminated[r],
            truncated=step.truncated[r],
            outcome=step.outcome[r],
            episode_length=step.episode_length[r],
            final_state=step.final_state[r],
        )

    def zombie_view(self, step: MatchStep) -> ZombieStep:
        e, a = self.zombie_env_sel, self.zombie_agent_sel
        return ZombieStep(
            next=ZombieBatch(step.next.zombie_obs[a], step.next.state[e]),
            reward=step.zombie_reward[a],
            terminated=step.terminated[e],
            truncated=step.truncated[e],
            outcome=step.outcome[e],
            episode_length=step.episode_length[e],
            final_obs=step.final_zombie_obs[a],
            final_state=step.final_state[e],
        )


def _selector(idx: np.ndarray) -> slice | np.ndarray:
    """`idx` as a slice when it is a contiguous ascending range, else unchanged."""
    if len(idx) and idx[-1] - idx[0] + 1 == len(idx) and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx
