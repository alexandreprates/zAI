"""Batched matches between fixed policies."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from zai_env.vec import OUTCOME_CAUGHT, OUTCOME_STARVED, OUTCOME_TRUNCATED, ZaiVecEnv

HumanFactory = Callable[[dict[str, Any], int, int], Any]
ZombieFactory = Callable[[dict[str, Any], int, int, int], Any]


@dataclass(frozen=True)
class MatchStats:
    episodes: int
    caught: float  # fraction of episodes
    starved: float
    survived: float  # reached the episode cap
    mean_length_s: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def play(
    config: dict[str, Any],
    human_factory: HumanFactory,
    zombie_factory: ZombieFactory,
    episodes: int = 64,
    seed: int = 0,
    num_threads: int | None = None,
) -> MatchStats:
    """Plays one episode per environment (`episodes` environments in parallel).

    Policies may define `reset(done_mask)` to clear per-env memory when an episode ends.

    Only the first episode of each environment is counted, so short episodes are not
    over-represented.
    """
    venv = ZaiVecEnv(config, num_envs=episodes, seed=seed, num_threads=num_threads)
    human = human_factory(config, episodes, seed)
    zombies = zombie_factory(config, episodes, venv.num_zombies, seed + 1)
    decision_s = config["dt_s"] * config["action_repeat"]

    obs = venv.reset()
    finished = np.zeros(episodes, dtype=bool)
    outcome = np.zeros(episodes, dtype=np.uint8)
    length = np.zeros(episodes, dtype=np.float64)
    while not finished.all():
        step = venv.step(human.act(obs.human_obs, obs.human_mask), zombies.act(obs.zombie_obs))
        done = step.terminated | step.truncated
        for policy in (human, zombies):
            if hasattr(policy, "reset"):
                policy.reset(done)
        ended = done & ~finished
        outcome[ended] = step.outcome[ended]
        length[ended] = step.episode_length[ended] * decision_s
        finished |= ended
        obs = step.obs

    return MatchStats(
        episodes=episodes,
        caught=float((outcome == OUTCOME_CAUGHT).mean()),
        starved=float((outcome == OUTCOME_STARVED).mean()),
        survived=float((outcome == OUTCOME_TRUNCATED).mean()),
        mean_length_s=float(length.mean()),
    )
