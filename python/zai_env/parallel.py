"""PettingZoo `ParallelEnv` over a single `zai_sim.Env` (API compliance, debugging, 3rd-party libs).

Agents: `human` and `zombie_0 .. zombie_{N-1}`. The human's action is `(movement, interact)`;
its action mask is in `infos["human"]["action_mask"]`. All agents end together.
"""

import json
from typing import Any

import numpy as np
import zai_sim
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import load_config
from .vec import OUTCOMES


class ZaiParallelEnv(ParallelEnv):
    metadata = {"name": "zai_v0", "render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        self.config = config if config is not None else load_config()
        self.render_mode = render_mode
        n = self.config["zombie_count"]
        self.possible_agents = ["human"] + [f"zombie_{i}" for i in range(n)]
        self.agents: list[str] = []
        human_obs = spaces.Box(-np.inf, np.inf, (zai_sim.HUMAN_OBS_DIM,), np.float32)
        zombie_obs = spaces.Box(-np.inf, np.inf, (zai_sim.ZOMBIE_OBS_DIM,), np.float32)
        self.observation_spaces = {
            a: human_obs if a == "human" else zombie_obs for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.MultiDiscrete([zai_sim.MOVE_ACTIONS, zai_sim.INTERACT_ACTIONS])
            if a == "human"
            else spaces.Discrete(zai_sim.MOVE_ACTIONS)
            for a in self.possible_agents
        }
        self.state_space = spaces.Box(-np.inf, np.inf, (zai_sim.STATE_DIM,), np.float32)
        self._seeds = np.random.default_rng()
        self._env = zai_sim.Env(json.dumps(self.config), 0)

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seeds = np.random.default_rng(seed)
        self._env.reset(int(self._seeds.integers(0, 2**63)))
        self.agents = list(self.possible_agents)
        return self._observations(), self._infos(0)

    def step(self, actions: dict[str, Any]):
        h = np.asarray(actions.get("human", (zai_sim.MOVE_ACTIONS - 1, 0)), dtype=np.int64)
        zombies = [int(actions.get(a, zai_sim.MOVE_ACTIONS - 1)) for a in self.agents[1:]]
        hr, zr, term, trunc, outcome = self._env.step(int(h[0]), int(h[1]), zombies)
        obs = self._observations()
        rewards = {"human": float(hr)}
        rewards.update({a: float(r) for a, r in zip(self.agents[1:], zr, strict=True)})
        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        infos = self._infos(outcome)
        if term or trunc:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self._env.state()

    def _observations(self) -> dict[str, np.ndarray]:
        zombie = self._env.zombie_obs()
        obs = {"human": self._env.human_obs()}
        for i, a in enumerate(self.possible_agents[1:]):
            obs[a] = zombie[i]
        return obs

    def _infos(self, outcome: int) -> dict[str, dict[str, Any]]:
        infos: dict[str, dict[str, Any]] = {a: {"outcome": OUTCOMES[outcome]} for a in self.agents}
        if "human" in infos:
            infos["human"]["action_mask"] = self._env.human_mask()
        return infos
