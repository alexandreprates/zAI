"""Scripted zombies: walk to the most recent heard noise, otherwise wander between random
waypoints. A zombie blocked by a wall takes a random step to slide around it."""

from typing import Any

import numpy as np
import zai_sim

from zai_env.layout import ZOMBIE

from .common import direction_to_action


class ScriptedZombies:
    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int,
        num_zombies: int,
        seed: int = 0,
        arrive_m: float = 3.0,
    ):
        self.size = config["map_size_m"]
        self.hearing = config["zombie"]["hearing_radius_m"]
        self.arrive_m = arrive_m
        self.rng = np.random.default_rng(seed)
        self.waypoints = self.rng.uniform(0.0, 1.0, (num_envs, num_zombies, 2))

    def act(self, zombie_obs: np.ndarray) -> np.ndarray:
        pos = np.stack(
            [ZOMBIE.scalar(zombie_obs, "self", "x"), ZOMBIE.scalar(zombie_obs, "self", "y")],
            axis=-1,
        )
        blocked = ZOMBIE.scalar(zombie_obs, "self", "blocked") > 0.5

        # Newest heard event is slot 0; chase it until close enough.
        heard_valid = ZOMBIE.field(zombie_obs, "heard", "valid")[..., 0] > 0.5
        heard_dx = ZOMBIE.field(zombie_obs, "heard", "dx")[..., 0]
        heard_dy = ZOMBIE.field(zombie_obs, "heard", "dy")[..., 0]
        heard_dist = ZOMBIE.field(zombie_obs, "heard", "dist")[..., 0] * self.hearing
        chase = heard_valid & (heard_dist > self.arrive_m)

        to_wp = self.waypoints - pos
        reached = np.linalg.norm(to_wp, axis=-1) * self.size < self.arrive_m
        renew = reached | blocked
        self.waypoints[renew] = self.rng.uniform(0.0, 1.0, (int(renew.sum()), 2))
        to_wp = self.waypoints - pos

        dx = np.where(chase, heard_dx, to_wp[..., 0])
        dy = np.where(chase, heard_dy, to_wp[..., 1])
        actions = direction_to_action(dx, dy)
        random = self.rng.integers(0, zai_sim.MOVE_ACTIONS - 1, actions.shape)
        return np.where(blocked, random, actions).astype(np.int64)
