"""Scripted human, in priority order:

1. flee visible zombies closer than `flee_m` (away from their inverse-distance-weighted mean);
2. when carrying fewer than `reserve` packages, go to the nearest visible food and pick it up
   (food is eaten automatically from the inventory; one spare is often not enough because
   visible food can be 100 m away, i.e. 100 s at 1 m/s, longer than a 90 s cycle; but each
   package costs 20% of base speed, and two packages make the human slower than zombies);
3. otherwise walk to a random house, choosing a new one on arrival (random waypoints when the
   map has no houses).
"""

from typing import Any

import numpy as np
import zai_sim

from zai_env.layout import HUMAN

from .common import direction_to_action

PICKUP = 1


class ScriptedHuman:
    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int,
        seed: int = 0,
        flee_m: float = 30.0,
        reserve: int = 2,
        arrive_m: float = 3.0,
    ):
        self.size = config["map_size_m"]
        self.vision = config["human"]["vision_radius_m"]
        self.houses = config["house_count"]
        self.flee_m = flee_m
        self.reserve = reserve
        self.arrive_m = arrive_m
        self.rng = np.random.default_rng(seed)
        self.num_envs = num_envs
        self.target_house = self.rng.integers(0, max(self.houses, 1), num_envs)
        self.waypoints = self.rng.uniform(0.0, 1.0, (num_envs, 2))

    def act(self, human_obs: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
        rows = np.arange(self.num_envs)

        # 1. Flee.
        z_valid = HUMAN.field(human_obs, "zombies", "valid") > 0.5
        z_dist = HUMAN.field(human_obs, "zombies", "dist") * self.vision
        threat = z_valid & (z_dist < self.flee_m)
        weight = threat / np.maximum(z_dist, 1e-3)
        flee_dx = -(weight * HUMAN.field(human_obs, "zombies", "dx")).sum(axis=1)
        flee_dy = -(weight * HUMAN.field(human_obs, "zombies", "dy")).sum(axis=1)
        flee = threat.any(axis=1)

        # 2. Food.
        carried = np.rint(HUMAN.scalar(human_obs, "self", "carried") * 5)
        hungry = carried < self.reserve
        food_valid = HUMAN.field(human_obs, "food", "valid")[:, 0] > 0.5
        go_food = hungry & food_valid
        can_pickup = human_mask[:, zai_sim.MOVE_ACTIONS + PICKUP]

        # 3. Wander between houses (or waypoints).
        if self.houses > 0:
            hx = HUMAN.field(human_obs, "houses", "dx")[rows, self.target_house]
            hy = HUMAN.field(human_obs, "houses", "dy")[rows, self.target_house]
            arrived = np.hypot(hx, hy) * self.size < self.arrive_m
            self.target_house[arrived] = self.rng.integers(0, self.houses, int(arrived.sum()))
            wander_dx = HUMAN.field(human_obs, "houses", "dx")[rows, self.target_house]
            wander_dy = HUMAN.field(human_obs, "houses", "dy")[rows, self.target_house]
        else:
            pos = np.stack(
                [HUMAN.scalar(human_obs, "self", "x"), HUMAN.scalar(human_obs, "self", "y")],
                axis=-1,
            )
            to_wp = self.waypoints - pos
            arrived = np.linalg.norm(to_wp, axis=-1) * self.size < self.arrive_m
            self.waypoints[arrived] = self.rng.uniform(0.0, 1.0, (int(arrived.sum()), 2))
            wander_dx, wander_dy = (self.waypoints - pos).T

        food_dx = HUMAN.field(human_obs, "food", "dx")[:, 0]
        food_dy = HUMAN.field(human_obs, "food", "dy")[:, 0]
        dx = np.where(flee, flee_dx, np.where(go_food, food_dx, wander_dx))
        dy = np.where(flee, flee_dy, np.where(go_food, food_dy, wander_dy))
        move = direction_to_action(dx, dy)
        interact = np.where(go_food & can_pickup & ~flee, PICKUP, 0)
        return np.stack([move, interact], axis=1).astype(np.int64)
