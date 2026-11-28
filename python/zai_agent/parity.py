"""Observation parity of the gRPC path with training (docs/ARCHITECTURE.md 4.9.5, step 2).

`play_and_verify` plays both sides of a live server match over gRPC with randomly initialized
actors (dims from the local layout, so nothing to regenerate when a layout changes), then
replays the recorded wire traffic in a local `zai_sim.Env` on the same config, seed and
actions: per tick the observation and mask bytes, `state_hash` and rewards must match
exactly, and so must the outcome. In the deterministic episode, fresh in-process policies fed
the local observations must choose the actions that went over the wire. With `http`, the
server's replay must hold the same action log and final hash. Raises `AssertionError`.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

import numpy as np
import torch
import zai_sim

from zai_env.vec import OUTCOMES
from zai_rl.models import HumanActor, ZombieActor
from zai_rl.policy import RecurrentHumanPolicy, RecurrentZombiePolicy

from .client import Client, EpisodeResult, digest


def _actors() -> tuple[HumanActor, ZombieActor]:
    torch.manual_seed(0)
    return HumanActor(hidden=32).eval(), ZombieActor(hidden=32).eval()


async def _play(target, token, config, seed, actors, deterministic):
    ha, za = actors
    async with Client(target, token) as client:
        mid = await client.create_match(config=config, seed=seed)
        h, z = await asyncio.gather(
            client.play(
                mid,
                "human",
                lambda c, n: RecurrentHumanPolicy(ha, 1, "cpu", deterministic),
                "parity-human",
                record=True,
            ),
            client.play(
                mid,
                "zombies",
                lambda c, n: RecurrentZombiePolicy(za, 1, n, "cpu", deterministic),
                "parity-zombies",
                record=True,
            ),
        )
    return h[0], z[0]


def verify(h: EpisodeResult, z: EpisodeResult, actors, deterministic: bool) -> None:
    """Replays one recorded episode locally and checks it tick by tick."""
    assert len(h.record) == len(z.record) == h.ticks > 0
    env = zai_sim.Env(json.dumps(h.config), h.seed)
    n = env.num_zombies
    local_h = RecurrentHumanPolicy(actors[0], 1, "cpu", deterministic=True)
    local_z = RecurrentZombiePolicy(actors[1], 1, n, "cpu", deterministic=True)
    prev_hr, prev_zr = [0.0], [0.0] * n
    for ht, zt in zip(h.record, z.record, strict=True):
        assert ht.tick == zt.tick == env.decisions
        assert ht.state_hash == zt.state_hash == env.state_hash(), f"state at tick {ht.tick}"
        hobs, mask, zobs = env.human_obs(), env.human_mask(), env.zombie_obs()
        assert ht.obs == digest(hobs.tobytes()), f"human obs at tick {ht.tick}"
        assert ht.mask == digest(mask.astype(np.uint8).tobytes()), f"mask at tick {ht.tick}"
        assert zt.obs == digest(zobs.tobytes()), f"zombie obs at tick {zt.tick}"
        assert ht.reward == prev_hr and zt.reward == prev_zr, f"rewards at tick {ht.tick}"
        ha, za = np.frombuffer(ht.actions, np.uint8), np.frombuffer(zt.actions, np.uint8)
        if deterministic:
            assert (local_h.act(hobs[None], mask[None])[0] == ha).all(), f"tick {ht.tick}"
            assert (local_z.act(zobs[None])[0] == za).all(), f"tick {ht.tick}"
        hr, zr, *_ = env.step(int(ha[0]), int(ha[1]), za.tolist())
        prev_hr, prev_zr = [hr], list(zr)
    assert OUTCOMES[env.outcome] == h.outcome == z.outcome
    assert env.state_hash() == h.state_hash == z.state_hash
    assert h.reward == prev_hr and z.reward == prev_zr


def verify_replay(http: str, h: EpisodeResult, z: EpisodeResult) -> None:
    with urllib.request.urlopen(f"{http}/api/replays/{h.replay_id}") as r:
        replay = json.load(r)
    pairs = zip(h.record, z.record, strict=True)
    assert replay["actions"] == [list(ht.actions) + list(zt.actions) for ht, zt in pairs]
    assert replay["human_model"] == "parity-human"
    assert int(replay["final_hash"], 16) == h.state_hash


def play_and_verify(
    target: str,
    config: dict[str, Any],
    deterministic: bool,
    seed: int = 11,
    http: str | None = None,
    token: str | None = None,
) -> int:
    """Plays and verifies one episode; returns its number of decisions."""
    actors = _actors()
    h, z = asyncio.run(_play(target, token, config, seed, actors, deterministic))
    verify(h, z, actors, deterministic)
    if http:
        verify_replay(http, h, z)
    return h.ticks
