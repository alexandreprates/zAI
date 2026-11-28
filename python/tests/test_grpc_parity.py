"""Observation parity of the gRPC path with training (docs/ARCHITECTURE.md 4.9.5, step 2).

Both sides of a live server match are played over gRPC by randomly initialized actors (dims
from the local layout, so no fixture to regenerate when a layout changes). The recorded wire
traffic is then replayed in a local `zai_sim.Env` on the same config, seed and actions: per
tick the observation and mask bytes, `state_hash` and rewards must match exactly, and so
must the outcome. In the deterministic episode, fresh in-process policies fed the local
observations must choose the actions that went over the wire. The server's replay holds the
same action log.
"""

import asyncio
import json
import urllib.request

import numpy as np
import pytest
import torch
import zai_sim

from zai_agent.client import Client, digest
from zai_env.config import config_dir, load_config
from zai_env.vec import OUTCOMES
from zai_rl.models import HumanActor, ZombieActor
from zai_rl.policy import RecurrentHumanPolicy, RecurrentZombiePolicy

CONFIGS = {
    "s0_small": load_config(config_dir() / "s0_small.yaml", env_overrides=False),
    # The PRD defaults (500 m, 20 zombies) with a short cap.
    "default": load_config(overrides={"episode_cap_s": 40.0}, env_overrides=False),
}


def _actors():
    torch.manual_seed(0)
    return HumanActor(hidden=32).eval(), ZombieActor(hidden=32).eval()


async def _play(target, config, seed, actors, deterministic):
    ha, za = actors
    async with Client(target) as client:
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


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("deterministic", [False, True])
def test_grpc_observations_equal_the_training_env(zai_server, name, deterministic):
    target, http = zai_server
    actors = _actors()
    h, z = asyncio.run(_play(target, CONFIGS[name], 11, actors, deterministic))
    assert len(h.record) == len(z.record) == h.ticks > 0

    env = zai_sim.Env(json.dumps(h.config), h.seed)
    local_h = RecurrentHumanPolicy(actors[0], 1, "cpu", deterministic=True)
    local_z = RecurrentZombiePolicy(actors[1], 1, env.num_zombies, "cpu", deterministic=True)
    prev_hr, prev_zr = [0.0], [0.0] * env.num_zombies
    for ht, zt in zip(h.record, z.record, strict=True):
        assert ht.tick == zt.tick == env.decisions
        assert ht.state_hash == zt.state_hash == env.state_hash()
        hobs, mask, zobs = env.human_obs(), env.human_mask(), env.zombie_obs()
        assert ht.obs == digest(hobs.tobytes()), f"human obs at tick {ht.tick}"
        assert ht.mask == digest(mask.astype(np.uint8).tobytes())
        assert zt.obs == digest(zobs.tobytes()), f"zombie obs at tick {zt.tick}"
        assert ht.reward == prev_hr and zt.reward == prev_zr, f"rewards at tick {ht.tick}"
        ha, za = np.frombuffer(ht.actions, np.uint8), np.frombuffer(zt.actions, np.uint8)
        if deterministic:
            assert (local_h.act(hobs[None], mask[None])[0] == ha).all()
            assert (local_z.act(zobs[None])[0] == za).all()
        hr, zr, *_ = env.step(int(ha[0]), int(ha[1]), za.tolist())
        prev_hr, prev_zr = [hr], list(zr)
    assert OUTCOMES[env.outcome] == h.outcome == z.outcome
    assert env.state_hash() == h.state_hash
    assert h.reward == prev_hr and z.reward == prev_zr

    with urllib.request.urlopen(f"{http}/api/replays/{h.replay_id}") as r:
        replay = json.load(r)
    pairs = zip(h.record, z.record, strict=True)
    wire = [list(ht.actions) + list(zt.actions) for ht, zt in pairs]
    assert replay["actions"] == wire
    assert replay["human_model"] == "parity-human"
    assert int(replay["final_hash"], 16) == h.state_hash
