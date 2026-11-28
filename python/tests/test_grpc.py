"""Agent API against a real `zai-server` subprocess (docs/ARCHITECTURE.md 4.9)."""

import asyncio
import dataclasses
import json

import grpc
import pytest
import zai_sim

from zai_agent.client import PROTOCOL_VERSION, Client, MatchFailed
from zai_agent.compat import IncompatibleServer, check
from zai_agent.policies import human_policy, zombie_policy
from zai_env.layout import HUMAN, ZOMBIE

SMALL = {
    "map_size_m": 75.0,
    "house_count": 3,
    "zombie_count": 3,
    "food_count": 20,
    "episode_cap_s": 30.0,
    "min_spawn_distance_m": 20.0,
}


async def _info(target):
    async with Client(target) as client:
        return await client.info()


def test_get_info_matches_the_local_build(zai_server):
    info = asyncio.run(_info(zai_server[0]))
    assert info.version == zai_sim.__version__
    assert info.protocol == PROTOCOL_VERSION
    assert info.obs_version == zai_sim.OBS_VERSION
    assert info.human.obs_dim == zai_sim.HUMAN_OBS_DIM
    assert info.zombie.obs_dim == zai_sim.ZOMBIE_OBS_DIM
    assert info.human.action_dims == (zai_sim.MOVE_ACTIONS, zai_sim.INTERACT_ACTIONS)
    assert info.zombie.action_dims == (zai_sim.MOVE_ACTIONS,)
    for spec, lay in ((info.human, HUMAN), (info.zombie, ZOMBIE)):
        assert set(spec.blocks) == set(lay.blocks)
    assert json.loads(info.base_config_json)["zombie_count"] > 0
    check(info)


def test_compat_refuses_another_layout(zai_server):
    info = asyncio.run(_info(zai_server[0]))
    stale = dataclasses.replace(info, obs_version=info.obs_version - 1)
    with pytest.raises(IncompatibleServer, match="obs_version"):
        check(stale)
    shrunk = dataclasses.replace(
        info, human=dataclasses.replace(info.human, obs_dim=info.human.obs_dim - 1)
    )
    with pytest.raises(IncompatibleServer, match="human obs_dim"):
        check(shrunk)


def test_python_bots_play_both_sides_of_a_match(zai_server):
    async def run():
        async with Client(zai_server[0]) as client:
            mid = await client.create_match(config=SMALL, seed=3)
            h, z = await asyncio.gather(
                client.play(mid, "human", human_policy("scripted"), "py-scripted"),
                client.play(mid, "zombies", zombie_policy("random"), "py-random"),
            )
            return h, z

    h, z = asyncio.run(run())
    assert len(h) == len(z) == 1
    assert h[0].outcome in ("caught", "starved", "truncated")
    assert h[0].state_hash == z[0].state_hash and h[0].replay_id


def test_a_refused_join_raises_and_a_failed_match_reports(zai_server):
    async def run():
        async with Client(zai_server[0]) as client:
            mid = await client.create_match(config=SMALL, human="remote", zombies="scripted")
            with pytest.raises(grpc.aio.AioRpcError) as err:
                await client.play(mid, "zombies", zombie_policy("random"))
            assert err.value.code() == grpc.StatusCode.ALREADY_EXISTS

            class Broken:
                def act(self, obs, mask):
                    return [[99, 0]]  # movement out of range

            with pytest.raises((MatchFailed, grpc.aio.AioRpcError)):
                await client.play(mid, "human", lambda c, n: Broken())

    asyncio.run(run())
