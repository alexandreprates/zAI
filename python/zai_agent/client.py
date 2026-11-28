"""gRPC client of the ZAI agent API (proto/zai/v1/agent.proto, docs/ARCHITECTURE.md 4.9).

`Client(target)` (asyncio, `grpc.aio`): `info()` returns the server's versions and layouts,
`create_match(...)` creates a match, `play(match_id, side, make_policy)` plays one side until
the match ends. One process can drive both sides of a match by running two `play` calls
concurrently (`asyncio.gather`). Actions are sampled here, by the policy; the server only
applies them.

A policy has the bot interface of `zai_rl.bots` with a batch of one environment: the human
`act(obs [1, D], mask [1, 20]) -> [1, 2]`, the zombies `act(obs [1, N, D]) -> [1, N]`, and an
optional `reset(ended [1])`. `make_policy(config, num_agents)` builds it at the first episode.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np
import zai_sim

from .proto import agent_pb2, agent_pb2_grpc

PROTOCOL_VERSION = 1
SIDES = {"human": agent_pb2.SIDE_HUMAN, "zombies": agent_pb2.SIDE_ZOMBIES}


@dataclass(frozen=True)
class SideInfo:
    obs_dim: int
    blocks: dict[str, tuple[int, int]]  # name -> (start, len)
    action_dims: tuple[int, ...]


@dataclass(frozen=True)
class ServerInfo:
    version: str
    protocol: int
    obs_version: int
    human: SideInfo
    zombie: SideInfo
    base_config_json: str


def _side(spec: agent_pb2.SideSpec) -> SideInfo:
    return SideInfo(
        obs_dim=spec.obs_dim,
        blocks={b.name: (b.start, b.len) for b in spec.blocks},
        action_dims=tuple(spec.action_dims),
    )


def digest(data: bytes) -> str:
    """Short hash of observation or mask bytes (parity records)."""
    return hashlib.blake2b(data, digest_size=8).hexdigest()


@dataclass
class Tick:
    """What one decision looked like on the wire (kept with `play(..., record=True)`)."""

    tick: int
    obs: str  # digest of the observation bytes
    mask: str
    reward: list[float]
    state_hash: int
    actions: bytes


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    config: dict[str, Any]
    outcome: str
    ticks: int
    state_hash: int
    replay_id: str
    reward: list[float]
    record: list[Tick] = field(default_factory=list)


class MatchFailed(RuntimeError):
    pass


class Client:
    """Connection to a server's agent API, e.g. `async with Client("server:50051") as c`.

    `token` is sent as `authorization: Bearer <token>` (the server's `ZAI_AGENT_TOKEN`);
    `tls_root` (a PEM file, e.g. Caddy's local root certificate) switches to TLS, for the
    Caddy gRPC site of `compose.tls.yaml`."""

    def __init__(
        self,
        target: str,
        token: str | None = None,
        tls_root: str | None = None,
        timeout_s: float = 10.0,
    ):
        self.target = target
        self.timeout_s = timeout_s
        if tls_root:
            with open(tls_root, "rb") as f:
                creds = grpc.ssl_channel_credentials(root_certificates=f.read())
            self._channel = grpc.aio.secure_channel(target, creds)
        else:
            self._channel = grpc.aio.insecure_channel(target)
        self._metadata = (("authorization", f"Bearer {token}"),) if token else None
        self._stub = agent_pb2_grpc.AgentServiceStub(self._channel)

    async def info(self) -> ServerInfo:
        r = await self._stub.GetInfo(
            agent_pb2.GetInfoRequest(), timeout=self.timeout_s, metadata=self._metadata
        )
        return ServerInfo(
            version=r.version,
            protocol=r.protocol,
            obs_version=r.obs_version,
            human=_side(r.human),
            zombie=_side(r.zombie),
            base_config_json=r.base_config_json,
        )

    async def create_match(
        self,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        human: str = "remote",
        zombies: str = "remote",
        speed: float = 0.0,
        auto_restart: bool = False,
        deterministic: bool = False,
        realtime: bool = False,
    ) -> str:
        """Creates a match; sides are `remote`, `scripted`, `random` or `policy:<model id>`."""
        req = agent_pb2.CreateMatchRequest(
            config_json=json.dumps(config) if config else "",
            human=human,
            zombies=zombies,
            speed=speed,
            auto_restart=auto_restart,
            deterministic=deterministic,
            realtime=realtime,
        )
        if seed is not None:
            req.seed = seed
        r = await self._stub.CreateMatch(req, timeout=self.timeout_s, metadata=self._metadata)
        return r.match_id

    async def play(
        self,
        match_id: str,
        side: str,
        make_policy: Callable[[dict[str, Any], int], Any],
        agent_name: str = "",
        record: bool = False,
    ) -> list[EpisodeResult]:
        """Plays `side` (`human` or `zombies`) of a match until it ends; returns its episodes.
        Raises `MatchFailed` when the match fails and `grpc.aio.AioRpcError` on a refused join."""
        dim = zai_sim.HUMAN_OBS_DIM if side == "human" else zai_sim.ZOMBIE_OBS_DIM
        outgoing: asyncio.Queue = asyncio.Queue()
        await outgoing.put(
            agent_pb2.ClientMessage(
                join=agent_pb2.Join(
                    match_id=match_id,
                    side=SIDES[side],
                    protocol=PROTOCOL_VERSION,
                    obs_version=zai_sim.OBS_VERSION,
                    obs_dim=dim,
                    agent_name=agent_name,
                )
            )
        )

        async def requests():
            while (msg := await outgoing.get()) is not None:
                yield msg

        call = self._stub.Play(requests(), metadata=self._metadata)
        policy, config, episodes, ticks = None, {}, [], []
        start = None
        try:
            async for m in call:
                kind = m.WhichOneof("msg")
                if kind == "start":
                    start = m.start
                    config = json.loads(start.config_json)
                    if policy is None:
                        policy = make_policy(config, start.num_agents)
                    elif hasattr(policy, "reset"):
                        policy.reset(np.ones(1, dtype=bool))
                    ticks = []
                elif kind == "obs":
                    o = m.obs
                    # [num_agents, obs_dim]: the human is one env x one agent ([1, D]), the
                    # zombies one env of N agents ([1, N, D] for the policy).
                    obs = np.frombuffer(o.obs, dtype="<f4").reshape(o.num_agents, o.obs_dim).copy()
                    if side == "human":
                        mask = np.frombuffer(o.mask, dtype=np.uint8).astype(bool)[None]
                        a = policy.act(obs, mask)[0]
                    else:
                        a = policy.act(obs[None])[0]
                    actions = np.asarray(a, dtype=np.uint8).tobytes()
                    if record:
                        ticks.append(
                            Tick(
                                o.tick,
                                digest(o.obs),
                                digest(o.mask),
                                list(o.reward),
                                o.state_hash,
                                actions,
                            )
                        )
                    await outgoing.put(
                        agent_pb2.ClientMessage(
                            actions=agent_pb2.Actions(
                                episode=o.episode, tick=o.tick, actions=actions
                            )
                        )
                    )
                elif kind == "end":
                    e = m.end
                    episodes.append(
                        EpisodeResult(
                            e.episode,
                            start.seed,
                            config,
                            e.outcome,
                            e.ticks,
                            e.state_hash,
                            e.replay_id,
                            list(e.reward),
                            ticks,
                        )
                    )
                elif kind == "match_end":
                    if m.match_end.status == "failed":
                        raise MatchFailed(m.match_end.reason)
                    break
        finally:
            await outgoing.put(None)
            call.cancel()
        return episodes

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
