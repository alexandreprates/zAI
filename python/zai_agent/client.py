"""gRPC client of the ZAI agent API (proto/zai/v1/agent.proto, docs/ARCHITECTURE.md 4.9).

M7.0: `Client.info()` returns the server's versions and observation layouts; `compat` checks
them against the local `zai_sim` build. Playing a side (`play`) lands with M7.1-M7.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import grpc

from .proto import agent_pb2, agent_pb2_grpc

PROTOCOL_VERSION = 1


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


class Client:
    """Connection to a server's agent API, e.g. `Client("server:50051")`."""

    def __init__(self, target: str, timeout_s: float = 10.0):
        self.target = target
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(target)
        self._stub = agent_pb2_grpc.AgentServiceStub(self._channel)

    def info(self) -> ServerInfo:
        r = self._stub.GetInfo(agent_pb2.GetInfoRequest(), timeout=self.timeout_s)
        return ServerInfo(
            version=r.version,
            protocol=r.protocol,
            obs_version=r.obs_version,
            human=_side(r.human),
            zombie=_side(r.zombie),
            base_config_json=r.base_config_json,
        )

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
