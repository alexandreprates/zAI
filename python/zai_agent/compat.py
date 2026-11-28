"""Checks that a server speaks the observation layout of the local `zai_sim` build."""

from __future__ import annotations

import zai_sim

from zai_env import layout

from .client import PROTOCOL_VERSION, ServerInfo, SideInfo


class IncompatibleServer(RuntimeError):
    pass


def check(info: ServerInfo) -> None:
    """Raises `IncompatibleServer` unless protocol, `OBS_VERSION` and both observation sizes
    equal the local build's (a policy trained on another layout would read garbage)."""
    local = {
        "protocol": PROTOCOL_VERSION,
        "obs_version": zai_sim.OBS_VERSION,
        "human obs_dim": zai_sim.HUMAN_OBS_DIM,
        "zombie obs_dim": zai_sim.ZOMBIE_OBS_DIM,
    }
    remote = {
        "protocol": info.protocol,
        "obs_version": info.obs_version,
        "human obs_dim": info.human.obs_dim,
        "zombie obs_dim": info.zombie.obs_dim,
    }
    diff = [f"{k}: server {remote[k]}, local {v}" for k, v in local.items() if remote[k] != v]
    for side, spec, local_layout in (
        ("human", info.human, layout.HUMAN),
        ("zombie", info.zombie, layout.ZOMBIE),
    ):
        if _blocks(spec) != _local_blocks(local_layout):
            diff.append(f"{side} observation blocks differ")
    if diff:
        raise IncompatibleServer(f"server {info.version} is incompatible: " + "; ".join(diff))


def _blocks(spec: SideInfo) -> dict[str, tuple[int, int]]:
    return dict(spec.blocks)


def _local_blocks(lay: layout.Layout) -> dict[str, tuple[int, int]]:
    return {n: (b["start"], b["slots"] * len(b["fields"])) for n, b in lay.blocks.items()}
