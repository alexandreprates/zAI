"""Agent API against a real `zai-server` subprocess (docs/ARCHITECTURE.md 4.9.5).

The binary comes from the `cargo test` step of `make test` (`$CARGO_TARGET_DIR/debug`); the
tests are skipped when it has not been built.
"""

import dataclasses
import json
import os
import socket
import subprocess
from pathlib import Path

import grpc
import pytest
import zai_sim

from zai_agent.client import PROTOCOL_VERSION, Client
from zai_agent.compat import IncompatibleServer, check
from zai_env.layout import HUMAN, ZOMBIE


def _binary() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).parents[2] / "target"))
    return target / "debug" / "zai-server"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    binary = _binary()
    if not binary.is_file():
        pytest.skip(f"{binary} not built (run `cargo test -p zai-server` first)")
    http, grpc_port = _free_port(), _free_port()
    env = {
        **os.environ,
        "ZAI_HTTP_ADDR": f"127.0.0.1:{http}",
        "ZAI_GRPC_ADDR": f"127.0.0.1:{grpc_port}",
        "ZAI_DATA_DIR": str(tmp_path_factory.mktemp("data")),
        "RUST_LOG": "warn",
    }
    proc = subprocess.Popen([str(binary)], env=env)
    target = f"127.0.0.1:{grpc_port}"
    try:
        grpc.channel_ready_future(grpc.insecure_channel(target)).result(timeout=20)
        yield target
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_get_info_matches_the_local_build(server):
    with Client(server) as client:
        info = client.info()
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


def test_compat_refuses_another_layout(server):
    with Client(server) as client:
        info = client.info()
    stale = dataclasses.replace(info, obs_version=info.obs_version - 1)
    with pytest.raises(IncompatibleServer, match="obs_version"):
        check(stale)
    shrunk = dataclasses.replace(
        info, human=dataclasses.replace(info.human, obs_dim=info.human.obs_dim - 1)
    )
    with pytest.raises(IncompatibleServer, match="human obs_dim"):
        check(shrunk)
