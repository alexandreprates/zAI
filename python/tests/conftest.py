import pytest

from zai_env.config import load_config


@pytest.fixture
def small_config():
    """S0-like configuration: small map, few zombies, short episodes so resets happen."""
    return load_config(
        overrides={
            "map_size_m": 150.0,
            "house_count": 3,
            "zombie_count": 4,
            "food_count": 20,
            "episode_cap_s": 30.0,
            "min_spawn_distance_m": 20.0,
        },
        env_overrides=False,
    )


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def zai_server(tmp_path_factory):
    """A real `zai-server` (the binary built by the `cargo test` step of `make test`) on free
    ports: yields `(grpc_target, http_url)`. Skipped when the binary has not been built."""
    import os
    import subprocess
    from pathlib import Path

    import grpc

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).parents[2] / "target"))
    binary = target_dir / "debug" / "zai-server"
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
        yield target, f"http://127.0.0.1:{http}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
