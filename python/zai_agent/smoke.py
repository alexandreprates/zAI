"""gRPC parity smoke test against a running server (`make grpc-smoke`, arena container): the
checks of `zai_agent.parity` on `s0_small` and the PRD defaults, sampled and deterministic."""

from __future__ import annotations

import argparse
import os

from zai_env.config import config_dir, load_config

from .parity import play_and_verify


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--server", default=os.environ.get("ZAI_GRPC_TARGET", "server:50051"))
    p.add_argument("--http", default=os.environ.get("ZAI_SERVER_URL"))
    p.add_argument("--token", default=os.environ.get("ZAI_AGENT_TOKEN") or None)
    args = p.parse_args()
    configs = {
        "s0_small": load_config(config_dir() / "s0_small.yaml", env_overrides=False),
    "s0": load_config(config_dir() / "s0.yaml", env_overrides=False),
        "default": load_config(overrides={"episode_cap_s": 40.0}, env_overrides=False),
    }
    for name, cfg in configs.items():
        for det in (False, True):
            ticks = play_and_verify(args.server, cfg, det, http=args.http, token=args.token)
            print(f"{name} {'deterministic' if det else 'sampled'}: {ticks} decisions match")
    print("gRPC parity OK")


if __name__ == "__main__":
    main()
