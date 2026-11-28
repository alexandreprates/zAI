"""Plays checkpoints or bots in a live server match over gRPC (docs/ARCHITECTURE.md 4.9.4).

    python -m zai_agent.play --server server:50051 --human ckpt.pt --zombies scripted

Without `--match` it creates one match per episode with the given sides (checkpoints and the
Python bots play as remote sides; `server:scripted` / `server:random` / `server:policy:<id>`
leave a side to the server). With `--match ID` it joins the given remote side(s) of an
existing match. The viewer shows the match live; every episode is stored as a replay.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from zai_env.config import load_config

from .client import Client
from .compat import check
from .policies import human_policy, zombie_policy

log = logging.getLogger("zai_agent.play")


def _server_side(spec: str) -> str | None:
    """Server-played side of a `server:<agent>` spec."""
    return spec.removeprefix("server:") if spec.startswith("server:") else None


def _name(spec: str) -> str:
    p = Path(spec)
    return f"{p.parent.name}/{p.stem}" if p.suffix == ".pt" else spec


async def run(args: argparse.Namespace) -> None:
    async with Client(args.server, args.token, args.tls_root) as client:
        info = await client.info()
        check(info)
        log.info("server %s, protocol %d, obs v%d", info.version, info.protocol, info.obs_version)
        config = load_config(args.config, env_overrides=False) if args.config else None
        sides = {"human": args.human, "zombies": args.zombies}
        factories = {
            "human": lambda s: human_policy(s, args.seed, args.deterministic),
            "zombies": lambda s: zombie_policy(s, args.seed + 1, args.deterministic),
        }
        remote = {k: v for k, v in sides.items() if v and _server_side(v) is None}
        if not remote:
            raise SystemExit("at least one side must be played by this client")
        matches = [args.match] if args.match else [None] * args.episodes
        for k, match_id in enumerate(matches):
            if match_id is None:
                agents = {
                    s: "remote" if s in remote else _server_side(v or "server:scripted")
                    for s, v in sides.items()
                }
                match_id = await client.create_match(
                    config=config,
                    seed=None if args.seed_match is None else args.seed_match + k,
                    human=agents["human"],
                    zombies=agents["zombies"],
                    speed=args.speed,
                    deterministic=args.deterministic,
                    realtime=args.realtime,
                )
                log.info("match %s: %s vs %s", match_id, sides["human"], sides["zombies"])
            results = await asyncio.gather(
                *(
                    client.play(match_id, s, factories[s](spec), agent_name=_name(spec))
                    for s, spec in remote.items()
                )
            )
            for ep in results[0]:
                log.info(
                    "match %s episode %d: %s after %d decisions (replay %s)",
                    match_id,
                    ep.episode,
                    ep.outcome,
                    ep.ticks,
                    ep.replay_id,
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server", default=os.environ.get("ZAI_GRPC_TARGET", "server:50051"))
    p.add_argument("--token", default=os.environ.get("ZAI_AGENT_TOKEN") or None)
    p.add_argument("--tls-root", help="PEM root certificate (TLS, e.g. through Caddy)")
    p.add_argument("--realtime", action="store_true", help="deadlines of one decision")
    p.add_argument("--human", help="checkpoint .pt, scripted, random or server:<agent>")
    p.add_argument("--zombies", help="checkpoint .pt, scripted, random or server:<agent>")
    p.add_argument("--match", help="join this existing match instead of creating one")
    p.add_argument("--config", help="env config YAML (default: the server's base config)")
    p.add_argument("--speed", type=float, default=1.0, help="x0.25-x16, 0 = unthrottled")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--deterministic", action="store_true", help="argmax actions")
    p.add_argument("--seed", type=int, default=0, help="seed of the client-side sampling")
    p.add_argument("--seed-match", type=int, help="seed of the first match (default: random)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
