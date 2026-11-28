"""Live "what the league plays now" match (docs/ARCHITECTURE.md 4.9.4).

    python -m arena.showcase --run-id <league run>

Follows `<ZAI_CHECKPOINT_DIR>/league/<run>/pool.json` and keeps one `auto_restart` server
match of the latest human snapshot against the latest zombie snapshot (both played over gRPC
by this process, so pool snapshots need no ONNX export), on the current stage's env config
with its largest zombie count. When a newer snapshot appears, the match is stopped through
the HTTP API (`ZAI_SERVER_URL`) and recreated.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

from zai_agent.client import Client
from zai_agent.compat import check
from zai_agent.policies import human_policy, zombie_policy
from zai_env.config import deep_merge, load_config
from zai_rl.train import resolve

log = logging.getLogger("arena.showcase")


def latest(pool: dict[str, Any], side: str) -> dict[str, Any]:
    """The newest league snapshot of `side`, else its stage starter."""
    entries = pool["sides"][side]
    league = [e for e in entries if e.get("source") == "league" and e.get("path")]
    if league:
        return max(league, key=lambda e: e["steps"])
    starter = pool["stage"]["starters"][side]
    return next(e for e in entries if e["id"] == starter)


def pick(pool_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """`(human entry, zombie entry, env config)` of the pool's current state."""
    pool = json.loads(pool_path.read_text())
    stage = pool["stage"]
    cfg = load_config(resolve(stage["env_config"]), env_overrides=False)
    cfg = deep_merge(cfg, {"zombie_count": max(stage["zombie_counts"])})
    return latest(pool, "human"), latest(pool, "zombie"), cfg


def stop_match(server_url: str | None, match_id: str) -> None:
    if not server_url:
        return
    req = urllib.request.Request(
        f"{server_url}/api/matches/{match_id}/control",
        data=json.dumps({"stop": True}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with contextlib.suppress(OSError):
        urllib.request.urlopen(req, timeout=10).close()


async def run(args: argparse.Namespace) -> None:
    pool_path = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints"))
    pool_path = pool_path / "league" / args.run_id / "pool.json"
    server_url = os.environ.get("ZAI_SERVER_URL")
    async with Client(args.server, args.token) as client:
        check(await client.info())
        while True:
            human, zombie, cfg = pick(pool_path)
            names = (f"{args.run_id}/{human['id']}", f"{args.run_id}/{zombie['id']}")
            match_id = await client.create_match(
                config=cfg, human="remote", zombies="remote", speed=args.speed, auto_restart=True
            )
            log.info("match %s: %s vs %s", match_id, *names)
            tasks = [
                asyncio.create_task(
                    client.play(match_id, "human", human_policy(human["path"]), names[0])
                ),
                asyncio.create_task(
                    client.play(match_id, "zombies", zombie_policy(zombie["path"]), names[1])
                ),
            ]
            while not any(t.done() for t in tasks):
                await asyncio.sleep(args.poll_s)
                new_h, new_z, _ = pick(pool_path)
                if (new_h["id"], new_z["id"]) != (human["id"], zombie["id"]):
                    log.info("newer snapshots %s / %s: restarting", new_h["id"], new_z["id"])
                    stop_match(server_url, match_id)
                    break
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    log.warning("match %s ended: %s", match_id, r)
            await asyncio.sleep(1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--server", default=os.environ.get("ZAI_GRPC_TARGET", "server:50051"))
    p.add_argument("--token", default=os.environ.get("ZAI_AGENT_TOKEN") or None)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--poll-s", type=float, default=60.0, help="pool.json polling interval")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
