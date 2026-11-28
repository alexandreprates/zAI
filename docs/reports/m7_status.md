# M7 status (2026-11-28, branch `m7`)

Plan: `docs/ARCHITECTURE.md` 4.9 (decisions E1-E13 accepted, all option a).

Done and committed (all `make test` / `make lint` green, incl. vitest):
- M7.0 protocol and codegen (`proto/zai/v1/agent.proto`, `build.rs`, `GetInfo`, Python stubs
  with drift check, `make proto`).
- M7.1 remote sides in lockstep (`remote.rs`, two-phase match loop, `waiting` status, join /
  action timeouts, disconnect and invalid-action failures, remote match cap,
  `ReplayAgent::Remote`), `tests/grpc.rs` incl. byte-for-byte obs parity with a local `Env`.
- M7.2 `zai_agent` client, `make play`, parity test on `s0_small`, S0 and the defaults
  (`zai_agent.parity`, `test_grpc_parity.py`), throughput report `grpc_throughput.md`.
- M7.3 realtime mode, `ZAI_AGENT_TOKEN`, Caddy gRPC TLS site (port 50052), `arena` compose
  service, `arena.showcase`, `make showcase`, `make grpc-smoke`.
- M7.4 first-person view (`seen_zombies` / `seen_food` in snapshots, camera modes, radial fog,
  x-ray, HUD additions, seen outlines in 2D). Not done: front/back door labels inside a house.

Still open for M7.5 (acceptance, 4.9.8):
- Criterion 2: `make play` with real checkpoints from the `arena`/trainer container, both sides
  remote, at x4 and in realtime at x16; check the replay names both agents and verifies.
- `make grpc-smoke` from the container against the server container.
- Criterion 4: a v1 checkpoint (`human-s0-gpu1`) playing over gRPC through the adapter.
- Criterion 5: `arena.showcase` switching snapshots while a league runs.
- Criterion 6: browser check of first person (live, replay with seek, only `seen_*` with x-ray
  off, >= 50 fps at 1080p with `?fps=1`); needs a person at a browser.
- Criterion 7: re-measure lockstep throughput on an idle CPU (the reported ~200 decisions/s was
  measured next to three GPU fine-tunes).
- `CLAUDE.md` commands and the section 9 row; then merge `m6` / `m7` towards `main`.
