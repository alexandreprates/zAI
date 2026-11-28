# gRPC lockstep throughput (M7.2, 2026-11-28)

One live match, both sides remote and driven by one Python process (`zai_agent.Client`, two
`play` tasks under `asyncio.gather`), randomly initialized CPU PyTorch actors of the trained
sizes (human GRU 256, zombie GRU 128, `torch.set_num_threads(1)`), sampled actions, speed 0
(unthrottled lockstep). Server: the release `zai-server` image; client: the dev container, on
the same host, over the compose network. 5 episodes per case.

| Case | Decisions | Time | Decisions/s | ms per decision |
|---|---|---|---|---|
| `s0_small` (75 m, 3 zombies, 300 s cap) | 1440 | 7.2 s | 201 | 5.0 |
| PRD defaults (500 m, 20 zombies, 120 s cap) | 900 | 5.2 s | 174 | 5.7 |

Measured while the GPU server's CPU was busy with three zombie fine-tunes (diagnostic D3 of
`docs/ARCHITECTURE.md` 4.8.9, 10 simulation threads each), so these are lower bounds; the plan
estimated 500-1000 decisions/s on an idle CPU (4.9.4). Either way a 20 min S2 episode (2400
decisions) takes ~12-15 s, realtime play needs 2-32 decisions/s, and batched Elo cross-play
stays in-process (decision E11): in-process `VecEnv` evaluation runs ~10 k env steps/s.

Reproduce: `python /path/to/grpc_bench.py server:50051` from the dev container (script in the
M7.2 commit message), or time `make play ARGS="--human <ckpt> --zombies <ckpt> --speed 0"`.
