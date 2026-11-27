# zAI

[![ci](https://github.com/alexandreprates/zAI/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandreprates/zAI/actions/workflows/ci.yml)

A humans-vs-zombies survival game built as a multi-agent reinforcement learning environment. Two separate neural models are trained against each other: one controls the single human, the other controls the zombie horde. The goal is emergent tactics (foraging, hiding and door tricks for the human; coordinated search and group hunting for the zombies), not random wandering.

## The game

- **Map:** 500 x 500 m with 10 houses, each with a front and a back door. Houses can only be entered through doors, and closing one door opens the other.
- **Human (1 per match):** sees 100 m inside a 120-degree field of view, walks 1 m/s, must eat a food package every 90 s or starve, and slows down by 20% of the base speed for every package carried. It dies on contact with a zombie and is the only one who can close doors.
- **Zombies (default 20):** they cannot see. They hear noise within 300 m, walk 0.7 m/s and attack by colliding with the human.
- **Noise:** walking is heard within 10 m, picking up food within 100 m and closing a door within 200 m.

The observations are asymmetric: the human perceives by sight, the zombies by sound. Training uses a curriculum of smaller maps (S0: 150 m, 3 houses; warm-up `s0_small`: 75 m) before the full PRD map.

## Architecture

- **`crates/zai-sim`:** a deterministic simulation core in Rust and the only implementation of the game rules.
- **`crates/zai-py`:** PyO3 bindings exposing a batched `VecEnv` (rayon-parallel, writing into preallocated NumPy buffers), so training has no network hop.
- **`python/`:** `zai_env` (config loading, observation layouts, PettingZoo wrappers), `zai_rl` (recurrent PPO for the human, recurrent MAPPO with a shared actor and a centralized critic for the zombies, scripted bots) and `arena` (bot matches, balance sweeps, replay and ONNX export).
- **`crates/zai-server`:** an axum server that runs live matches between bots or trained ONNX models, records replays, stores training metrics in SQLite and serves the viewer.
- **`web/`:** a Vite, TypeScript and PixiJS viewer with a top-down map, a third-person 3D view (three.js), replays and training curves.

The full design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and the requirements are in [`ZAI_PRD.md`](ZAI_PRD.md) (in Portuguese).

## Status

| Milestone | Status |
|---|---|
| M0 to M3: scaffold, simulation core, Python bindings, server, replays and web viewer | done |
| M4: human recurrent PPO vs scripted zombies on S0 | done: ~97% survival vs an 82.8% scripted baseline |
| M5: zombie MAPPO vs the scripted human | done on `s0_small`: ~97% caught vs a 14% scripted baseline |
| Co-training by hand (fine-tunes against frozen opponents) | in progress, see `docs/ARCHITECTURE.md` section 4.7 |
| M6: automated opponent-pool scheduler and arena (Elo, tactic metrics) | planned |
| M7: gRPC agent API and first-person view | planned |

## Getting started

Only Docker with Compose v2 is needed on the host. Every target runs in a container through the `Makefile`.

```sh
make test     # Rust tests, build the Python extension, pytest
make lint     # rustfmt, clippy, ruff, TypeScript type check
make up       # server and viewer on http://localhost:8080
```

To start a bot match, use the viewer or run:

```sh
curl -X POST localhost:8080/api/matches -H 'content-type: application/json' \
  -d '{"speed": 4, "auto_restart": true}'
```

### Training

```sh
make train-human ARGS="--total-steps 2000000 --run-id my-human"
make train-zombie ARGS="--total-steps 2000000 --run-id my-zombies"
make tensorboard   # http://localhost:6006
```

While the server is running, the training curves and the replays recorded during training appear under "Training runs" in the viewer. Checkpoints can be exported as viewer replays (`make export-replays`) or as ONNX models that play live matches (`make export-model`).

Long runs target an NVIDIA P100 server (`make check-gpu`, `make train-human-gpu`, `make train-zombie-gpu`). Setup, port tunneling and HTTPS are described in [`docs/GPU_SERVER.md`](docs/GPU_SERVER.md).

## Repository layout

```
crates/zai-sim     simulation core (rules, map geometry, observations)
crates/zai-py      Python bindings (Env, VecEnv)
crates/zai-server  game server, replays, metrics API, ONNX policies
python/            zai_env, zai_rl (trainers), arena (evaluation, exports)
web/               viewer (top-down, 3D, charts)
configs/           environment stages and training configs
docker/            container images
docs/              architecture, GPU server guide, balance reports
```
