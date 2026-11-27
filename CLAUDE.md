# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Milestones M0 (scaffold), M1 (simulation core in `crates/zai-sim`), M2 (Python bindings, `zai_env` wrappers, scripted bots, balance sweep) M3 (`zai-server` with live bot matches, replays, metrics API, and the top-down web viewer in `web/`) and M4 (human recurrent PPO vs scripted zombies on S0, training curves in the viewer) are done. Zombie training (M5) is implemented; on S0 the learned zombies stayed below the scripted baseline, so they were trained on the smaller warm-up stage `s0_small` (`zombie-s0small-gpu1`: ~97% caught against the scripted human vs a 14% scripted-zombie baseline; plan and results in `docs/ARCHITECTURE.md` section 4.6). Co-training is currently done by hand, fine-tuning each side against the other's frozen policy (`opponent: policy` + `init_checkpoint`, configs `configs/train/*_ft_gpu.yaml`, results in section 4.7); M6 (league trainer with groups A/B/C and a PFSP opponent pool, arena with Elo and tactic metrics, observation layout v2, curriculum s0_small -> S0 -> S1) is planned in `docs/ARCHITECTURE.md` section 4.8, with the decisions of 4.8.11 accepted (2026-11-27), and is being implemented in the sub-milestone order M6.0-M6.5; `zai_agent` / gRPC and the first-person view (M7) are still to come. The M4 GPU run (`human-s0-gpu1`, 200 M steps in ~2.4 h on the P100) reached ~97% survival on S0 against an 82.8% scripted baseline; measured throughput is ~25 k env steps/s, about 3x below the planning estimate (see `docs/ARCHITECTURE.md` sections 4.1 and 4.4). The M2 balance sweep found that scripted zombies almost never catch a careful human, even after raising zombie speed to 0.7 m/s; catchability is to be re-evaluated with learned zombies in M5 (risk 1 in `docs/ARCHITECTURE.md`). `ZAI_PRD.md` (written in Portuguese) is the source of truth for requirements. `mcp/` contains only an empty `package-lock.json`. The proposed implementation architecture (Rust simulation core shared by an in-process Python training env and the game server, PPO/MAPPO trainers, P100 GPU target) and the decisions that resolve PRD ambiguities are in `docs/ARCHITECTURE.md`; follow it when implementing. The `ml-architect` subagent (`.claude/agents/ml-architect.md`) owns that document.

## Commands

Nothing is installed on the host; every target runs in Docker (Compose v2) through the `Makefile`. Targets prefixed with `ci-` are the ones executed inside the `dev` container.

- `make test`: Rust tests (`cargo test --workspace --exclude zai-py`), builds the `zai_sim` extension with `maturin develop`, then `pytest python`.
- `make lint` / `make fmt`: rustfmt, clippy (`-D warnings`) and ruff.
- `make shell`: shell in the `dev` container (repo mounted at `/workspace`; Cargo target dir and uv cache live in the `dev-cache` volume at `/cache`).
- Single Rust test: `docker compose run --rm dev cargo test -p zai-sim <name>`.
- Single Python test: `docker compose run --rm dev bash -c "make ci-py-develop && pytest python -k <name>"` (each `run` starts from the image's venv, so the extension must be rebuilt in the same command).
- Random-agent episodes (outcome stats, steps/s): `docker compose run --rm -T dev cargo run --release -p zai-sim --bin zai-random -- --episodes 50`.
- Simulation benchmark (criterion): `docker compose run --rm -T dev cargo bench -p zai-sim --bench step`.
- Balance sweep (writes the `--out` path `.md` / `.json`; reports live in `docs/reports/`): `docker compose run --rm -T dev bash -c "make ci-py-develop && cd python && python -m arena.balance_sweep --out ../docs/reports/balance_sweep"`.
- `make up`: builds and runs the server with the viewer on http://localhost:8080 (data in the `data` volume). Start a bot match from the viewer or with `curl -X POST localhost:8080/api/matches -H 'content-type: application/json' -d '{"speed": 4, "auto_restart": true}'`.
- `make web-dev`: Vite dev server on http://localhost:5173 proxying to a server on port 8080 (e.g. one started with `make up`). `make web-install` installs `web/` dependencies into the `web-node-modules` volume; `make lint` also runs `npm run typecheck`.
- Server tests only: `docker compose run --rm -T dev cargo test -p zai-server` (`tests/e2e.rs` starts a real server and drives the WebSockets).
- `make train-human ARGS="--total-steps N --run-id ID"`: human PPO (config `configs/train/human_s0.yaml`) in the trainer container; with the server running (`make up` in another terminal, or `docker compose up -d server`) the curves appear under "Training runs" in the viewer. Checkpoints go to the `checkpoints` volume (`/checkpoints/human/<run_id>/`), TensorBoard logs to `runs` (`make tensorboard`, port 6006). Locally (dev container): `cd python && python -m zai_rl.train --total-steps 65536` with `ZAI_CHECKPOINT_DIR` / `ZAI_RUNS_DIR` / `ZAI_SERVER_URL` as needed.
- `make train-zombie ARGS="--total-steps N --run-id ID"`: zombie MAPPO (M5, `python -m zai_rl.train_zombie`, config `configs/train/zombie_s0.yaml`) against the scripted human; same outputs as the human trainer with model `zombie` and checkpoints under `/checkpoints/zombie/<run_id>/`. It measures the scripted-zombie catch rate at startup and logs `eval/caught` with Wilson intervals per zombie count. GPU: `make train-zombie-gpu` (`zombie_s0_gpu.yaml`, also evaluates against the frozen M4 human).
- Fine-tunes (either trainer): `opponent: policy` with `zombie_checkpoint` (human trainer) or `human_checkpoint` (zombie trainer) trains against the other side's frozen policy; `init_checkpoint` starts from this side's trained weights with a fresh optimizer and counters (unlike `--resume`). `eval_scripted/*` tracks the scripted opponent to catch forgetting. Example: `make train-zombie-gpu ARGS="--config /app/configs/train/zombie_s0small_crowd_ft_gpu.yaml --run-id ID"`.
- `make export-replays ARGS="--checkpoint /checkpoints/human/<run_id>/latest.pt --episodes 10"` (and/or `--zombie-checkpoint /checkpoints/zombie/<run_id>/latest.pt`): plays checkpoints against bots for the missing side (S0 config by default) in the trainer container and writes the episodes as replays into the `data` volume; they show up under "Replays" in the viewer with `policy/scripted`. On the GPU server use `make export-replays-gpu` (same ARGS); the server must run a build that knows `"human": "policy"` or it skips those files.
- `make export-model ARGS="--checkpoint /checkpoints/<side>/<run_id>/latest.pt"` (GPU server: `make export-model-gpu`): exports checkpoints as ONNX models (`arena.export_model`) into the `data` volume's `models/`; trainers also export `<run_id>` on every checkpoint when `ZAI_DATA_DIR` is set. They show up as "trained" options in the viewer's New match form.
- HTTPS (self-signed, Caddy local CA): `ZAI_TLS_SITES="host:8080, ..." docker compose -f compose.yaml -f compose.tls.yaml up -d server proxy`; Caddy takes port 8080 and proxies to `server:8080` (details in `docs/GPU_SERVER.md` section 6).
- GPU server (P100): `make check-gpu`, then `make train-human-gpu ARGS="--run-id ID"` (config `configs/train/human_s0_gpu.yaml`); full setup and port tunneling in `docs/GPU_SERVER.md`.
- `make build`: builds the `server` and CPU `trainer` images. `docker compose up server` serves `GET /healthz` on port 8080.
- `make check-gpu`: on the P100 server only; builds the trainer with `TORCH_INDEX=cu126` (`compose.gpu.yaml`) and fails unless the PyTorch build has `sm_60` kernels and sees the GPU.

The user's shell exports `DOCKER_DEFAULT_PLATFORM=linux/amd64`; the `Makefile` unexports it so images are native arm64 on the Apple Silicon pilot machine. When calling `docker compose` directly (as in the single-test commands above), prefix with `env -u DOCKER_DEFAULT_PLATFORM`, otherwise images are built for emulated amd64.

## Simulation core (`crates/zai-sim`)

- `Env` (`env.rs`) is one deterministic match: `step(HumanAction, &[u8])` runs one decision (`action_repeat` = 5 physics ticks of 0.1 s). Order inside a step: human interaction (pickup / close door, masked actions are no-ops), then ticks (movement with wall sliding, zombie soft push, contact check, hunger, food respawn), then the walking noise, noise routing to zombies, truncation, perception update, rewards.
- `map.rs` holds houses (6 wall segments + 2 door segments each) and the geometric queries (line of sight, ray casts, clearance) culled per house. `obs.rs` writes the fixed-size observation vectors whose layouts are specified in `docs/ARCHITECTURE.md` section 3.6; keep both in sync.
- Rule tests live in `src/tests/` (crate-internal so they can arrange exact states via `pub(crate)` fields): `rules.rs` has one test per PRD rule, `props.rs` has proptest invariants (no wall penetration or tunneling, determinism by `state_hash`).
- `Config` defaults are the PRD values (curriculum stage S2); `ZAI_ZOMBIE_COUNT` / `ZAI_FOOD_COUNT` apply through `Config::with_env_overrides`.

## Python side (`crates/zai-py`, `python/`)

- `zai_sim` (PyO3 module from `crates/zai-py`): `Env` (single match) and `VecEnv` (rayon-parallel, GIL released, auto-reset, writes into preallocated NumPy arrays that are reused and overwritten on every call; `info["final_*"]` holds the last observation of finished episodes). Per-episode seeds come from `episode_seed(base, env_index, episode)`, which the parity test relies on.
- Configs cross into Rust as JSON; `zai_env.config.load_config` merges defaults <- YAML <- overrides <- env vars and Rust validates (unknown keys are errors). Do not duplicate the schema in Python.
- Never hardcode observation indices in Python: use `zai_env.layout` (`HUMAN`, `ZOMBIE`, `STATE`), built from the layout tables in `crates/zai-sim/src/obs.rs`.
- `zai_rl.bots` holds the vectorized scripted/random policies; `arena.matches.play` runs batched bot-vs-bot matches (first episode per env only).
- `make test` builds the extension with `maturin develop --release`; Python tests include a VecEnv-vs-Env parity test and the PettingZoo API tests.

## Server and viewer (`crates/zai-server`, `web/`)

- `zai-server` (axum): `matches.rs` runs each live match as a tokio task (bots from `zai_sim::bots`), broadcasting MessagePack frames built in `frames.rs`; viewers joining mid-episode get the latest `init` + `snapshot` first. Either side of a match can be a trained model (`"human": {"policy": "<model id>"}`): `policy.rs` loads `<ZAI_DATA_DIR>/models/<id>.onnx` (exported by `python/arena/export_model.py`, metadata in `<id>.json`, listed by `GET /api/models`) with `tract` and samples masked actions from the logits; the GRU is exported as an explicit cell (`n + z * (h - n)`, since tract 0.21 negates `1 - z`), and `tests/policy_parity.rs` checks tract against PyTorch outputs in `tests/fixtures` (regenerate with `python -m arena.export_model --parity-fixtures crates/zai-server/tests/fixtures` after changing an actor). Finished episodes become replays (`replay.rs`, action log + final `state_hash`, re-simulated for playback/seek); `python/arena/export_replays.py` writes the same format for trained policies (`ReplayAgent::Policy`). Metrics live in SQLite (`metrics.rs`). Env: `ZAI_HTTP_ADDR`, `ZAI_DATA_DIR`, `ZAI_WEB_DIR`, `RUST_LOG`.
- Frame and API shapes are mirrored by hand in `web/src/types.ts`; change both together.
- Training runs in the viewer: curves come from the metrics API (`charts.ts` groups `baseline_*`, `*_lo` / `*_hi` and per-count `*_zN` keys onto shared charts). Trainers with `replay_every > 0` and `ZAI_DATA_DIR` set (the trainer container mounts the `data` volume) record `replay_episodes` episodes of the current policy at iteration 1 and every `replay_every` iterations, on fixed seeds, as replays with `match_id = run_id` (`arena.export_replays.record_training_replays`); the viewer lists them as chips above the run's charts instead of in the Replays sidebar.
- `web/` (Vite + TypeScript + PixiJS 8): `topdown.ts` renders the map (world y-up flipped to screen y-down) and interpolates between per-decision snapshots; `scene3d.ts` is a third-person 3D view of the same frames (three.js, loaded on demand by the 2D/3D toggle; world y maps to three's -z; follow or free orbit camera); `main.ts` wires the sidebar, toolbar and HUD. The production build is served by `zai-server` (`ZAI_WEB_DIR`), built in the server image's Node stage.

## Training (`python/zai_rl`)

- M5 zombies: `envs.ZombieTrainingEnv` lays the zombies of all groups on a flat agent axis `[A]` with a static `agent_env [A]` index (per-env arrays such as `state` stay `[E]` and are gathered); `algos/mappo_zombie.ZombieMAPPO` is recurrent MAPPO with a shared `ZombieActor` and a centralized `ZombieCritic` on `concat(state, own obs)`; per-zombie rewards come from Rust (`VecEnv` `zombie_reward [E, N]`). Shared pieces (GAE, `RunningMeanStd`, BPTT reshape, checkpoints) live in `algos/common.py`, and `models/encoders.run_gru_segments` is the GRU-with-resets used by both actors.
- `envs.HumanTrainingEnv` concatenates one `ZaiVecEnv` per zombie count; zombie actions come from a fixed opponent policy. `algos/ppo_human.HumanPPO` does rollout (`collect`), GAE with truncation bootstrap folded into the last reward, and the PPO update over BPTT sequences (`HumanActor.sequence`, which must stay equivalent to repeated `step` calls; a test checks this).
- `policy.RecurrentHumanPolicy` wraps an actor or checkpoint with the bot interface; `arena.matches.play` calls a policy's `reset(done)` so recurrent state is cleared between episodes.
- The dev image has CPU PyTorch for tests; the trainer image installs PyTorch from the `TORCH_INDEX` build arg (CPU or cu126) plus TensorBoard.

`zai-py` is excluded from `cargo test` because its test binary would need libpython; the bindings are tested from Python.

## What is being built

A humans-vs-zombies survival game used as a multi-agent reinforcement learning environment, training two separate neural models: one controlling the single human, one controlling the zombies. The goal is emergent tactics (survival for the human, coordinated group hunting for zombies), not random wandering.

### Planned components (per PRD)

- **Game server**: authoritative simulation — map layout, food distribution, noise events, door state, collisions, and pushing environment observations to agents.
- **Web viewer** (served by the game server): top-down full-map view and a first-person 3D view from the human's perspective, plus training/evaluation metrics for both models.
- **Two model trainers** (human, zombie): communicate with the game server to receive observations and send movement actions.
- **Infrastructure**: everything runs in containers (nothing installed on the host); trainers must use GPU.

### Game rules that drive the simulation design

- Map: 500 x 500 m; 10 houses spread uniformly, each with a front and back door. Houses are only enterable through doors. Closing one door automatically opens the other.
- Food packages scattered on the map; count is configurable via variable.
- Human (exactly 1 per match): vision radius 100 m with a 120-degree frontal field of view around the heading (the direction it last moved; it has no separate look action); walks 1 m/s; must eat 1 food package every 90 s or dies; carries unlimited food but each carried package reduces speed by 20% of the base speed (linear, not compounding: `speed = 1 * max(0, 1 - 0.2 * carried)`, so carrying 5+ packages means the human cannot move); dies on contact with a zombie; can close doors (zombies cannot).
- Zombies: count set via environment variable (default 20; food default 120); hearing radius 300 m; walk 0.7 m/s (raised from the original 0.5 m/s after the M2 balance sweep); attack by colliding with the human.
- Noise events (what zombies perceive): walking = 10 m radius, picking up food = 100 m radius, closing a door = 200 m radius.

Observations are asymmetric: the human perceives by sight (100 m radius), zombies by sound (300 m radius) — the server must route noise events to zombies within range rather than exposing full state.
