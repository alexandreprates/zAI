# ZAI Architecture

- **Status:** Proposed
- **Date:** 2026-11-24
- **Source of truth:** `ZAI_PRD.md` (summarized in `CLAUDE.md`)
- **Resolved by the user before this proposal:** human vision radius is 100 m; the food speed penalty is linear on base speed (`speed = 1.0 * max(0, 1 - 0.2 * carried)`, so 5+ packages means zero speed).
- **User decisions on the open questions (2026-11-24):**
  - Q1: accepted. Training runs the Rust core in-process; the server's gRPC agent API is used for evaluation and live matches.
  - Q2: option (b). Zombies perceive noise, touch, the static map layout, 10 m proximity rays and up to 8 peers within 50 m. No learned communication (kept only as an ablation).
  - Q3: food respawns by default (see section 3.2).
  - Q4: zombie count and food count defaults may be tuned by the M2 balance sweep. Speeds and noise radii stay as in the PRD.
  - Q5: pilot runs locally (macOS, CPU-only containers); full training later moves to a Linux server with an NVIDIA Tesla P100 (see section 6).
  - Q6: surviving to the 20 min cap counts as a human win in evaluation.
  - P100 server: 24 CPU cores, 64 GB RAM.
  - Q7 (after the M2 balance sweep): zombie speed raised from 0.5 to 0.7 m/s in the PRD; then human vision narrowed to a 120-degree frontal field of view, with defaults of 20 zombies and 120 food (scripted bots: ~12% caught, ~48% starved, ~39% survived).

---

## 1. Summary

A single deterministic simulation core written in **Rust** (`zai-sim`) is the only implementation of the game rules. It is used in two ways:

1. **Training (throughput path):** compiled into a Python extension (PyO3 + maturin) that exposes a *batched vector environment*: one `step()` call advances 1024+ environments in parallel on all CPU cores (rayon), writing observations into preallocated NumPy buffers with zero copies. A single **trainer** container (PyTorch, CUDA) hosts two independent learners: the **human learner** (recurrent PPO) and the **zombie learner** (recurrent MAPPO with parameter sharing and a centralized critic). No network hop exists in the training loop.
2. **Observation / evaluation (PRD path):** the **game server** (Rust, axum + tonic) embeds the same core, runs live matches, exposes a **gRPC agent API** through which policy processes receive observations and send actions (satisfying "models communicate with the game server"), streams state to the **web viewer** over WebSocket, and stores/serves training metrics. An **arena** container loads checkpoints, plays them against each other through the server and in-process, and computes Elo and tactic metrics.

Because both paths call the same Rust functions for physics *and* observation encoding, a policy trained in-process behaves identically when it drives a live match over gRPC.

```mermaid
flowchart LR
  subgraph TR["trainer container (GPU)"]
    VE["zai_sim.VecEnv<br/>Rust core via PyO3<br/>E=1024 envs, rayon"]
    HL["Human learner<br/>PPO + GRU"]
    ZL["Zombie learner<br/>MAPPO, shared params"]
    POOL[("snapshot pool<br/>/checkpoints")]
    VE <--> HL
    VE <--> ZL
    HL --> POOL
    ZL --> POOL
  end

  subgraph SV["server container"]
    CORE["zai-sim core<br/>live matches"]
    GRPC["gRPC agent API"]
    WS["WebSocket viewer stream"]
    MET["metrics API + SQLite"]
    WEB["static web app"]
    CORE --- GRPC
    CORE --- WS
  end

  subgraph AR["arena container"]
    EVAL["evaluator / Elo /<br/>tactic metrics"]
    PCL["policy clients<br/>(human, zombies)"]
  end

  POOL --> EVAL
  POOL --> PCL
  PCL <-->|"bidi stream"| GRPC
  HL -->|"POST /api/metrics"| MET
  ZL -->|"POST /api/metrics"| MET
  EVAL -->|"POST /api/metrics"| MET
  BR["Browser:<br/>top-down, 3D, charts"] <--> WS
  BR --> MET
  BR --> WEB
```

---

## 2. Components and responsibilities

| Component | Language / libs | Responsibility |
|---|---|---|
| `zai-sim` (crate) | Rust, `glam` (vectors), `rand_chacha` (seeded RNG) | Pure game logic, no I/O: map generation, movement, collisions, doors, food, hunger, noise routing, line-of-sight, observation/state encoding, rewards, termination. Deterministic given `(config, seed, actions)`. |
| `zai-py` (crate) | Rust, `pyo3`, `numpy` (rust-numpy), `rayon`, built with `maturin` | Python module `zai_sim`: `VecEnv` (batched, GIL released, parallel over envs, zero-copy buffers) and `Env` (single env, for tests/debug). |
| `zai-server` (crate) | Rust, `tokio`, `axum` (HTTP/WebSocket), `tonic` (gRPC), `rusqlite`, `rmp-serde` (MessagePack) | Match manager (create/run/pause, real-time with speed multiplier or lockstep), gRPC agent API, viewer WebSocket stream, replay recording, metrics ingest/query, serves the built web app. |
| `zai_env` (Python pkg) | `gymnasium`, `pettingzoo` | Thin wrappers: PettingZoo `ParallelEnv` over `zai_sim.Env` (API compliance tests, debugging, third-party libs) and a typed wrapper over `VecEnv` used by the trainer. Config loading (YAML merge; validation is delegated to Rust) and named access to observation fields using the layout exported by Rust (`zai_sim.obs_layout()`). |
| `zai_rl` (Python pkg) | `torch` 2.x | Models, recurrent PPO / MAPPO, rollout storage, co-training scheduler, snapshot pool, curriculum, scripted bots, metrics logging (TensorBoard + server push). CleanRL-style single-file-per-algorithm code, no framework. |
| `zai_agent` (Python pkg) | `grpcio`, `torch` (CPU ok) | Remote policy client: connects to the server, loads a checkpoint, batches all agents of its team per decision tick. |
| `arena` (Python entrypoint) | `zai_rl`, `zai_agent` | Cross-play evaluation, Elo, tactic metrics, showcase matches on the server for the viewer. |
| `web` | TypeScript, Vite, PixiJS (top-down), Three.js (3D), uPlot (charts), `@msgpack/msgpack` | Viewer UI. Built in a Node stage and served as static files by `zai-server`. |

### Repository layout

```
zai/
  Cargo.toml                 # Rust workspace
  crates/
    zai-sim/                 # core simulation (no I/O)
      src/{config,map,physics,doors,food,noise,vision,obs,reward,env}.rs
      benches/step.rs        # criterion benchmark
    zai-py/                  # PyO3 bindings -> python module `zai_sim`
    zai-server/              # axum + tonic server
  proto/
    zai/v1/agent.proto       # gRPC agent API
  python/
    pyproject.toml           # uv-managed; packages below
    zai_env/
    zai_rl/
      algos/{ppo_human.py, mappo_zombie.py}
      models/{encoders.py, human.py, zombie.py, critics.py}
      league/{pool.py, elo.py, scheduler.py}
      curriculum.py
      bots/{scripted_human.py, scripted_zombie.py}
      metrics.py
      train.py               # entrypoint
    zai_agent/
    arena/
    tests/
  web/
    src/{topdown/, fps3d/, charts/, net/}
  configs/
    env/default.yaml         # PRD values
    curriculum/default.yaml
    train/default.yaml
  docker/
    server.Dockerfile
    trainer.Dockerfile
    arena.Dockerfile
    dev.Dockerfile           # toolchain image for builds/tests
  compose.yaml
  compose.gpu.yaml           # GPU override (NVIDIA)
  Makefile                   # wraps `docker compose run ...`
  docs/ARCHITECTURE.md
```

Configuration: YAML files (`configs/env/`) merged in Python and validated by the Rust `serde` schema (`deny_unknown_fields`), which is the single source of truth; Python passes configs to Rust as JSON. Environment-variable overrides are applied by Rust. The PRD-mandated variables are `ZAI_ZOMBIE_COUNT` (default 20) and `ZAI_FOOD_COUNT` (default 120).

---

## 3. Environment specification

### 3.1 Time

| Parameter | Value | Reason |
|---|---|---|
| Physics tick | 10 Hz (`dt = 0.1 s`) | Human moves 0.1 m/tick, zombie 0.07 m/tick; relative closing speed 0.17 m/tick is far below the 0.6 m contact distance, so no tunneling. |
| Decision rate | 2 Hz (action repeat 5 ticks) | One decision per 0.5 s keeps horizons tractable: the 90 s hunger cycle is 180 decisions. |
| Episode cap | 20 min sim time = 2400 decisions (configurable) | Truncation, not termination: value is bootstrapped. |
| Discount | `gamma = 0.995` (effective horizon ~200 decisions = 100 s) | Covers one hunger cycle. |

### 3.2 World geometry

- Map: 500 x 500 m, solid boundary (entities are clamped).
- Entities are circles of radius 0.3 m. Human-zombie contact (`distance <= 0.6 m` with line of sight, so touching through a wall does not count) kills the human. Zombie-zombie overlap is resolved by a soft push (no blocking deadlocks).
- Houses: 10, each a 12 x 8 m rectangle with wall segments; a 1.5 m door gap at the center of the front wall and of the back wall (opposite walls). Placement: stratified 5 x 2 grid (cells 100 x 250 m), house center jittered uniformly within +/-30 m of the cell center, orientation in {0, 90, 180, 270} degrees, regenerated per episode from the seed. This satisfies "uniform distribution" while guaranteeing spacing.
- Doors: a door is a wall segment that is solid when closed. Initial state: both doors open. The human can close an open door when within 2.0 m of it; that door closes and the *other* door of the same house opens (PRD rule). A door cannot be closed while any entity overlaps the doorway. Doors cannot be opened directly. Consequence: after the first close, every house has exactly one open door, so houses are chokepoints, never sealed refuges.
- Food: `ZAI_FOOD_COUNT` packages placed uniformly at random (rejection-sampled outside walls, at least 5% placed inside houses). Respawn is on by default: each package picked up is replaced by a new one at a uniformly random valid position after `food_respawn_interval_s` (default 60 s; set `null` to disable), keeping the map population at `ZAI_FOOD_COUNT`. The interval is one of the M2 balance sweep parameters.
- Entity-entity and entity-food queries are brute force (at most 64 zombies and 128 food, cheaper than maintaining a spatial hash); walls are culled per house by its bounding circle.
- Grid for other house counts (curriculum): the `cols x rows` grid with the fewest empty cells, then the most square (10 -> 5 x 2, 3 -> 3 x 1, 6 -> 3 x 2); jitter is reduced if a cell is too small.

### 3.3 Human rules

- Spawn: random position at least 150 m from every zombie; inventory 0; hunger timer 90 s.
- Speed: `1.0 * max(0, 1 - 0.2 * carried)` m/s.
- Hunger: timer decreases continuously. When it reaches 0: if `carried >= 1`, one package is consumed automatically (`carried -= 1`, timer reset to 90 s); otherwise the human dies of starvation.
- Pickup is an explicit action (food within 1.5 m), which makes the 100 m pickup noise a choice.
- No attack, no drop action (see section 7).

### 3.4 Zombie rules

- Count `ZAI_ZOMBIE_COUNT`; during training sampled per environment group from a range (section 4.4) so the policy generalizes across counts.
- Speed 0.7 m/s (PRD originally 0.5 m/s, raised by the user after the M2 balance sweep), cannot interact with doors, blocked by walls and closed doors.
- Spawn: uniform random, at least 150 m from the human.
- Zombies are silent (they emit no noise; the human has no hearing).

### 3.5 Noise model

| Event | Radius | Emission |
|---|---|---|
| Human walking | 10 m | Once per decision step in which the human moved > 0.05 m, at the end position. |
| Food pickup | 100 m | On each successful pickup. |
| Door close | 200 m | On each successful close, at the door position. |

A zombie hears an event iff `distance(zombie, source) <= min(event_radius, hearing_radius=300)`. Walls do not attenuate. Source position is exact by default; `noise_localization_sigma` (fraction of distance, Gaussian) is available as a curriculum knob.

### 3.6 Observation spaces

All coordinates are world-frame deltas normalized by the stated range. Variable-size sets are fixed-size padded slots sorted by distance with a validity mask. Encoding lives in Rust (`obs.rs`) and is shared by trainer and server.

The tables below are layout **v1** (current code). Where they say `/500`, the code divides by `map_size_m` (500 only at S2), and the zombie `episode time fraction` is `t / episode_cap_s`. Layout **v2** (constant normalization, house `valid` flags, wall distances; human 608, zombie 276, state 1144) is planned for M6 in section 4.8.2 and replaces these tables when implemented.

**Human actor observation** (flat `float32[584]`, plus action mask):

| Block | Shape | Features |
|---|---|---|
| Self | 16 | `x/500, y/500`, heading sin/cos, velocity `vx, vy` (m/s), `carried/5`, one-hot carried bucket {0,1,2,3,4,5+} (6), `hunger_timer/90`, inside-house flag, time since last own noise / 30 s |
| Visible zombies | 32 x 6 | `dx/100, dy/100, dist/100, vx/0.7, vy/0.7, valid` (within 100 m and line-of-sight) |
| Visible food | 16 x 4 | `dx/100, dy/100, dist/100, valid` |
| Houses (all 10) | 10 x 12 | `dx/500, dy/500`, rotation sin/cos, front door `dx, dy` /500, back door `dx, dy` /500, front closed, back closed (last observed state), door state currently visible, human inside this house |
| Ray cast | 64 x 3 | 64 rays over 360 degrees, range 100 m: `hit_dist/100`, hit-is-wall, hit-is-closed-door |

Vision is a 120-degree frontal cone around the heading (`human.vision_fov_deg`, adopted after the M2 balance sweep; 360 restores omnidirectional vision): with a field of view below 360 degrees, zombies, food and doors are only seen inside the cone around the heading, the 64 rays are spread across that cone instead of 360 degrees, and the critic's `sees_human` flag uses the same test), radius 100 m, occluded by walls and closed doors (LOS segment test). The heading is the direction of the last movement action; the human has no separate look action, so it turns its view by moving.

**Zombie actor observation** (per zombie, flat `float32[252]`):

| Block | Shape | Features |
|---|---|---|
| Self | 12 | `x/500, y/500`, heading sin/cos, `vx, vy`, inside-house flag, blocked-last-tick flag, time since last heard event / 30 s, `num_zombies/64`, episode time fraction, bias 1.0 |
| Heard noise buffer | 8 x 9 | FIFO of last 8 heard events (max age 30 s): `dx/300, dy/300, dist/300`, type one-hot {walk, pickup, door} (3), `age/30`, `fresh` (heard during the last decision step), `valid`. Consecutive walk events overwrite the newest walk slot if it is < 2 s old, so walking does not flush door/pickup events. |
| Peer zombies | 8 x 5 | 8 nearest zombies within 50 m: `dx/50, dy/50, dist/50, same-house flag, valid` |
| Houses (all 10, static map) | 10 x 8 | `dx/500, dy/500`, rotation sin/cos, front door `dx, dy`, back door `dx, dy` (door *state* not observed) |
| Proximity rays | 16 x 3 | 16 rays, range 10 m: `hit_dist/10`, hit-is-wall, hit-is-closed-door (local navigation sense, "touch") |

Zombies never observe the human directly; they only learn where the human is through noise.

**Global state for critics** (privileged, training only, `float32[1144]` padded): human (16: pos, velocity, carried, hunger, inside house id), zombies 64 x 8 (pos, velocity, inside-house, valid, distance to human, LOS-to-human flag), food 128 x 4 (pos, valid, inside-house), houses 10 x 10 (pos, rotation, door positions, door states), time features 4. The zombie critic additionally receives the evaluated agent's own actor observation (agent-specific global state, as in MAPPO).

### 3.7 Action spaces

Discrete, world-frame (consistent with world-frame observations; episodes are augmented with random 90 degree map rotations/flips).

| Agent | Space | Meaning |
|---|---|---|
| Human | `MultiDiscrete([17, 3])` | Head 1: 16 movement directions (22.5 degree steps) or stay. Head 2: `none`, `pickup`, `close_door`. Invalid choices are masked (`pickup` without food in 1.5 m; `close_door` without an open closable door in 2 m). Interaction executes before movement in the step. |
| Zombie | `Discrete(17)` | 16 directions or stay. |

Movement always uses the full available speed (the PRD defines a single walking speed). Wall contact uses slide resolution (the blocked component of the displacement is removed).

### 3.8 Rewards

Human (per decision step):
- `+0.005` per decision survived (= 0.01 per second).
- `-1.0` on death (caught or starved).
- Curriculum stages S0-S1 only: potential-based food shaping `F = gamma * Phi(s') - Phi(s)` with `Phi = -0.5 * [hunger_timer < 30 s] * min(dist_to_visible_food, 100)/100`, annealed to 0 before S2. Potential-based, so it does not change the optimal policy. **Replaced in M4:** the first pilot stayed at 0% survival for 650 k steps, and the formula had two flaws: with a negative potential and `Phi(terminal) = 0`, dying paid a bonus of up to `+scale` (cancelling most of the -1 death penalty), and picking up the nearest food lowered the potential (the target disappears). The implemented potential is `Phi = scale * (min(hunger + 90 * carried, 180) / 180 + 0.25 * proximity)`, `proximity = 1 - dist_to_nearest_visible_food / 100` (0 when none is visible): non-negative, so termination never pays; a pickup is worth about `+0.25 * scale` at the moment of pickup (reserve +0.5 minus the lost proximity of at most 0.25); eating from the inventory leaves it unchanged. S0 uses scale 1.0, always on (`human_food_shaping`, `human_food_shaping_reserve_s` in the config). The zombie distance potential has the same negative-potential issue (a starvation ending pays zombies up to `+scale * dist / map`); to be revisited in M5.

Zombies (shared team reward, identical for all zombies):
- `+1.0` to every zombie when any zombie catches the human.
- `-0.002` per decision (pressure to hunt rather than idle).
- Starvation of the human gives zombies nothing (their PRD objective is "find and attack", so starvation is not their win).
- Curriculum stages S0-S1 only: potential-based shaping on `Phi = -min_zombie_distance_to_human / 500` (privileged, computed in the env, never in the actor input), scale 0.5, annealed to 0 before S2. **Replaced in M5** by a non-negative per-zombie potential (section 4.6): `Phi_i = c * (1 - min(d_i, 100) / 100)`, `c` on a catch and 0 on starvation; S0 also sets the step penalty to 0.

Reward-hacking review:
- **Human hoarding**: self-limiting (2 packages already make the human slower than zombies: 0.6 vs 0.7 m/s; 5 freeze it); the pickup action is explicit, so there is no automatic trap. Tracked via the "carried distribution" metric.
- **Human freezing in silence**: legitimate "noise discipline" tactic, bounded by starvation (at most `(carried + 1) * 90 s`). Not a hack, but it can create a degenerate equilibrium if zombies never hear anything (section 8).
- **Zombies camping food/houses**: legitimate ambush tactic. The per-step penalty is small enough not to forbid it.
- **Shaping exploits** (e.g. zombies orbiting at constant distance): prevented by using potential-based shaping only and removing it entirely in S2.
- **Truncation bias**: truncated episodes bootstrap `V(s_T)`; treating truncation as death would teach the human that minute 20 is fatal.

### 3.9 Episode termination

| Condition | Type | Human | Zombies |
|---|---|---|---|
| Human contact with a zombie | terminal | -1.0 | +1.0 each |
| Hunger timer 0 with no food | terminal | -1.0 | 0 |
| 20 min sim time | truncation | bootstrap | bootstrap |

All agents end together. `VecEnv` auto-resets finished environments and returns the final observation in `info` for bootstrapping.

---

## 4. Training approach

### 4.1 Algorithms

**Human: recurrent PPO with an asymmetric (privileged) critic.**
- Actor: per-type entity encoders (MLP 64 per zombie/food/house slot, masked mean + max pooling), circular 1D conv over the 64 rays (2 layers, 32 channels), concat with self features, MLP 256, **GRU 256**, two categorical heads with masking. About 0.45 M parameters.
- Critic: separate network on the global state (entity self-attention, 2 layers, 64 dims, 4 heads, then MLP 256). Not shared with the actor, so privileged information cannot leak into the policy.

**Zombies: MAPPO (recurrent, parameter sharing, centralized critic, decentralized execution).**
- One actor network shared by all zombies: encoders as above, MLP 128, **GRU 128**, 17-way categorical head. About 0.15 M parameters. No agent ID input (zombies are homogeneous; symmetry breaking comes from different observations and sampling).
- Centralized critic: attention over global state + the agent's own observation embedding, outputs a per-agent value.
- No explicit communication channel by default. Coordination emerges from the shared policy, the centralized critic, peer observation within 50 m and the common signal (all zombies in range hear the same event). An optional learned 4-dim broadcast message to zombies within 50 m is planned as an ablation (section 8).

Why this pairing: the human is a single agent with rich vision, so plain PPO with memory is the right tool. Zombies are many homogeneous agents with a team reward, where MAPPO with shared parameters and a centralized critic is the strongest simple baseline (Yu et al. 2022) and scales to variable counts. Both are on-policy, so they share one rollout loop.

**Hyperparameters (starting point):** Adam lr `3e-4` (linear decay), clip 0.2, `gamma 0.995`, GAE `lambda 0.95`, entropy `0.01 -> 0.001`, 4 epochs (zombies 3), 8 minibatches, grad clip 0.5, value normalization (running mean/std), rollout 128 decisions per env, BPTT chunk 64 with stored hidden states at chunk start. Batch at E=1024: 131 k human transitions and ~1.3 M zombie transitions per iteration (10 zombies).

**Library choice:** custom PyTorch implementation (CleanRL-style) instead of RLlib/TorchRL. The co-training scheduler, opponent pools, mixed learning/frozen policies per env group and the batched Rust env do not fit framework abstractions cleanly, and both algorithms together are ~1.5 k lines. Alternatives considered: RLlib multi-agent (supports PettingZoo and policy mapping, but heavy and slow to iterate), TorchRL MAPPO, Sample Factory (fast, but weak multi-team support). The PettingZoo wrapper keeps those options open.

**As implemented in M4 (human only):** `python/zai_rl/` (`models/human.py`, `algos/ppo_human.py`, `envs.py`, `train.py`, config `configs/train/human_s0.yaml`). Differences from the plan above: the human critic is an MLP over the global state (512-256) instead of entity self-attention (revisit if the value fit is poor); the actor uses `nn.GRU` over whole BPTT chunks, running the encoders once over all steps and splitting the GRU only where an episode restarts (4-5x faster update than a per-step `GRUCell` loop on CPU, verified equal by a test); the actor has ~0.69 M parameters. Training envs are one `VecEnv` group per zombie count (S0: 2, 3, 4), with zombies played by the Python scripted bots. Curriculum stage configs (`configs/env/s0.yaml`) are loaded without the `ZAI_ZOMBIE_COUNT` / `ZAI_FOOD_COUNT` overrides, which apply to live matches and full-spec runs. Evaluation every 10 iterations plays 64 episodes against scripted zombies (3 zombies) and reports `eval/survived` next to `eval/baseline_survived` (scripted human). Measured S0 baselines (256 episodes): scripted human survives the 5 min cap in 80-85% of episodes (2-4 zombies), random human 0%. CPU pilot throughput (Apple Silicon, 128 envs, 4 torch threads): ~2.2 k env steps/s, dominated by the PPO update (~5 s per 16 k-step iteration vs ~2 s of rollout). **GPU run `human-s0-gpu1` (P100, `configs/train/human_s0_gpu.yaml`, 1024 envs, 2026-11-24):** 200 M env steps in 8770 s (2 h 26 min), ~23-26 k env steps/s end-to-end (~5.3 s per 131 k-step iteration). Evaluation (128 episodes, 3 scripted zombies) went from 0% survival (98% starved) to above the 82.8% baseline at iteration 139 (18 M steps, ~12 min) and finished at 96.9-98.4% survived, 0.8-2.3% caught, 0.8% starved (peak 99.2%). Ten exported replays of the final checkpoint (`python -m arena.export_replays`): 9 survived the 5 min cap, 1 caught at 155 s. M4 acceptance (survival above the scripted human baseline on S0) is met.

### 4.2 Rollout and process topology

One trainer process, one GPU. Each iteration:
1. Rust `VecEnv.step()` on CPU (all cores).
2. Batched GPU inference for all humans and all zombies (two forward passes).
3. After 128 decisions: compute GAE, run PPO update for the human and MAPPO update for the zombies (sequentially on the same GPU).

The two learners have separate networks, optimizers, checkpoints and metrics, so they are logically the "two trainers" of the PRD while avoiding cross-process lockstep, which would halve throughput.

### 4.3 Co-training schedule

Simultaneous training with an opponent pool. Environments are split into groups:

| Group | Share | Human policy | Zombie policy | Who learns |
|---|---|---|---|---|
| A | 50% | latest | latest | both |
| B | 25% | latest | sampled snapshot | human |
| C | 25% | sampled snapshot | latest | zombies |

- Snapshots saved every 20 M env steps per side to `/checkpoints/{human,zombie}/`. Pool capped at 50 per side (oldest evicted, but scripted bots and every 10th snapshot kept permanently).
- Sampling: prioritized fictitious self-play (PFSP), weight `p * (1 - p)` with `p` = current side's recent win rate against that snapshot, so training focuses on opponents it neither always beats nor always loses to.
- Scripted bots are always in the pool: *scripted zombie* (walk to the most recent heard event, else random waypoint wander), *scripted human* (flee visible zombies closer than 30 m, go to the nearest visible food while carrying fewer than 2 packages, else walk to a random house). The original "food when hunger < 45 s and carrying 0" rule starved: visible food can be 100 m (100 s) away.
- Imbalance guard: if one side's win rate in group A stays above 85% for 3 consecutive evaluations, the losing side gets an extra update epoch per iteration and the winning side's lr is halved until it recovers.
- Made concrete for M6 in sections 4.8.4-4.8.5: frozen opponents in resident slots with leases (an opponent never changes inside an episode), snapshot storage under `/checkpoints/league/<run_id>/pool/`, a 20% scripted share, and a proposed change of the PFSP weight to `(1 - p)^2` + 10% uniform (decision D4).

### 4.4 Curriculum

Stages are advanced by gates measured in the arena against the fixed scripted bots and the previous stage's best checkpoints.

| Stage | Map | Houses | Zombies (sampled) | Food | Cap | Shaping | Gate to advance |
|---|---|---|---|---|---|---|---|
| S0 | 150 x 150 | 3 | 2-4 | 20 | 5 min | on | Human beats scripted zombies (> 60% survive cap); zombies catch scripted human > 60% |
| S1 | 300 x 300 | 6 | 4-8 | 30 | 10 min | on, annealing | Same gates, plus group-A outcome between 25% and 75% |
| S2 | 500 x 500 | 10 | 6-16, eval at `ZAI_ZOMBIE_COUNT` | `ZAI_FOOD_COUNT` | 20 min | off | Final PRD spec |

Within a stage, `noise_localization_sigma` goes 0 -> 0.05. Starting houses/food counts scale with area so densities stay close to S2.

**Revised for M6 (section 4.8.1, proposed):** the curriculum starts at the warm-up stage `s0_small`; the gates become relative to scripted baselines measured at stage start (the absolute "> 60% caught" S0 zombie gate is out of reach: learned zombies caught 1-3.6% on S0), plus balance, forgetting, Elo and exploitability conditions; S0 trains with 4/6/8 zombies; S1 gets 45 food and 8/12/16 zombies. The S2 range above (6-16) predates the default of 20 zombies and should be revisited in M8 (e.g. 12-24 around `ZAI_ZOMBIE_COUNT`).

Budget estimate: S0 ~200 M env steps, S1 ~500 M, S2 1-3 B. At ~70 k env steps/s (section 5.1) that is roughly 1 h, 2 h and 4-12 h respectively on one GPU plus 16-32 CPU cores. On the target P100 server (24 cores, 64 GB RAM) the estimate stays at ~60-100 k env steps/s end-to-end, so S2 takes roughly 3-14 h; measure in M4 before committing to S2 budgets. **Measured in M4:** ~25 k env steps/s end-to-end on the P100 with 1024 envs and 20 simulation threads (section 4.1), about 3x below the estimate, so S1 takes ~6 h and S2 ~11-33 h at the current throughput. Where the time goes (rollout with CPU inference vs the PPO update on the GPU) has not been profiled yet; do it before S1/S2 budgets are committed.

### 4.5 Evaluation

Run by the arena every 20 M env steps, on fixed seeds (500 episodes per matchup), at the PRD spec (S2) unless stated.

- **Fixed-opponent evaluation:** latest human vs scripted zombies and vs a frozen "reference" zombie checkpoint per stage; latest zombies vs scripted human and vs a reference human.
- **Elo:** match outcome is zombie win (catch), human win (survives cap) or draw (human starves). Cross-play between the last 10 snapshots of each side and the scripted bots (anchored at Elo 1000). Reported as Elo curves per side. (Section 4.8.6: cross-play is bipartite, so only one anchor can be fixed; the plan anchors the scripted zombies at 1000 and reports the scripted human's rating as a measured value.)
- **Exploitability probe** (every 200 M steps): train a fresh best-response for 20 M steps against the frozen latest opponent; a large gap means the latest policy is exploitable (cycling).

Tactic metrics (logged per evaluation, these show *what* is learned, not just reward):

| Side | Metric | Definition |
|---|---|---|
| Human | Survival time distribution | median, p10, p90; split by cause of death |
| Human | Food efficiency | packages eaten / picked; mean carried; fraction of time with `carried >= 3` |
| Human | Noise discipline | noise events per minute by type; pickups made while a zombie is within 100 m |
| Human | Door usage | closes per episode; *defensive closes* (a zombie within 30 m of the closed door at close time); escapes via the opposite door within 20 s |
| Human | House usage | fraction of time inside houses; zombie distance when entering |
| Zombies | Catch rate, time to catch | per zombie count |
| Zombies | Noise response | latency from a heard event to movement toward it; fraction of zombies that respond vs stay as sentinels |
| Zombies | Search dispersion | mean pairwise distance while no event heard in 30 s (spread-out search vs clumping) |
| Zombies | Encirclement | for zombies within 50 m of the human, max angular gap; encircled if < 120 degrees; fraction of catches preceded by encirclement |
| Zombies | Flanking | distinct approach sectors (8 sectors) of the 3 nearest zombies in the 30 s before a catch |
| Zombies | Door coverage | while the human is in a house, fraction of time both doors have a zombie within 10 m |

Baselines for every metric come from the scripted bots and random policies; "emergent tactics" means sustained improvement over those baselines, not just reward growth.

### 4.6 M5 plan: zombie training

Decisions accepted by the user (2026-11-24): zombie step penalty 0 on S0; per-agent distance shaping (team `min` kept for ablation); train against the scripted human only, with the frozen M4 human as an evaluation opponent; acceptance relative to the measured scripted-zombie baseline (the S0 gate of section 4.4 to be revisited in M6); observation normalization by `map_size_m` kept until M6 (fix it before S1, since changing it now invalidates the M4 checkpoint used as an evaluation opponent).

**As implemented (code for pilot stages 1-3):** `crates/zai-sim` (`Env::zombie_potentials`, `Env::zombie_rewards`, `reward.zombie_distance_shaping_range_m` / `_mode`; `StepResult.zombie_reward` is now only the team part), `python/zai_rl/` (`algos/common.py`, `algos/mappo_zombie.py`, `models/zombie.py`, `envs.ZombieTrainingEnv`, `policy.RecurrentZombiePolicy`), configs `configs/train/zombie_s0{,_gpu}.yaml`, `make train-zombie{,-gpu}`. Differences from the plan below: the entrypoint is a separate module `zai_rl.train_zombie` instead of a `side` switch in `train.py` (the human path stays untouched); `HumanTrainingEnv` was kept and `ZombieTrainingEnv` added next to it (shared `group_sizes`), to be merged into one `MatchTrainingEnv` for M6; minibatches split the sequences with `tensor_split` because the agent count depends on the group split (256 envs give 767 agents); the actor uses 32-unit entity encoders and a 64-unit ray projection (0.17 M parameters). Diagnostics added for the pilot: `train/nearest_zombie_m` (mean human-to-nearest-zombie distance over the rollout, from the global state) and episodes recorded during training (`replay_every`, watched per run in the viewer). The stage 2 sanity run against `RandomHuman` was dropped: the random human starves at ~90 s in every episode and almost never makes a sound louder than walking (10 m), so zombies have nothing to hear (2 M steps on the CPU pilot: 0% caught, scripted zombies 0.3%). First CPU measurement (64 episodes per count): scripted zombies catch the scripted human in 6.8% of S0 episodes (2 zombies 1.6%, 3 zombies 6.2%, 4 zombies 12.5%), random zombies 0%; the trainer measures it again at startup with 256 episodes per count.

**GPU run on S0 and warm-up stage `s0_small` (2026-11-24):** `zombie-s0-gpu1` plateaued at 1-3.6% caught (baseline 4.3% [3.1, 6.0]) and a nearest-zombie distance of ~44 m over its first 35 M steps, and was stopped. Diagnosis (user observation, consistent with the mechanics): the scripted human takes long to pick its first package, and until it does the only sound it makes is walking (10 m), so zombies have nothing to localize; and with a 50 m spawn distance most of the episode is spent where the distance potential barely changes. Warm-up stage `configs/env/s0_small.yaml`: 75 m map, 3 houses, 50 food with 3 packages within 15 m of the human's spawn (new `food_near_spawn_count` / `food_near_spawn_radius_m`, default 0, which leaves maps and replays unchanged), spawn distance 25 m. Scripted baseline there (256 episodes per count): caught 12.1% / 16.0% / 14.8% with 2 / 3 / 4 zombies, starved 33% / 28% / 23% (the scripted human flees more than it forages when zombies are always close). The run `zombie-s0small-gpu1` (`configs/train/zombie_s0small_gpu.yaml`) has no frozen-human evaluation, because observations are normalized by map size and the M4 human was trained on 150 m. **Result (2026-11-24, 200 M env steps in 12 672 s, ~16 k env steps/s):** eval catch rate against the scripted human went from 1.8% to 96.6% (93.8% / 96.9% / 99.2% with 2 / 3 / 4 zombies; peak 98.2%) against a 14.3% scripted-zombie baseline, mean episode length 83 s, starved 3%. M5 acceptance is met on `s0_small` (not on S0, where `zombie-s0-gpu1` stays below the baseline). This team is the frozen zombie opponent of the fine-tunes in section 4.7.

**Data layout: flat agent axis, no padding.** Each env group has a fixed zombie count, so the total number of zombie agents `A = sum_g E_g * N_g` is constant for the whole run (S0, E=1024, counts 2/3/4: A ~ 3072). Zombie tensors are `[A, 252]` with a static `agent_env [A]` index (and `agent_slot [A]`). Rollout buffers are `[T, A, ...]`, so `gae()`, the `[T, E] -> [chunks*E, L]` BPTT reshape and the `h0` storage of `ppo_human.py` apply unchanged with `E` replaced by `A`. Per-env arrays (`done`, `state`) are broadcast with `agent_env`. This replaces the "pad to `N_max` with a validity mask" sketch of section 5.1; `zombie_valid` is all-true inside a group and is only asserted.

**Reuse and new code (mapped onto M4):**

| Piece | M4 code | M5 action |
|---|---|---|
| GAE, `RunningMeanStd`, `EpisodeStats`, `save_checkpoint`, BPTT reshape, clipped PPO loss | `algos/ppo_human.py` | Move to `algos/common.py`; `HumanPPO` imports them (no behavior change, existing tests must pass). |
| GRU over segments split at episode starts | `HumanActor.sequence` | Extract `run_gru_segments(gru, x, h0, starts)` into `models/encoders.py`; used by both actors. |
| `SetEncoder`, `RayEncoder`, `mlp` | `models/encoders.py` | Reused as is (16 zombie rays -> 4 conv positions). |
| Env groups + fixed opponent | `envs.HumanTrainingEnv` | Generalize to `MatchTrainingEnv(groups, human_factory=None, zombie_factory=None)`: the side without a factory is supplied by the caller; `step()` returns both sides' views (human `[E]`, zombies flat `[A]` including `final_zombie_obs` / `final_state` regrouped). `HumanTrainingEnv` stays as a thin wrapper. This is also the M6 group A/B/C structure. |
| Actor | `models/human.py` | New `models/zombie.py`: `ZombieActor` (self 12 + `SetEncoder` over heard 8x9 / peers 8x5 / houses 10x8 with valid masks + `RayEncoder` 16x3 -> MLP 128 -> GRU 128 -> 17 logits, no mask, no agent id), `step` and `sequence` with the same equivalence test as the human. |
| Critic | `HumanCritic` (MLP on state) | `ZombieCritic`: agent-specific global state `concat(state[agent_env], own_obs)` = 1396 -> 512 -> 256 -> 1, per-agent value. Store `state` per env `[T, E, 1144]` (0.6 GB at E=1024) and gather at update time, not per agent (2.2 GB). |
| Algorithm | `HumanPPO.collect/update` | New `algos/mappo_zombie.py` `ZombieMAPPO`: per-agent rewards, per-agent GAE and values, advantage normalization per minibatch over agents, value normalization, truncation bootstrap from `final_zombie_obs` + `final_state`; hyperparameters as section 4.1 with 3 epochs, 8 minibatches, `hidden 128`. |
| Entrypoint | `train.py` (human only) | `TrainConfig.side: human | zombie` plus `opponent` and `mappo` sections; checkpoints under `/checkpoints/zombie/<run_id>/`; `MetricsLogger(run_id, "zombie")` (the server already accepts `zombie`). Configs `configs/train/zombie_s0.yaml` (CPU pilot, 256 envs) and `zombie_s0_gpu.yaml` (1024 envs, 20 sim threads); Makefile `train-zombie`, `train-zombie-gpu`. |
| Eval policy wrapper | `policy.RecurrentHumanPolicy` | New `RecurrentZombiePolicy(actor, num_envs, num_zombies)`: `act([E, N, 252]) -> [E, N]`, `reset(done)` clears the N hidden states of finished envs; `from_checkpoint`. |
| Replays | `arena/export_replays.py` (human checkpoint only) | Add `--zombie-checkpoint` and `--human {scripted,random,checkpoint}`; `record_episode` is already policy-agnostic; replay field `zombies: "policy"` (`ReplayAgent::Policy` exists). |

**Rewards for M5 (replace the section 3.8 zombie shaping):**
- Distance potential, per agent: `Phi_i(s) = c * (1 - min(d_i, D) / D)` with `d_i` the distance of zombie `i` to the human, `c = 0.5`, `D = 100 m` (a constant, so the per-meter signal does not change with the map size). Non-negative. Terminal potentials: `Phi_i(Caught) = c` (the value at `d = 0`) for every zombie, `Phi_i(Starved) = 0`; truncation uses the real `Phi_i(s_T)` and bootstraps. Per episode the shaping sums to `gamma^T * Phi_T - Phi_0`: a starvation ending adds nothing beyond the policy-independent `-Phi_0`, a catch adds an aligned bonus `+c * gamma^T` (catch effectively worth up to 1.5), so neither ending can be exploited. The current formula (`-scale * min_d / map_size`, terminal 0) paid up to `+0.5 * 212 / 150 ~ +0.71` on S0 for a starvation ending. The `(gamma - 1) * Phi` term costs at most `0.0025` per decision when close without catching, which pushes towards closing the gap (aligned), unlike the old negative form, which paid for staying far. Implemented in Rust (`Env::zombie_potential` per agent); `VecEnv` returns `zombie_reward [E, N]`; the team mode (nearest zombie) stays as `zombie_distance_shaping_mode: team_min` for ablation.
- `zombie_step_penalty: 0` on S0. With `-0.002` per decision and a low catch probability, continuing is worth about `-0.002 / (1 - 0.995) = -0.4`, while a starvation ending is worth 0, so making the human starve is worth ~40% of a catch, which turns into a food-guarding objective that the PRD does not ask for. `gamma` already gives zombies a preference for early catches.
- Catch `+1.0` shared by every zombie, unchanged.

**S0 configuration:** `configs/env/s0.yaml` as in M4 (150 x 150 m, 3 houses, 20 food, 5 min cap, spawn distance 50 m, 120-degree human field of view, zombies 0.7 m/s), training counts 2/3/4 in equal groups, opponent `ScriptedHuman` (reserve 2, flee 30 m).

**Baseline and acceptance:** the scripted-zombie catch rate against the scripted human on S0 has not been measured (M4 only logged the human's 82.8% survival). The trainer measures it at startup: `arena.matches.play` with scripted human vs scripted zombies, 256 episodes per count (2, 3, 4) on fixed seeds, reporting `eval/baseline_caught` per count and pooled with a Wilson 95% interval; random zombies are logged as the floor. Evaluation every 10 iterations uses the same seeds (paired). Acceptance: pooled learned catch rate above the baseline with non-overlapping 95% intervals (at 768 episodes and p ~ 0.1 the interval is about +/-2.1 points). Also reported, not gating: catch rate against the frozen M4 human (`human-s0-gpu1` latest), time to catch, starved share, mean nearest-zombie distance.

**De-risking before the GPU run:**
1. Tests (CPU, minutes): `ZombieActor.sequence` equals repeated `step` with resets; flat-agent regrouping round-trip; per-agent GAE stops at env done; truncation bootstrap uses `final_zombie_obs`; Rust rule tests for the potential (starvation ending pays 0, catch step pays `+1 + gamma * c - Phi`).
2. Sanity run against `RandomHuman` (noisy, no fleeing), 128 envs, 2 M env steps (~15 min on the pilot): catch rate must clearly exceed scripted zombies vs random human.
3. Pilot against the scripted human, 256 envs, 5-10 M env steps: go if the mean nearest-zombie distance falls, critic explained variance > 0.2 and `eval/caught` trends above the baseline.
4. Read `perf/rollout_s` / `perf/update_s` of `human-s0-gpu1` (already logged) before the long run. Expected GPU throughput ~15-25 k env steps/s (45-75 k agent steps/s; the simulation cost equals M4 because `VecEnv` always encodes all observations; the update handles 3x more transitions with a smaller network), so 200 M env steps take ~2.5-4 h.

### 4.7 Fine-tunes against frozen opponents (first co-training step)

Before the M6 scheduler (section 4.3) exists, co-training is approximated by hand: each side continues from its best checkpoint against the other side's frozen latest policy, alternating. All runs are on `s0_small` on the GPU server.

**Mechanism.** Both trainers accept `opponent: policy` with a checkpoint of the other side (`zombie_checkpoint` in `zai_rl.train`, `human_checkpoint` in `zai_rl.train_zombie`) and an `init_checkpoint` of their own side. `init_checkpoint` loads actor, critic and `value_rms` but starts a fresh optimizer and fresh step/iteration counters, unlike `--resume`. Frozen recurrent opponents clear the hidden state of finished envs (`reset(done)` in `HumanTrainingEnv.step`), and the frozen human factory shares one actor across all env groups. The starting point is evaluated and logged at step 0, so the eval curves start at the inherited performance. With a policy opponent, the eval runs against the training opponent (`eval/*`) and also against the scripted bot of the other side (`eval_scripted/*`) to detect forgetting. The baselines change meaning: `eval/baseline_survived` of the human fine-tune is the scripted human against the frozen zombies, and `eval/baseline_caught` of the zombie fine-tune is still the scripted zombies against the training opponent (the frozen human). Fine-tunes use a lower learning rate (human 1e-4, zombies 1.5e-4) and entropy (0.005 -> 0.001) so the start does not wash out the pretrained policy. Training replays and exported models use the largest trained zombie count (`stage_name` ignores `zombie_count` when matching a model to a stage), since the env file's own count (3 on `s0_small`) may not be one of them.

| Run (config) | Starts from | Trains against | Zombie counts | Steps (wall) | Eval vs training opponent | Eval vs scripted opponent |
|---|---|---|---|---|---|---|
| `human-s0small-ft1` (`human_s0small_ft_gpu.yaml`) | `human-s0-gpu1` (M4, trained on S0) | frozen `zombie-s0small-gpu1` | 2, 3, 4 | 88 M (96 min), `final.pt` | survived 15.6% -> 95.3% (100% at 81 M); caught 84% -> 4%; scripted human 1.6% | scripted zombies: survived 86-98% (93.0% at the end; 96.1% at the first eval), little forgetting |
| `human-s0small-z10-ft2` (`human_s0small_z10_ft_gpu.yaml`) | `human-s0small-ft1` | frozen `zombie-s0small-gpu1` | 10 | 11 M (12 min), stopped | survived 96.1% -> 97.7%; scripted human 0% | scripted zombies: 95-98% |
| `zombie-s0small-crowd-ft1` (`zombie_s0small_crowd_ft_gpu.yaml`) | `zombie-s0small-gpu1` (M5) | frozen `human-s0small-ft1` | 8, 9, 10 | 17 M (49 min), stopped, still rising | caught 18.0% -> 48.7% (z8 43.8%, z9 53.1%, z10 49.2%); scripted zombies 18.1%; mean length 267 -> 216 s | scripted human: caught 96.6-99.0%, no forgetting |

Observations:
- The M4 human transfers to `s0_small` badly: it survives 15.6% of episodes against the M5 zombies at step 0 (84% caught), but recovers to ~95-100% in ~80 M steps while staying at 86-98% (93% at the end) against the scripted zombies.
- More zombies did not help the M5 team against that human. `human-s0small-ft1` already survived 96% of episodes against 10 frozen M5 zombies at the start of `-z10-ft2`, so that run had little to learn and was stopped early.
- The crowd fine-tune is the first run where learned zombies beat a learned human clearly above the scripted baseline (48.7% vs 18.1% with 8-10 zombies). It was stopped at 17 M of the planned 100 M steps with the catch rate still rising, so the next step is to continue it (`--resume`) and then fine-tune the human against the resulting team.
- Throughput of the crowd fine-tune is ~6 k env steps/s (~54 k agent steps/s at 9 zombies on average), against ~15-16 k env steps/s for the human fine-tune and the 2-4 zombie M5 run: the zombie update scales with the agent count.
- Every step of this loop trains against a single frozen opponent, so each side can overfit to its current opponent and cycle. Comparing each new checkpoint against older snapshots of the other side is the M6 opponent pool's job (section 4.3) and is not done yet; the `eval_scripted/*` curves only guard against forgetting the scripted bots.

### 4.8 M6 plan: co-training and arena

Status: accepted (2026-11-27) with the recommended option of every decision in 4.8.11; implementation in progress. M6 replaces the manual fine-tune loop of section 4.7 with one league trainer (both learners, env groups A/B/C of section 4.3, snapshot pool with PFSP) and an arena (cross-play Elo, exploitability probe, tactic metrics) whose results reach the viewer. It is split into six sub-milestones, each runnable and testable on its own, in this order: **M6.0** throughput profiling (no behavior change), **M6.1** observation layout v2 and distilled starting checkpoints, **M6.2** Rust episode tactic counters and arena tactic metrics, **M6.3** `MatchTrainingEnv` and the league trainer, **M6.4** arena (Elo, cross-play, probe) and viewer league panel, **M6.5** curriculum gates and the full run s0_small -> S0 -> S1. The observation change comes before any long run because it invalidates every checkpoint (4.8.2); profiling comes first because it is cheap and decides how much optimization work M6.5 needs (4.8.3).

#### 4.8.1 Starting stage and curriculum gates

M6 starts on `s0_small`, the only stage where both sides have competent learned policies (4.6, 4.7). The S0 zombie gate of section 4.4 (catch the scripted human > 60%) is not reachable as written: on S0 the best learned zombies caught the scripted human in 1-3.6% of episodes against a 4.3% scripted baseline with 2-4 zombies, and 4.7 showed that even on `s0_small` the team needed 8-10 zombies to beat a learned human. The gates are therefore relative to baselines measured at the start of each stage on the same seeds (as M5 acceptance was), and S0 is trained with more zombies.

| Stage | Env config | Zombie counts (equal groups) | Budget (min / max env steps) | Starts from |
|---|---|---|---|---|
| `s0_small` | `configs/env/s0_small.yaml` (75 m, 3 houses, 50 food, 25 m spawn, 5 min) | 3, 6, 9 | 100 M / 300 M | distilled starters (4.8.2) |
| S0 | `configs/env/s0.yaml` unchanged (150 m, 3 houses, 20 food, 50 m spawn, 5 min) | 4, 6, 8 (was 2-4) | 100 M / 500 M | latest learners of `s0_small` |
| S1 | new `configs/env/s1.yaml`: 300 m, 6 houses, 45 food (S2 food density), 75 m spawn, 10 min cap, shaping on, annealed to 0 over the last 40% of the budget | 8, 12, 16 | 200 M / 500 M | latest learners of S0 |

Gate `G(stage)` passes when every condition holds at two consecutive arena evaluations (fixed seeds, 128 episodes per training zombie count, so 384 per matchup; `b_*` are the scripted baselines measured at stage start):

| Condition | `s0_small` -> S0 | S0 -> S1 | S1 -> S2 (reported, not required for M6) |
|---|---|---|---|
| Human anchor: latest human vs scripted zombies, survived | >= 80% | >= max(60%, `b_survived` + 10 pts) | as S0 |
| Zombie anchor: latest zombies vs scripted human, caught | >= 80% | >= max(3 * `b_caught`, `b_caught` + 15 pts), Wilson lower bound above the baseline's upper bound (at `b_caught` = 4.3%: >= 19.3%) | as S0 |
| Balance: zombie score `s_z = caught + 0.5 * starved` over group A training episodes of the last 20 M steps | within [0.2, 0.8] | within [0.2, 0.8] | within [0.25, 0.75] |
| No forgetting: each side's mean score against pool snapshots of this stage older than 40 M steps | >= 0.55 | >= 0.55 | >= 0.55 |
| Elo: each side's latest snapshot vs its stage starter | +100, bootstrap 95% intervals not overlapping | same | same |
| Exploitability probe gap (4.8.6), both sides | <= 20 pts | <= 20 pts | <= 20 pts |
| Shaping | on | on | scale 0 for the last 100 M steps |

A gate is only evaluated after the stage's minimum budget. If it has not passed at the maximum budget, the league stops (it does not advance) and reports which conditions failed; that is a user decision point, not an automatic retry. On advancing: the learners keep their weights and `value_rms` (fresh optimizers, learning-rate schedule restarted), the pool is reset to the scripted bots plus the previous stage's final snapshots (protected, tagged with their stage), win-rate statistics are reset, the scripted baselines are re-measured, and Elo restarts as a separate table per stage (ratings are not comparable across maps). The previous stage's final snapshots also become the fixed "reference" opponents of section 4.5 for the new stage.

#### 4.8.2 Observation layout v2: normalization by constants

**Status (2026-11-27): implemented** as specified below (`obs.rs`, `OBS_VERSION = 2`, scales exported to Python as `zai_env.layout.NEAR_M` / `FAR_M` / `TIME_SCALE_S`). One change from the plan: door deltas need no approximation in the adapter, because doors sit at `center +- axis_y * depth / 2` and the rotation and house depth are known, so `zai_env.legacy.adapt_v1_*` rebuild v1 exactly (checked to 1e-5 against `python/tests/fixtures/obs_v1.npz`, recorded from the v1 encoders on `s0_small`, S0 and the defaults, together with unchanged state hashes and identical scripted-bot actions). v1 checkpoints (no `obs_version` key) play as frozen policies through the adapter when given the env config; training, resuming, `init_checkpoint` and ONNX export refuse them. The server skips models whose `obs_version` (default 1) is not the simulation's. Distillation is still to do.

**Problem.** Every absolute position and house/door delta is divided by `cfg.map_size_m` (`obs.rs`: `let size = cfg.map_size_m`; section 3.6 says `/500`, which was only true at S2). A door 2 m away is 0.027 on `s0_small`, 0.013 on S0 and 0.004 on S2, so the same input means different distances on each stage, which is part of why the M4 human (S0) fell to 15.6% survival on `s0_small` (4.7). The zombie `self.time` field (`time / episode_cap_s`) has the same problem across caps of 5, 10 and 20 min.

**v2 rules.** Two constants in `obs.rs`: `NEAR_M = 100` (vision radius, shaping range) and `FAR_M = 500` (PRD map side). Deltas to static landmarks use `sat(v) = v / max(|v|, NEAR_M)` (linear up to 100 m, then a unit direction), plus a far-scale distance `|v| / FAR_M`, so precision near the agent is the same on every map and far landmarks keep their direction and distance. Absolute positions are `pos / FAR_M`. Boundary awareness comes from four wall distances `min(d, NEAR_M) / NEAR_M`. A new `OBS_VERSION = 2` constant is exported by `zai_sim`, saved in checkpoints and ONNX metadata, and checked on load (`from_checkpoint`, `export_model`, server `policy.rs`).

| Vector | Block (v1 -> v2) | v2 fields |
|---|---|---|
| Human obs (584 -> **608**) | self 16 -> **20** | `x, y` (`/500`), `wall_w, wall_e, wall_s, wall_n`, heading sin/cos, `vx, vy`, `carried`, one-hot `carried_0..5`, `hunger`, `inside`, `since_noise` |
| | zombies 32 x 6, food 16 x 4, rays 64 x 3 | unchanged (already `/vision_radius`, `/speed`) |
| | houses 10 x 12 -> 10 x **14** | `valid` (new; replaces the `abs().sum() > 0` test in the actors), `dx, dy` = `sat(center - pos)`, `dist` = `|center - pos| / 500`, rot sin/cos, `front_dx, front_dy`, `back_dx, back_dy` (`sat`), `front_closed`, `back_closed`, `door_visible`, `inside` |
| | starts | self 0, zombies 20, food 212, houses 276, rays 416 |
| Zombie obs (252 -> **276**) | self 12 -> **16** | `x, y` (`/500`), `wall_w, wall_e, wall_s, wall_n`, heading sin/cos, `vx, vy`, `inside`, `blocked`, `since_heard`, `num_zombies`, `time_left` = `(cap - t) / 1200` (replaces `t / cap`), `bias` |
| | heard 8 x 9, peers 8 x 5, rays 16 x 3 | unchanged (`/300`, `/50`, `/10`) |
| | houses 10 x 8 -> 10 x **10** | `valid`, `dx, dy` (`sat`), `dist` (`/500`), rot sin/cos, `front_dx, front_dy`, `back_dx, back_dy` (`sat`) |
| | starts | self 0, heard 16, peers 88, houses 128, rays 228 |
| State (1144, unchanged size) | human, zombies, food, houses | all positions `/500`; `zombies.dist_to_human` = `min(d, 100) / 100` (the shaping potential's own scale) |
| | time 4 | `time` (`t / cap`), `time_left` (`(cap - t) / 1200`, replaces `bias`), `num_zombies`, `active_food` |

The actors exclude `self.x, self.y` from their input by default (a column selection in `HumanActor` / `ZombieActor`, not a layout change): absolute position would be out of distribution on every larger map, while houses already act as landmarks. The fields stay in the layout for the Python scripted bots, diagnostics and the adapter below. `train/nearest_zombie_m` must be computed from state positions (`* 500`), since `dist_to_human` now saturates at 100 m.

**What this invalidates.** Every human and zombie checkpoint (input dims change), every ONNX model in `data/models/` (the server already rejects them on `obs_dim`; move them to `data/models/legacy/` so the viewer does not list them), the policy parity fixtures (`python -m arena.export_model --parity-fixtures crates/zai-server/tests/fixtures`, since `obs_dim` changes), the Python scripted bots (they read `self.x/y` and `houses.dx * map_size`; they must read meters from the new fields, and their baselines are re-measured), and the `layout_fields_match_the_writers` test. Replays stay valid: they store actions and the dynamics do not change, which a test re-simulating an existing replay file to its stored `state_hash` must confirm.

**Cheapest new starting checkpoints: distillation through an adapter (about 1 h of GPU time instead of about 8 h).** v2 keeps enough information to rebuild v1 on maps up to about 140 m: `x/size = (x/500) * 500 / size`, house deltas = direction x far distance, `time / cap` from `time_left`, state positions rescaled and `dist_to_human` from positions. Door deltas beyond 100 m saturate; the adapter then uses the house-center distance (at most ~4 m error, and only on a 75 m map when the human is more than 100 m from a door). `zai_env.legacy.adapt_v1(obs_v2, config)` lets the old frozen actors and critics play in the v2 env, and `python -m zai_rl.distill` trains a fresh v2 actor and critic on them:
- On-policy DAgger: each env's episode is driven by the teacher with probability `beta` (1 -> 0 over the first 30% of steps), otherwise by the student. Loss: `KL(teacher || student)` over masked logits (both human heads), on 64-step BPTT chunks; critic: MSE to the teacher's normalized value on `adapt_v1(state)`, `value_rms` copied from the teacher.
- Opponents: half of the envs against the other side's teacher (through the adapter), half against the scripted bot of the other side. Counts as in the `s0_small` stage.
- Teachers: human `human-s0small-ft1/final.pt` (main starter), zombies `zombie-s0small-crowd-ft1/latest.pt` (main starter, stopped at 17 M; resuming it on v1 first would cost ~4 h for a policy the league continues anyway), plus `zombie-s0small-gpu1` (M5) and `human-s0-gpu1` (M4) as extra pool seeds (the M4 human also seeds S0).
- Budget: 10 M env steps per teacher (~10 min each at ~20 k env steps/s; only a forward pass for the teacher and a supervised backward pass for the student).
- Acceptance per student, 512 fixed-seed episodes per count: outcome rates against the scripted bot and against the other main teacher within 3 points of the teacher's (with overlapping Wilson intervals), mean per-decision KL < 0.05.

Alternatives: train both sides from scratch inside the league (validates the league end to end but costs an unknown number of hours with a cold pool and zombies that get no signal from a starving random human, 4.6); or re-run the M4/M5/4.7 recipes on v2 (~8 h of GPU time across five runs).

#### 4.8.3 Throughput: profiling and optimizations

**What is known.** End-to-end on the P100 (1024 envs, 20 simulation threads): 25 k env steps/s for the human vs scripted zombies (5.3 s per 131 k-step iteration), 15.8 k for zombies with 2-4 agents (8.3 s per iteration, wall time including evaluation) and ~6 k with 8-10 agents (22 s). From those two zombie runs, each zombie agent costs ~2.2 ms per iteration (~17 us per agent decision, rollout inference plus update), with a fixed cost of about 1.5 s per iteration (simulation, glue code, scripted opponent). `perf/steps_per_s` excludes evaluation, replay and checkpoint time (it is measured before them), so the wall clock is 5-10% slower than logged. The raw simulation (9.2 us per decision step per core, M1) is not the bottleneck: 1024 envs on 20 cores take ~0.5-1 ms per step, ~0.1 s per 128-step rollout.

**M6.0 profiling (GPU server, about 1 h, no behavior change).**
1. Read `perf/rollout_s`, `perf/update_s` and the wall time between metric points of `human-s0-gpu1`, `zombie-s0small-gpu1` and `zombie-s0small-crowd-ft1` from the metrics database (`GET /api/metrics?run_id=<id>&keys=perf/rollout_s,perf/update_s,perf/steps_per_s`). The wall-time gaps give the evaluation overhead.
2. Add finer timers, logged as `perf/*` by both current trainers: `sim_s` (time inside `VecEnv.step`), `opponent_s` (bot or frozen policy `act`), `policy_s` (learner forward including host-to-device and device-to-host copies, with `torch.cuda.synchronize()` around it on profiling runs only), `glue_s` (rest of the rollout: `np.concatenate`, `.copy()`, list appends), `update_s` split into `update_gather_s` (state gather) and `update_step_s`, plus `eval_s`, `replay_s` and `checkpoint_s`.
3. One `torch.profiler` trace of a single iteration per trainer (`record_function` ranges for the phases above), plus `nvidia-smi dmon -s u` and per-core CPU utilization sampled across one iteration. The expected pattern: 20 cores idle during the update and the GPU idle during the rollout.
4. Write the result to `docs/reports/throughput_profile.md` and replace the budget estimates below with the measured ones.

**Optimizations to consider (in expected order of payoff; each is checked for equal learning curves on `s0_small` at 20 M steps):**
1. **Factorized zombie critic first layer.** `W1 [s; o] = W_s s + W_o o`: compute `W_s s` once per (t, env), `[T, E, 512]`, and gather it per agent, instead of gathering `[mb, 64, 1144]` states (675 MB per minibatch at 9 zombies) and multiplying per agent. Mathematically identical (existing critic weights split into `W_s`, `W_o`); it removes ~65% of the critic's first-layer FLOPs at 6 zombies and more at higher counts.
2. **One Rust call per step for all groups.** Today each group is a separate `VecEnv` (own rayon pool) stepped one after another from Python, with per-group `np.concatenate`; M6 has 9+ groups (A/B/C x 3 counts), so each parallel loop gets ~100 envs over 20 threads and the copies grow. Add `zai_sim.VecEnv.from_groups([(config, num_envs), ...])`: one env list with per-env configs, one parallel loop, zombie obs written directly on the flat agent axis `[A, 276]` with `agent_env`, and no Python concatenation.
3. **Preallocated rollout tensors and pinned copies.** Replace the per-step lists plus `torch.stack` (a second copy of ~1.2 GB of zombie observations per iteration) with `[T, ...]` tensors filled in place; copy VecEnv buffers through a pinned staging tensor with `non_blocking=True`.
4. **Evaluation out of the training loop.** The arena (4.8.6) does the fixed-seed evaluations in its own process; the league trainer only logs training-episode outcomes and tactic counters, which come for free.
5. **Zombie update subsampling.** Use a random 50% of the zombie sequences per epoch (transitions of zombies in the same env are strongly correlated). This changes the algorithm, so it must show equal curves before adoption.
6. **CUDA graphs for the fixed-shape rollout forward passes.** Learner batch sizes are static. `torch.compile` is not an option: Triton (the inductor GPU backend) requires compute capability 7.0, and the P100 is 6.0.
7. Later, if still bound by the update: overlap the next rollout with the current update (one-iteration policy lag, corrected by the PPO ratio). Not planned for M6.

**Budget estimates for co-training on the P100** (model: 1.5 s fixed + 2.9 s human learner on 75% of the envs + 2.2 ms per learner zombie agent on 75% of the agents + 0.3 s frozen inference, per 131 k env-step iteration; "improved" assumes optimizations 1-4 cut the per-agent cost to 1.0 ms and the fixed and human parts to 1.0 s and 2.0 s):

| Stage | Mean zombies | Env steps | Current: env steps/s, time | Improved: env steps/s, time |
|---|---|---|---|---|
| `s0_small` | 6 | 200 M | ~8.9 k, ~6 h | ~16.6 k, ~3.3 h |
| S0 | 6 | 300 M | ~8.9 k, ~9.4 h | ~16.6 k, ~5 h |
| S1 | 12 | 500 M | ~5.2 k, ~27 h | ~10.5 k, ~13 h |
| S2 (M8, for reference) | 18 | 1-3 B | ~3.7 k, 3-9 days | ~7.7 k, 1.5-4.5 days |

M6 in total (three stages) is about 42 h at the current throughput and about 22 h improved, plus ~10% for the arena and about 1 h per exploitability probe. Optimizations 1-4 are therefore recommended before S1; S2 (M8) is not practical without them.

**Measured in M6.0 (2026-11-27, `docs/reports/throughput_profile.md`).** The trainers are GPU-bound (simulation 8-11% of an iteration, CPU user time 12-14%, GPU 46-73% busy). The largest single cost was not in the list above: `run_gru_segments` ran the GRU once per time step at which any row of a minibatch restarted an episode, which in large minibatches is nearly every step. Running every episode segment as its own zero-padded row (exact, tested against the old implementation) cut the 8-10 zombie update from 9.5 s to 6.8 s (7.6 k -> 8.8 k env steps/s) and makes the update cost independent of episode length. Measured cost model per iteration: 1.8 s fixed, 2.8 s for the human learner, 1.21 ms per learner zombie agent and ~0.5 s of frozen inference, which gives ~12.2 k env steps/s on `s0_small`/S0 (6 zombies), ~8.1 k on S1 and ~6.0 k on S2, so about 28 h for the three M6 stages instead of 42 h. Remaining order of payoff: factorized critic first layer (~1-1.5 s per crowd iteration), evaluation in the arena (6-10%), opponent inference batched per slot (16% in the crowd run), one Rust call for all groups with preallocated buffers (13-17%), CUDA graphs. The trainer service sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, without which the crowd run fragmented GPU memory and failed with ~10 GB available (another container held 5.5 GB).

#### 4.8.4 `MatchTrainingEnv` and the league trainer

**Status (2026-11-27): implemented** in `zai_rl/train_league.py` (`League`, `LeagueConfig`, `Guard`), `zai_rl/league/pool.py`, `zai_rl/league/slots.py` and `envs.MatchTrainingEnv` / `league_groups`, configs `configs/train/league_cpu.yaml` and `league_s0small_gpu.yaml`, `make train-league{,-gpu}`. Differences from the text below: `HumanTrainingEnv` / `ZombieTrainingEnv` stay as they are (no wrappers); the learners' `collect` was split into `HumanRollout` / `ZombieRollout` (`begin/act/record/finish`, bit-identical rollouts) and the league drives both through them; curriculum gates and stage advance (M6.5) are not in yet, and the trainer runs a light anchor evaluation against the scripted bots (`anchor/*`, every `eval_every` iterations) until the arena takes over.

**Environment.** `envs.MatchTrainingEnv(config, groups, pool_sides, seed, num_threads)` with `groups` a list of `GroupSpec(kind in {A, B, C}, zombie_count, num_envs)`. Defaults: A 50%, B 25%, C 25% of `num_envs`, each split equally across the stage's zombie counts (`group_sizes`), so 9 groups. Env axis `[E]` ordered by kind, then count. Static index arrays built once:
- `human_env [E_h]`: envs of A and B (human learner rows); `zombie_env [E_z]`: envs of A and C; `agent_env [A_z]`: for each learner zombie, its row in `zombie_env` (zombie `n` of env `e`, as in M5), so the MAPPO buffers stay `[T, A_z, ...]` with state stored per env `[T, E_z, 1144]`.
- Opponent rows: zombies of group B envs (`[A_B]`) and humans of group C envs (`[E_C]`), each with a per-env `slot [E]` index (below).
- `step(human_actions [E_h, 2], zombie_actions [A_z]) -> MatchStep` holding `human: HumanStep` (rows `E_h`), `zombie: ZombieStep` (rows `A_z`, `final_obs`, `final_state`), and `episodes`: one record per finished env with `(env, kind, zombie_count, opponent_id, outcome, length, stats)`, where `stats` is the Rust tactic-counter row (4.8.6).
- `HumanTrainingEnv` and `ZombieTrainingEnv` become thin wrappers (all envs of kind B or C, with the opponent factory as a single pinned slot), so `train.py` and `train_zombie.py` keep working unchanged. A test checks that the wrappers produce the same trajectories as the current classes for the same seeds.

**Frozen opponents batched per slot.** Each opponent side has `K = 4` resident slots (configurable): each slot holds one pool entry, a frozen actor on the GPU or a scripted bot. Every opponent row carries its env's slot id, assigned at episode start. Per step, for each resident actor slot `k`: `idx = (slot == k).nonzero()`, then `actor_k.step(obs[idx], h[idx])`, scattered back; that is at most 4 small forward passes per side (stacking the slot weights with `torch.func.stack_module_state` + `vmap` is a later option if profiling shows it matters). Hidden states live in one `[rows, H]` tensor per side (one hidden size per side within a league, checked on load) and are cleared at episode end. Scripted slots: one bot instance per VecEnv group acts on all envs of the group each step and only the rows assigned to the slot are used (the bots are vectorized numpy and keep per-env waypoints, which reset harmlessly). **Leases:** every `lease_iters = 5` iterations, the oldest non-scripted slot starts draining: no new episodes are assigned to it, and when its last episode ends it is reloaded with a new PFSP sample from the pool (4.8.5). On S0, a slot drains in at most 600 decisions (about 5 iterations). An opponent therefore never changes within an episode, so win-rate bookkeeping and hidden states stay consistent. Episode start: a slot is drawn with probability proportional to the PFSP weight of its entry among the non-draining slots. Slot 0 of each side is pinned to the scripted bot with a fixed 20% share of the B/C episodes (scripted bots are always in the pool; this also keeps the anchors of the gates trained against).

**Co-training loop (`zai_rl.train_league`, one process, one GPU).** `HumanPPO.collect` and `ZombieMAPPO.collect` are split into `begin(T)`, `act(batch) -> (actions, aux)`, `record(step, aux)` and `finish(next_batch) -> rollout` (GAE, truncation bootstrap) over preallocated buffers. The existing `collect` methods become a loop over these four calls (same results, checked by test). Per iteration: 128 x (`human.act`, `zombie.act`, opponent slots act, `env.step`, both `record`), then `finish` and `update` for both learners sequentially on the GPU, then league bookkeeping (pool statistics, leases, guard, snapshots). Iteration size at E = 1024: human 768 x 128 = 98 k transitions, zombies 0.75 x 1024 x mean count x 128 (590 k at 6 zombies). `update()` gains `epochs` and `lr_scale` overrides for the imbalance guard. `total_steps` counts env steps over all groups; each learner logs its own transition count.

**Checkpoint layout.** Learner checkpoints stay where the existing tools look for them, and league state is separate:

```
/checkpoints/human/<run_id>/{latest.pt, itNNNNNN.pt}    # HumanPPO.state_dict + extra (as today, + obs_version, stage)
/checkpoints/zombie/<run_id>/{latest.pt, itNNNNNN.pt}   # ZombieMAPPO.state_dict + extra
/checkpoints/league/<run_id>/
  league.pt              # iteration, steps, stage index, pool statistics, slot leases, guard state, RNG states
  pool.json              # snapshot index (also read by the arena)
  pool/human/h0007.pt    # actor only: {actor, hidden, obs_version, side, steps, stage, source}  (~2.8 MB)
  pool/zombie/z0007.pt   # (~0.7 MB)
  crossplay.jsonl        # arena results, append-only
<ZAI_DATA_DIR>/leagues/<run_id>/{pool.json, arena.json}   # copies for the viewer (4.8.6)
```

`--resume /checkpoints/league/<run_id>/league.pt` restores both learners from their `latest.pt`. Exported ONNX models need distinct ids because both sides share the run id: `<run_id>-human` and `<run_id>-zombie` (`export_training_model` currently uses `run_id`, so the two would overwrite each other). Training replays (`record_training_replays`, `match_id = run_id`) play the latest human against the latest zombies.

#### 4.8.5 Snapshot pool and PFSP

- **Entries:** `{id, side, path, steps, stage, source: league | distilled | scripted | random, protected, created}`. Scripted bots are entries without a path. Random bots are Elo-only (floor), never sampled for training.
- **Snapshots:** each learner's actor every `snapshot_every_steps` (20 M on S0/S1, 10 M on `s0_small`; at ~9 k env steps/s that is one every 20-35 min), and always at a stage end. Seed entries per stage: the scripted bot, the stage starter, and the previous stage's final snapshot (on `s0_small`, the four distilled teachers).
- **Eviction:** cap of 50 per side; protected entries are scripted bots, seeds, stage finals and every 10th snapshot; when over the cap, the oldest unprotected entry is evicted (its file is kept on disk, but it no longer appears in `pool.json` for sampling).
- **Win-rate bookkeeping** (trainer side, from group B/C training episodes): score 1 for a learner win, 0.5 for a starvation draw, 0 for a loss (same scoring as Elo, section 7 #16). Each `(learner side, entry)` pair holds decayed sums `n <- 0.98 n + games`, `w <- 0.98 w + score` per iteration, and the estimate is `p = (w + 1) / (n + 2)` (prior 0.5). This measures the current stochastic learner against the entry under training conditions; Elo (4.8.6) is the fixed-seed measurement.
- **PFSP weight** when a lease is renewed: `f(p) = (1 - p)^2` ("hard" PFSP, AlphaStar), mixed with 10% uniform over the stage's entries. Section 4.3's `p (1 - p)` gives zero weight to entries the learner never beats, which is exactly the forgetting case the pool is meant to catch; it stays available as `pfsp: variance` for ablation.
- **Imbalance guard** (4.3, made concrete): every 10 iterations compute `s_z` over group A episodes. If `s_z > 0.85` for 3 consecutive windows, the human gets +1 update epoch and the zombies' lr is multiplied by 0.5; symmetric if `s_z < 0.15`. The guard is released when `s_z` returns to [0.3, 0.7]. It is logged as `league/guard` (-1, 0, +1). If more than 50% of group A episodes end in starvation for 3 windows, the trainer logs `league/starvation_alarm`, because zombies get no reward for starvation (section 3.8) and the league is drifting towards risk 2 of section 8.

#### 4.8.6 Arena: Elo, exploitability probe, tactic metrics

**Status (2026-11-27): implemented**: `arena/elo.py` (Newton-fitted Bradley-Terry with the single anchor and a Gaussian prior, binomial bootstrap), `arena/league.py` (`Arena`: Elo set, cross-play into `crossplay.jsonl`, `arena.json`, `arena` metrics), `arena/probe.py` (fine-tunes with the single-side trainers in a subprocess, gaps into `probes.jsonl` and `probe/<side>_gap`), the server's `GET /api/leagues{,/{id}}` (`leagues.rs`) and the viewer's league panel (`web/src/league.ts`: cross-play heatmap, pool tables with Elo, PFSP weight and learner score); `make arena-gpu`, `make probe-gpu`. Not yet: probe checkpoints entering the pool as exploiters (the pool belongs to the trainer process), and the gate checklist (M6.5).

**Process.** `python -m arena.league --run-id <id>` runs as a second process in the trainer container (`make arena-gpu`), on the same GPU (small batches) with 3 simulation threads (the league trainer drops to 17). It watches `pool.json`, plays each new snapshot against the **Elo set** of the other side (the last 10 snapshots of the stage, scripted, random, the stage's references) with `arena.matches.play` extended to return per-episode outcomes and tactic rows. That is 128 episodes per training zombie count on fixed seeds per matchup; results are appended to `crossplay.jsonl` and never recomputed. Cost: a new human/zombie snapshot pair needs ~26 matchups, about 2-4 min on S0, so the arena stays idle most of the time between snapshots.

**Elo.** Bradley-Terry maximum likelihood on episode scores (win 1, starvation 0.5, loss 0), on the Elo scale (`400 log10`), fitted over the whole stage table at each update (order-independent, unlike sequential Elo updates), with a weak Gaussian prior (sigma 400) for entries with few games. Cross-play is bipartite (humans only play zombies), so the ratings have a single free offset: **one anchor**, the scripted zombies at 1000. The scripted human's rating is then a measured number (it reflects game balance as much as skill). This refines section 4.5, which anchored both scripted bots at 1000, and that is over-determined. 95% intervals come from 200 bootstrap resamples over episodes. Elo assumes transitivity, so the viewer also shows the raw cross-play matrix, where cycles are visible.

**Exploitability probe.** Run at every gate check and every 200 M env steps. For each side, start from the learner's own latest checkpoint (`init_checkpoint`), fine-tune against the frozen latest opponent for 10 M env steps with the existing single-side trainers (`opponent: policy`, lr 1e-4, configs `configs/train/probe_{human,zombie}.yaml`), then compare: gap = probe score - learner score against that opponent, on the arena's fixed seeds. A from-scratch best response is not feasible at this budget (M4 needed 18 M steps just to beat the scripted baseline). Cost: ~15-25 min per side. During a probe the league trainer runs with 10 simulation threads (or pauses at gates). The probe's final checkpoints go into the pool as protected `source: probe` entries (exploiters, as in AlphaStar).

**Tactic metrics (subset of 4.5).** Status (2026-11-27): the P1 counters are implemented (`crates/zai-sim/src/stats.rs`, `STATS_LAYOUT` with 15 values, `VecEnv` `info["final_stats"]`, `Env.episode_stats()`, one rule test each in `src/tests/stats.rs`) and aggregated by `arena.tactics.tactic_metrics` from `arena.matches.play_detailed`; P2 and the arena/viewer wiring are still to do. A risky pickup is one heard by a zombie (within `min(pickup radius, hearing radius)`), and catch geometry is measured at the catching tick. Counters are accumulated in Rust during `Env::step`, so they are free for every training episode and identical in the server. They live in a new `crates/zai-sim/src/stats.rs`: `EpisodeStats`, exported with a `STATS_LAYOUT` table like the obs layouts, `Env::episode_stats()`, and `VecEnv` `final_stats [E, K]` (rows valid where an episode ended, K ~ 24). Python aggregates them (means, percentiles, Wilson or bootstrap intervals). Each counter gets a Rust rule test on an arranged state.

| Priority | Side | Metric | Where |
|---|---|---|---|
| P1 | Human | Survival time p10 / median / p90, by cause | Python (outcome, length) |
| P1 | Human | Food: packages picked, eaten from inventory, mean carried, fraction of time with `carried >= 3` | Rust |
| P1 | Human | Noise: events per minute by type; risky pickups (a zombie within 100 m, i.e. one hears it) per episode | Rust |
| P1 | Human | Doors: closes per episode; defensive closes (a zombie within 30 m of the door at close time) | Rust |
| P1 | Human | Fraction of time inside houses | Rust |
| P1 | Zombies | Catch rate and time to catch per count (exists) | Python |
| P1 | Zombies | Encirclement at catch: maximum angular gap of the zombies within 50 m of the human at the catching tick (mean), and the fraction of catches with gap < 120 degrees | Rust |
| P1 | Zombies | Flanking at catch: distinct 8-sectors of the 3 nearest zombies at the catching tick | Rust |
| P1 | Zombies | Search dispersion: mean pairwise distance over decisions in which no zombie heard anything in the last 30 s | Rust |
| P2 | Human | Escapes through the opposite door within 20 s of a close; zombie distance when entering a house | Rust |
| P2 | Zombies | Noise response (fraction of hearers that close >= 5 m on the source within 10 s, i.e. responders vs sentinels); door coverage; flanking over the 30 s before a catch (ring buffer) | Rust |

Baselines for every metric: scripted vs scripted and random, measured at stage start (`tactics/baseline_*`).

**How results reach the viewer.**
- Scalars: the trainer posts under `run_id` with models `human` and `zombie` (two `MetricsLogger`s); the arena posts with model `arena` (already accepted by `metrics.rs`) under the same `run_id`, at the snapshot's env step, so one run in the viewer shows all three. No schema change in `metrics.rs`.
- Tables that are not time series (snapshot list with Elo and intervals, cross-play matrix, PFSP weights, gate checklist, stage history): `pool.json` (trainer) and `arena.json` (arena) under `<ZAI_DATA_DIR>/leagues/<run_id>/`, served merged by a new `GET /api/leagues` and `GET /api/leagues/{id}` in `api.rs`, mirrored in `web/src/types.ts`.
- Web: `charts.ts` gets `/gate/`, `/anchor/`, `/elo/` and `/tactics/` at the front of `ORDER`; the key conventions below reuse the existing `baseline_`, `_lo` / `_hi` and `_zN` grouping. A new `web/src/league.ts` panel (shown when `/api/leagues/{id}` exists for the selected run) draws the cross-play heatmap (human snapshots x zombie snapshots, cell = human score), the snapshot table, the PFSP weights and the gate checklist.

**Metric keys** (new; the existing `train/*`, `episode/*` and `perf/*` stay):

| Model | Keys |
|---|---|
| `human`, `zombie` (trainer) | `episode_A/{caught,starved,survived}`, `league/{stage,guard,starvation_alarm}`, `pool/{size,pfsp_entropy,score_min,score_mean}`, `tactics_train/<metric>`, `perf/{sim_s,opponent_s,policy_s,glue_s,update_gather_s,update_step_s}` |
| `arena` | `elo/human`, `elo/zombie` (+ `_lo`, `_hi`), `elo/baseline_human` (scripted human, drawn on the `elo/human` chart), `anchor/survived` and `anchor/caught` (+ `baseline_`, `_lo`, `_hi`, `_zN`), `xplay/{human,zombie}_score_vs_old`, `probe/{human,zombie}_gap`, `gate/<condition>` (margin to the threshold, > 0 passes) and `gate/passed`, `tactics/<metric>` (+ `baseline_`, `_lo`, `_hi`) |

#### 4.8.7 Reuse and new code

| Piece | Existing code | M6 action |
|---|---|---|
| Env groups, fixed opponents | `envs.HumanTrainingEnv`, `envs.ZombieTrainingEnv`, `group_sizes` | New `MatchTrainingEnv` (4.8.4); the two classes become wrappers, with a trajectory-equality test. |
| Rollout loops | `HumanPPO.collect`, `ZombieMAPPO.collect` | Split into `begin/act/record/finish` over preallocated buffers; `collect` kept as a loop over them (equality test). |
| Updates | `HumanPPO.update`, `ZombieMAPPO.update` | Unchanged except `epochs` / `lr_scale` overrides; `ZombieCritic` gets the factorized first layer (weight-compatible). |
| GAE, BPTT, `RunningMeanStd`, `save_checkpoint` | `algos/common.py` | Reused; `save_checkpoint` adds `obs_version`. |
| Frozen policies | `policy.RecurrentHumanPolicy`, `RecurrentZombiePolicy` (`from_checkpoint`) | Reused for loading; new `league/slots.py` (`OpponentSlots`: resident slots, row indices, hidden states, leases). |
| Config loading | `TrainConfig.load`, `ZombieTrainConfig.load` | New `LeagueConfig` (`human: PPOConfig`, `zombie: MAPPOConfig`, `league:`, `stages:`), same unknown-key check; configs `configs/train/league_{cpu,gpu}.yaml`. |
| Entrypoint | `train.py`, `train_zombie.py` | New `zai_rl/train_league.py`; Makefile `train-league`, `train-league-gpu`, `arena-gpu`, `distill-gpu`. |
| Wilson, per-count catch rates | `train_zombie.wilson`, `catch_rates` | Move to `arena/stats.py`, shared by the trainers and the arena. |
| Batched matches | `arena.matches.play` | Add `play_detailed` returning per-episode outcome, length and tactic rows; `MatchStats` unchanged. |
| Pool, Elo, scheduler | (planned in section 2 as `league/{pool,elo,scheduler}.py`) | New: `league/pool.py` (index, eviction, decayed scores, PFSP), `league/elo.py` (Bradley-Terry fit, bootstrap), `league/scheduler.py` (group layout, guard, gates, stage advance). |
| Arena | `arena/` (balance sweep, export) | New `arena/league.py` (watcher, cross-play, Elo, `arena.json`, metrics) and `arena/probe.py` (launches the single-side trainers). |
| Metrics | `MetricsLogger` | Reused (two trainer instances plus one arena instance). |
| Model export, replays | `export_training_model`, `record_training_replays` | Reused; ids `<run_id>-human` / `<run_id>-zombie`; metadata gains `obs_version`. |
| Observation encoders | `obs.rs` | Layout v2 (4.8.2), `OBS_VERSION`; `layout_tests` updated. |
| Tactic counters | none | New `stats.rs` (`EpisodeStats`, `STATS_LAYOUT`), `VecEnv.final_stats`, `zai_env.layout.STATS`. |
| Vector env | `zai-py` `VecEnv` | `final_stats`; `VecEnv.from_groups` (one call for all groups) after M6.0 confirms the gain. |
| Legacy adapter, distillation | none | New `zai_env/legacy.py` (`adapt_v1`), `zai_rl/distill.py`. |
| Server | `policy.rs` (`obs_dim` check), `api.rs`, `metrics.rs` | `obs_version` check; `GET /api/leagues{,/{id}}`; `metrics.rs` unchanged. |
| Viewer | `charts.ts`, `types.ts` | `ORDER` entries; new `league.ts`; types mirrored. |
| Python bots | `bots/scripted_{human,zombie}.py` | Read meters from the v2 fields (absolute `x, y` now `/500`); a test checks that they choose the same actions as v1 on the same states (through the adapter). |

#### 4.8.8 Acceptance criteria for M6

1. **Layout v2 and starters:** all Rust and Python tests pass; parity fixtures are regenerated and `tests/policy_parity.rs` passes; an existing replay re-simulates to its stored `state_hash`; each distilled starter meets 4.8.2 (outcomes within 3 points of its teacher, KL < 0.05).
2. **Profiling:** `docs/reports/throughput_profile.md` with the per-phase split of the three reference runs and of a league iteration; the budget table above updated with measured numbers.
3. **League mechanics:** one `train_league` process trains both sides with groups A/B/C, slot leases, PFSP and the guard, and resumes from `league.pt` with identical pool state.
4. **Curriculum:** the league passes `G(s0_small)` and `G(S0)` (4.8.1) within their maximum budgets and advances automatically.
5. **Elo improvement against fixed anchors (the row-9 criterion, made measurable):** on S1, the latest snapshot of each side is at least +100 Elo above its S1 starter, with non-overlapping bootstrap intervals, and both anchor conditions of the S0 -> S1 gate hold on S1. Passing `G(S1)` is reported but not required; S1 continues in M8 if needed.
6. **No cycling:** at the end of each stage, `xplay/*_score_vs_old` >= 0.55 for both sides and the probe gap <= 20 points.
7. **Tactics:** on the last stage reached, at least two P1 tactic metrics per side differ from the scripted baseline in the intended direction with non-overlapping 95% intervals (for example: more defensive closes and fewer risky pickups for the human; more encirclement at catch and higher search dispersion for the zombies).
8. **Viewer:** the run page shows the Elo curves with intervals, anchors, gates, tactic charts with baselines, the cross-play heatmap and the snapshot table, live while the league runs.

#### 4.8.9 Staged rollout and de-risking before GPU runs

1. **Before touching `obs.rs`** (CPU, minutes): dump v1 observations, states and actions for 3 episodes per stage config from the current build into `python/tests/fixtures/obs_v1.npz`. After v2 lands, re-simulate the same seeds and actions and check `adapt_v1(v2) == v1` (atol 1e-5 on `s0_small`; door deltas beyond 100 m excepted on S0). This is the only cheap way to prove that the teachers see their own input distribution.
2. **CPU tests (pilot, minutes):** layout v2 tests; wrapper trajectory equality; `act/record/finish` equals `collect`; `MatchTrainingEnv` index round-trips (human rows, flat agents, opponent rows); slot draining never switches an opponent inside an episode; PFSP sampling frequencies match the weights (chi-square); the Elo fit recovers ratings on synthetic Bradley-Terry data within 30 points; decayed score bookkeeping; one Rust rule test per tactic counter; guard state machine; `league.pt` resume round-trip.
3. **CPU pilot run (Apple Silicon, ~30 min):** `league_cpu.yaml`, 128 envs, `s0_small`, 2 M env steps, pool seeded only with scripted bots and snapshots every 0.5 M steps; the arena running alongside. Go if nothing crashes, the pool grows, leases rotate, `arena.json` fills in and the viewer panels render.
4. **GPU profiling (M6.0, ~1 h):** as in 4.8.3, on the current code. Implement the optimizations chosen (decision 5 below) and re-measure before M6.5.
5. **GPU distillation (M6.1, ~1 h):** four teachers, acceptance as in 4.8.2.
6. **Short GPU league (~1-2 h, 30-50 M env steps on `s0_small`):** go if both critics' explained variance > 0.2, `s_z` stays within [0.1, 0.9], PFSP weights are not uniform after 20 M steps, the arena lags by at most one snapshot, throughput is within 20% of the estimate, and neither side's anchor score drops more than 10 points below its starter.
7. **Full run (M6.5, ~1-2 days):** `s0_small` -> S0 -> S1 with gates and probes; stop at any gate that fails at its maximum budget.

#### 4.8.10 Risks

- **S0 catchability against a learned human** (risk 1 of section 8, still open): the S0 zombie gate may not pass even with 4-8 zombies. Fallbacks, in order: `food_near_spawn_count: 3` on S0 (early pickup noise, as on `s0_small`); an intermediate 110 m stage; higher counts. Each is a user decision at the failed gate.
- **Simultaneous learning instability in group A** (both sides chasing each other): mitigated by groups B/C, the guard, and lower learning rates (start at the fine-tune values 1e-4 / 1.5e-4).
- **Cold pool:** early in a stage the pool holds only seeds, so PFSP has little to choose from; the 20% scripted share and the distilled seeds cover this.
- **Distillation fidelity of recurrent teachers:** the teacher's hidden state comes from its own adapted observations, the student's from v2. DAgger keeps the student's state distribution on-policy; if KL stays high, the fallback is retraining with the M4/M5 recipes (~8 h).
- **Arena and probe contention** with the trainer for cores and the GPU: thread budgets are fixed (17 / 3 / 4 torch), and probes run at gates or with the league slowed.
- **Non-transitivity:** Elo hides cycles; the cross-play matrix and the forgetting check are the guard, not Elo alone.
- **Starvation-dominated outcomes** (risk 2 of section 8): visible as `league/starvation_alarm`; zombies get no reward from it, so the league can stall without either side "winning".
- **Throughput:** if the optimizations deliver less than estimated, S1 takes ~27 h; M6 acceptance does not require passing `G(S1)`.

#### 4.8.11 Decisions

Accepted by the user on 2026-11-27: the recommended option of each row below (D1 a, D2 a, D3 a, D4 a with b as an ablation, D5 a after M6.0, D6 a, D7 a, D8 a, D9 a).

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Starting checkpoints after layout v2 | (a) distill the four teachers through the adapter (~1 h GPU); (b) from scratch inside the league; (c) re-run the M4/M5/4.7 recipes on v2 (~8 h); (d) first resume `zombie-s0small-crowd-ft1` to 100 M on v1 (+4 h), then (a) | (a) |
| D2 | Absolute position in the actors | (a) keep `x, y` (`/500`) in the layout but exclude them from the actor input; (b) feed them to the actor; (c) remove them from the layout (bots read the state instead) | (a) |
| D3 | S0 for M6 | (a) zombie counts 4/6/8 and the relative gate of 4.8.1; (b) original 2-4 with the relative gate; (c) original 4.4 gate (> 60% caught) | (a) |
| D4 | PFSP weighting and scripted share | (a) `(1 - p)^2` + 10% uniform, scripted pinned at 20% of B/C episodes; (b) `p (1 - p)` as in 4.3 | (a), with (b) as an ablation |
| D5 | Throughput work before S1 | (a) optimizations 1-4 of 4.8.3 (~2-3 days of development; S1 ~13 h instead of ~27 h); (b) only what M6.0 shows is cheap, accepting ~5 k env steps/s on S1 | (a), after M6.0 confirms where the time goes |
| D6 | Exploitability probe cadence | (a) at every gate check and every 200 M steps, 10 M steps per side; (b) only at gates; (c) every 100 M steps | (a) |
| D7 | Arena placement | (a) second process in the trainer container, same GPU, 3 sim threads; (b) separate CPU-only `arena` container (section 6) | (a) for M6; (b) once the gRPC path lands in M7 |
| D8 | S1 definition | (a) 300 m, 6 houses, 45 food, 75 m spawn, 10 min, counts 8/12/16; (b) section 4.4 as written (30 food, 4-8 zombies) | (a) |
| D9 | v1 ONNX models in `data/models/` | (a) move to `data/models/legacy/` (not listed); (b) delete; (c) keep (the server rejects them on `obs_dim` and logs it) | (a) |

---

## 5. Communication protocols and data flow

### 5.1 Training loop (in-process)

```python
venv = zai_sim.VecEnv(config, num_envs=1024, seed=0, num_threads=20)  # P100 server: 24 cores, 4 left for the trainer process and OS
obs = venv.reset()
# obs: human_obs [E, 584] f32, human_mask [E, 20] bool,
#      zombie_obs [E, N, 252] f32, zombie_valid [E, N] bool,
#      state [E, 1144] f32
obs, rew, term, trunc, info = venv.step(human_actions,   # [E, 2] i64
                                        zombie_actions)  # [E, N] i64
```

- Environments are partitioned into groups with a fixed `N` (zombie count) per group, so tensors are dense per group without ragged indexing; groups are concatenated for inference with padding to `N_max` and a validity mask.
- `step()` releases the GIL, runs `rayon` over envs, writes into preallocated buffers exposed as NumPy views; the trainer pins them and copies to GPU once per step (~3 MB at E=1024, N=10).

Throughput estimate (to be validated in M1 with a `criterion` benchmark):

| Option | Estimated env steps/s | Notes |
|---|---|---|
| In-process Rust `VecEnv` (recommended) | raw sim ~250-300 k on 16 cores (~40-60 us per env decision step incl. 5 ticks and obs encoding; **measured in M1: 9.2 us per decision step with all observations, ~108 k steps/s per core on Apple Silicon, 1.7 us without observations**); end-to-end with inference and PPO updates ~50-100 k | Per decision step and env: ~64 human rays + 16 rays x N zombies against ~20 culled segments each, ~5 k segment tests |
| Remote, batched gRPC (one message per team per step for all envs) | ~20-50 k | ~12 KB of observations per env step, ~12 MB per batch at E=1024; protobuf serialization dominates |
| Remote, per-env WebSocket/gRPC lockstep | ~1-3 k per connection | Round-trip latency (0.3-1 ms) per step kills throughput |
| Python simulation (NumPy, vectorized) | ~10-30 k | Ray casting and door logic are awkward to vectorize |

### 5.2 Live matches and viewer

```mermaid
sequenceDiagram
  participant A as arena / zai_agent (human client)
  participant Z as arena / zai_agent (zombie client)
  participant S as zai-server
  participant B as Browser
  A->>S: Control(stream) Join{match_id, team=HUMAN}
  Z->>S: Control(stream) Join{match_id, team=ZOMBIES}
  B->>S: WS /ws/match/{id}
  S-->>B: MapInit (houses, doors, food, config)
  loop every decision tick (0.5 s sim time)
    S-->>A: ObservationBatch{tick, agents[1]}
    S-->>Z: ObservationBatch{tick, agents[N]}
    A->>S: ActionBatch{tick, actions}
    Z->>S: ActionBatch{tick, actions}
    S-->>B: Snapshot (10 Hz) + events
  end
  S-->>A: EpisodeEnd{outcome}
  S-->>Z: EpisodeEnd{outcome}
```

**gRPC agent API** (`proto/zai/v1/agent.proto`, sketch):

```proto
service AgentService {
  rpc Control(stream AgentMessage) returns (stream ServerMessage);
  rpc ListMatches(ListMatchesRequest) returns (ListMatchesResponse);
}
message AgentMessage {
  oneof msg { Join join = 1; ActionBatch actions = 2; }
}
message ActionBatch { uint64 tick = 1; repeated AgentAction actions = 2; }
message AgentAction { uint32 agent_id = 1; repeated int32 action = 2; }
message ServerMessage {
  oneof msg { ObservationBatch obs = 1; EpisodeEnd end = 2; }
}
message ObservationBatch {
  uint64 tick = 1;
  uint32 obs_dim = 2;
  repeated uint32 agent_ids = 3;
  bytes obs = 4;            // little-endian f32, [len(agent_ids), obs_dim]
  bytes action_mask = 5;    // packed bools, may be empty
  repeated float reward = 6;
}
```

Observations are raw `f32` bytes produced by the same Rust encoder as training, so there is no schema drift. Match modes: `lockstep` (server waits for both teams; used for evaluation, runs as fast as clients answer) and `realtime` (configurable speed x1-x16; if actions miss a 200 ms deadline the previous action is repeated).

**As implemented in M3** (differences from the sketch below): live server matches are played by bots ported to Rust (`zai_sim::bots`, behaviorally similar to the Python bots, used only for server matches and demos) until the gRPC agent API lands in M7; one `snapshot` frame is sent per decision step (2 Hz sim time, i.e. 32 frames/s at x16) and the browser interpolates positions between them instead of 10 Hz frames; snapshots carry the full list of active food (about 1-2 KB per frame at 20 zombies / 120 food) instead of diffs; speed is x0.25-x16 or 0 for unthrottled (tests, evaluation); replays are JSON files `data/replays/{match}-e{episode}.json` with the action log and the final `state_hash`, re-simulated for playback and seek. HTTP API: `GET/POST /api/matches`, `GET /api/matches/{id}`, `POST /api/matches/{id}/control` (`speed`, `paused`, `stop`, `auto_restart`), `GET /api/replays`, `GET /api/replays/{id}`, `GET/POST /api/metrics`, `GET /api/runs`, `GET /api/metrics/stream` (SSE), `GET /api/config/default`, WebSockets `/ws/match/{id}` and `/ws/replay/{id}` (JSON commands `speed`, `pause`, `play`, `seek`).

**Viewer WebSocket** (`/ws/match/{id}`, MessagePack):
- `MapInit` once: map size, houses (pos, rotation, door segments), food positions, config.
- `Snapshot` at 10 Hz: `tick`, human `{x, y, heading, carried, hunger}`, zombies `[[id, x, y, heading]]`, `door_states` bitset (20 bits), food removed since last snapshot, noise events since last snapshot `[[type, x, y, radius]]`. Under 1 KB per frame at 10 zombies.
- Views: top-down (PixiJS: houses, doors, food, entities, vision circle, noise rings, zombie hearing radii on hover) and 3D first-person (Three.js: extruded houses from the same segments, billboard zombies, camera at the human position/heading, fog at 100 m to match vision).
- Replays: every server match is stored as `(config, seed, action log)` (tens of KB); the server re-simulates them deterministically for playback and scrubbing.

### 5.3 Metrics

- Trainer and arena push scalars every iteration/evaluation: `POST /api/metrics` with `{run_id, model: "human"|"zombie"|"arena", step, wall_time, scalars: {...}}`. The server stores them in SQLite (`/data/metrics.db`).
- Viewer reads history with `GET /api/metrics?run_id&model&keys&since` and live updates with SSE `/api/metrics/stream`; charts with uPlot (reward, episode length, losses, entropy, KL, Elo, tactic metrics).
- The trainer also writes TensorBoard event files to `/runs` for deep debugging (optional `tensorboard` compose service). W&B is an optional sink, off by default.
- Training environments never stream state to the viewer; only server matches do.

---

## 6. Infrastructure and containers

| Service | Image | Notes |
|---|---|---|
| `server` | multi-stage: `rust:<pinned>-bookworm` builds `zai-server`; `node:<pinned>-slim` builds `web/`; runtime `debian:bookworm-slim` | Ports 8080 (HTTP/WS/SSE), 50051 (gRPC). Volume `data`. |
| `trainer` | multi-stage: Rust builder runs `maturin build --release` to produce the `zai_sim` wheel; runtime `python:3.12-slim-bookworm` + `torch==2.8.0` from the PyTorch index selected by the `TORCH_INDEX` build arg (`cpu` for the pilot, `cu126` for the P100 server; the official `pytorch/pytorch` images are amd64-only, which would force emulation on the Apple Silicon pilot machine) + wheel + `python/` via `uv` | GPU through `compose.gpu.yaml` (`deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`). `shm_size: 8gb`, `cpus` set to the host core count. Volumes `checkpoints`, `runs`. |
| `arena` | same Python image as trainer, CPU-only by default | Reads `checkpoints`, talks to `server` over gRPC, posts metrics. |
| `tensorboard` (optional profile) | trainer image | Port 6006, reads `runs`. |
| `dev` | toolchain image (Rust, Python, Node) | `make test`, `make bench`, `make lint` run inside it; nothing installed on the host. |

- Host prerequisites: Docker (with Compose v2). For GPU training additionally a Linux host with the NVIDIA driver and the NVIDIA Container Toolkit; these cannot live inside a container. Without them, `docker compose up` runs everything CPU-only (useful for development and viewer work).
- Where the GPU helps: batched policy inference (11 k agent observations per step at E=1024, N=10) and PPO/MAPPO updates over ~1.4 M transitions per iteration. Where it does not: the simulation, which is CPU-bound; scaling is by CPU cores and `num_envs`. With these small networks a single mid-range GPU will be underutilized; the CPU core count is the main purchase/allocation decision.
- Target GPU: NVIDIA Tesla P100 (Pascal, compute capability 6.0, 16 GB HBM2, no Tensor Cores, no bf16). Consequences: train in FP32 (no bf16 autocast; FP16 AMP gives little benefit for these small MLP/GRU networks); 16 GB is ample for the planned batch sizes. Recent CUDA toolchains are dropping Pascal (CUDA 13 removes it, and some newer PyTorch CUDA wheels no longer ship `sm_60` kernels), so the trainer image must pin a PyTorch build whose CUDA wheels still include `sm_60` (e.g. a `cu126` build) and the host driver must support that CUDA version. Verify in M0 with `torch.cuda.get_arch_list()` containing `sm_60` inside the container.
- Target server sizing (24 CPU cores, 64 GB RAM): `num_threads=20` for the simulation, `num_envs=1024` as the starting point (try 2048 if the GPU update step, not the simulation, dominates iteration time). A 128-step rollout at E=1024, N=20 (the default zombie count) holds ~6.8 k floats per env step (human obs 584 + 20 x 252 zombie obs + 1144 critic state), about 3.5 GB in FP32; E=2048 is ~7.1 GB. Both fit comfortably in 64 GB host RAM and 16 GB GPU memory; keep `shm_size: 8gb`.
- Pilot: the same images run CPU-only on macOS (no GPU override), with `num_envs` 128-256 and stages S0/S1 only. The PyTorch pin above also has CPU wheels, so the pilot and the P100 server use the same versions.
- If the CPU simulation becomes the bottleneck later, the fallback is a GPU simulation (JAX or NVIDIA Warp) of the same rules, validated against the Rust core with replay hash tests. Not planned initially.

---

## 7. PRD ambiguities and recommended defaults

| # | Ambiguity | Recommended default |
|---|---|---|
| 1 | Human vision radius | **Resolved:** 100 m. Default: 360 degrees, occluded by walls/closed doors. |
| 2 | Food speed penalty | **Resolved:** linear on base speed, zero at 5+ packages. |
| 3 | How eating works | Automatic consumption from inventory when the 90 s timer expires; starvation if inventory is 0. |
| 4 | Food pickup automatic or explicit | Explicit `pickup` action within 1.5 m (makes the noise a decision, avoids the 5-package freeze trap). |
| 5 | Food respawn | **Resolved:** on. Each picked package respawns at a random position after `food_respawn_interval_s` (default 60 s, tunable in M2). |
| 6 | Initial human state and spawn | Hunger 90 s, inventory 0, at least 150 m from every zombie. |
| 7 | Noise radius vs 300 m hearing radius | Heard iff distance <= `min(event_radius, 300)`; walls do not attenuate; exact source position. |
| 8 | Walking noise frequency | One event per decision step (0.5 s) with movement. |
| 9 | What zombies perceive besides noise | Static map layout (houses and door positions, not door state), 10 m proximity rays, 8 nearest peers within 50 m. **Resolved:** option (b). |
| 10 | Zombie communication | None by default; learned local message as an ablation. |
| 11 | Door initial state, opening, blocking | Both open at start; human closes within 2 m; cannot close if the doorway is occupied; doors are never opened directly (only by closing the twin). |
| 12 | House size and "uniform" placement | 12 x 8 m; stratified 5 x 2 grid with +/-30 m jitter and random 90 degree rotation. |
| 13 | Zombie-zombie collisions | Soft push, no hard blocking. |
| 14 | Map boundary | Solid. |
| 15 | Episode length | 20 min cap as truncation for training. **Resolved:** reaching the cap counts as a human win in evaluation. |
| 16 | Starvation outcome for zombies | Not a zombie win (no reward), draw in Elo. |
| 17 | "Models communicate with the game server" | Training runs the same Rust core in-process (throughput); evaluation and live matches use the server's gRPC agent API. **Resolved:** accepted. |
| 18 | "Trainers" plural | One trainer container hosting two independent learners (separate nets, optimizers, checkpoints, metrics). |
| 19 | Human action set | Move (16 directions or stay), pickup, close door. No running, no drop, no attack. |
| 20 | Zombie count variable | `ZAI_ZOMBIE_COUNT` (default 20) for evaluation/live matches; training samples a range around it. `ZAI_FOOD_COUNT` default 120. **Resolved after the M2 sweep (was 10 / 40).** |

---

## 8. Risks and open questions

1. **Game balance with PRD numbers (highest risk).** Zombies walked 0.5 m/s in the original PRD (now 0.7 m/s): crossing the 500 m map took ~17 min (now ~12 min). The loudest event is 200 m (door), pickups are 100 m, walking 10 m. With 10 zombies and 40 food (one package per ~6250 m^2, mean spacing ~80 m), a careful human may rarely be heard and zombies may almost never catch it, giving them no learning signal. Conversely, with few food packages, starvation dominates and the human's outcome barely depends on zombies. Mitigation: a balance sweep with scripted bots (M2) over zombie count (5-50) and food count (20-120) before RL; pick defaults where scripted zombies catch the scripted human in 20-60% of episodes. The curriculum gives early signal on small maps.
   **M2 result at the original 0.5 m/s (`docs/reports/balance_sweep_zombie_0.5mps.md`, 64 episodes per cell, zombies 5-64 x food 20-120, respawn 60 s): scripted zombies caught the scripted human in 0-1.6% of episodes in every cell; outcomes are decided by starvation (survival to the 20 min cap rises with food, from ~0-5% at 20 food to ~80% at 120 food with 5 zombies; more zombies mean more fleeing and more starvation). The same holds on the S0 map (0% caught). Catching works mechanically (scripted zombies catch a noisy random human spawned nearby in ~17% of episodes vs ~2% for random zombies), so the cause is the 2:1 speed ratio plus the 10 m walking noise: a human that flees at sight cannot be run down, only ambushed. No zombie/food count inside the allowed levers reaches the 20-60% target; settling this needs a decision on the PRD constants or the curriculum (open question below).**
   **Re-run at 0.7 m/s (`docs/reports/balance_sweep_reserve{1,2}.md`; scripted human keeping 1 or 2 spare packages): still 0-3% caught in every cell. A quick lever check (10/20 zombies, 80 food) with zombie speed 0.9 m/s and walking noise 25-50 m also stayed at 0-3% caught; stronger zombies only raise starvation, because the scripted human flees at sight and stops foraging. Conclusions: (1) the scripted zombies (chase the last noise) are too weak to measure catchability, since catching a faster human with 360-degree 100 m vision needs ambush or encirclement, which only learned zombies can show; (2) zombie pressure is currently converted into human starvation, which gives zombies no reward (section 3.8). Superseded by the 120-degree decision below. Previously proposed: keep 10 zombies / 40 food, evaluate catch rate with learned zombies in M5 (with S0-S1 shaping), and revisit whether starvation under pressure should reward zombies if the M5 catch rate stays near zero.**
   **Experiment, 45-degree human field of view (`docs/reports/balance_sweep_fov45.md`, zombies 0.7 m/s, reserve 2): catches finally appear and grow with zombie count and food (up to 11-13% at 35-50 zombies and 20% at 64 zombies / 120 food), but starvation becomes dominant (75-100% in every cell; survival to the cap at most 20%), because the scripted human finds far less food and has no "look around" action (heading only changes by moving). Not adopted; the PRD keeps 360-degree vision unless the user decides otherwise.**
   **Experiment, 120-degree field of view (`docs/reports/balance_sweep_fov120.md`, same settings): a middle ground. Catches of 6-12.5% appear already at 10-20 zombies with 80-120 food and reach ~20% at 50-64 zombies; the best mix is 20 zombies / 120 food (12.5% caught, 48% starved, 39% survived). Starvation is still the most common outcome in every cell (48-100%), less dominant than at 45 degrees. **Adopted by the user (2026-11-24): 120-degree field of view, 20 zombies, 120 food as defaults (PRD updated).**
2. **Degenerate equilibrium: silent human vs wandering zombies.** If silence dominates, zombies converge to random search and the human to "stand still, pick up only when needed". Detection: noise events per minute -> 0 and zombie search dispersion flat. Mitigations: food respawn (forces movement), larger zombie counts, or a small probability that zombies hear walking up to 20 m.
3. **Non-stationarity and strategy cycling.** Addressed by the PFSP opponent pool and exploitability probes; still the main source of training instability.
4. **Long-horizon credit assignment.** Starvation consequences arrive ~180 decisions later. Mitigations: `gamma 0.995`, recurrent policies with 64-step BPTT, hunger timer in the observation. Fallback: macro-actions (e.g. "go to house k", "go to food j") if flat actions plateau.
5. **Zombie sparse reward.** Shaping only in S0-S1; if catch rate in S2 stays near zero, the balance parameters (risk 1) are the lever, not more shaping.
6. **Host GPU.** The pilot runs on macOS, where Docker cannot access the Apple GPU, so pilot training is CPU-only with a small `num_envs` (128-256) and is limited to curriculum stage S0/S1. Full training moves to a Linux server with an NVIDIA Tesla P100 and the NVIDIA Container Toolkit. The P100 constraints and server sizing (24 cores, 64 GB RAM) are described in section 6; CPU cores remain the main throughput factor.
7. **Determinism.** `f32` results are deterministic only for the same binary and CPU architecture; replays store the build hash. Parallelism is across envs only, never inside an env.
8. **Build complexity of PyO3 wheels in Docker.** Contained in a multi-stage build; validated in M0/M2.

**Open questions for the user**

Q1-Q6 were answered on 2026-11-24 (see the header), as was the server sizing (24 CPU cores, 64 GB RAM). Open:

- Q7 was answered: zombie speed is now 0.7 m/s (PRD changed). See risk 1 for the re-run sweep; see risk 1 for the re-run sweep. After the field-of-view experiments the user adopted a 120-degree human field of view with 20 zombies and 120 food as defaults (2026-11-24); catch rates are re-evaluated with learned zombies in M5.

---

## 9. Incremental implementation plan

Each milestone ends with something runnable through `docker compose` / `make` and covered by tests.

| Milestone | Deliverable | Runnable / testable result |
|---|---|---|
| **M0 Scaffold** | Repo layout, Cargo workspace, `pyproject.toml`, `dev` image, `Makefile`, `compose.yaml`, CI running `make test` | `make test` passes in a container on an empty skeleton. |
| **M1 Simulation core** | `zai-sim`: map generation, movement, collisions, doors, food, hunger, noise routing, LOS, obs/state encoding, rewards, termination | Unit tests per rule (door twin logic, speed formula, noise ranges, starvation, contact death); property tests (no entity inside walls); determinism test (same seed + actions -> same state hash); `criterion` benchmark reporting env steps/s. Headless CLI runs random agents. |
| **M2 Python env + scripted bots + balance sweep** | `zai-py` `VecEnv`/`Env`, `zai_env` PettingZoo/Gymnasium wrappers, scripted human/zombie bots, sweep script | PettingZoo `parallel_api_test` passes; Rust-vs-Python obs parity test; sweep report (catch rate and survival by zombie/food count) used to settle defaults (risk 1). |
| **M3 Server + top-down viewer** | `zai-server` match manager, viewer WebSocket, replay storage, metrics API; `web` top-down view | Scripted-bot match visible live in the browser at x1-x16; replay playback; metrics endpoint accepts and serves test data. |
| **M4 Human training** | `zai_rl` recurrent PPO for the human vs scripted zombies (S0), trainer image with GPU override, TensorBoard + metrics push | Human survival beats the scripted human baseline on S0; curves visible in the viewer. |
| **M5 Zombie training** | MAPPO with shared params and centralized critic vs scripted human (S0) | Zombie catch rate beats the scripted zombie baseline on S0. |
| **M6 Co-training + arena** (plan in section 4.8) | M6.0 throughput profiling; M6.1 observation layout v2 (constant normalization) with distilled starting checkpoints; M6.2 Rust episode tactic counters; M6.3 `MatchTrainingEnv` with env groups A/B/C in one league trainer, snapshot pool with PFSP and imbalance guard; M6.4 arena with cross-play Elo, anchors, exploitability probe and the viewer league panel; M6.5 curriculum gates `s0_small` -> S0 -> S1 | League passes the `s0_small` and S0 gates and advances automatically; on S1 both sides gain >= +100 Elo over their S1 starters against fixed anchors, without cycling (section 4.8.8); Elo, cross-play, gate and tactic metric panels in the viewer. |
| **M7 gRPC agents + 3D view** | `agent.proto`, `zai_agent` client, server agent API, Three.js first-person view | Arena drives a live server match with two checkpoints over gRPC; obs parity with training verified by a hash test; 3D view follows the human. |
| **M8 Full spec runs** | S2 training at PRD values, long runs, ablations (zombie communication, noise sigma, respawn) | Report on emergent tactics using section 4.5 metrics vs baselines. |
