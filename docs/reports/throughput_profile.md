# M6.0 throughput profile (2026-11-27)

Profiling of the two current trainers on the GPU server (Tesla P100 16 GB, 24 cores), as
planned in `docs/ARCHITECTURE.md` section 4.8.3. Configs: `configs/train/profile/*.yaml`, which
are the reference GPU configs (`human_s0_gpu`, `zombie_s0small_gpu`,
`zombie_s0small_crowd_ft_gpu`) cut to 12 iterations of 131 072 env steps (1024 envs x 128
decisions), with `profile_sync: true` (CUDA synchronized at the end of every phase, so the split
is exact) and a `torch.profiler` trace of iteration 8. Medians exclude iterations 1, 2 (warm-up)
and 8 (traced). A Stable Diffusion container held 5.5 GB of GPU memory and stayed idle during
the runs, so about 10 GB were available to the trainer.

## Before: per-phase split of one iteration (seconds)

| Phase | Human, S0, 2-4 zombies | Zombies, `s0_small`, 2-4 | Zombies, `s0_small`, 8-10 vs frozen human |
|---|---|---|---|
| `sim` (inside `VecEnv.step`) | 0.40 | 0.58 | 0.91 |
| `opponent` (bots or frozen policy) | 0.20 | 0.25 | 2.38 |
| `policy` (learner forward, host/device copies) | 1.31 | 1.51 | 2.90 |
| `glue` (rest of the rollout) | 0.46 | 0.71 | 1.07 |
| **rollout** | 2.32 | 3.05 | 7.46 |
| `update_gather` (critic state gather) | - | 0.06 | 0.18 |
| `update_step` | 2.78 | 2.55 | 9.31 |
| **update** | 2.78 | 2.62 | 9.49 |
| **iteration (wall, without eval)** | 5.10 | 5.80 | 17.28 |
| env steps/s | 25.7 k | 23.2 k | 7.6 k |
| eval, every 10 / 20 / 20 iterations | 3.6 | 12.3 | 35.7 |
| eval share of wall time at those intervals | 6.6% | 9.6% | 9.4% |

Utilization while training (1 s samples): GPU SM busy 73% / 46% / 54% on average, CPU user time
12-14% of the 24 cores. The trainers are **GPU-bound**: the simulation is 8-11% of an
iteration, and the 20 simulation threads are idle most of the time.

## Finding: the GRU with resets dominated the crowd update

`models/encoders.run_gru_segments` ran the GRU once per span between time steps at which *any*
row of the minibatch restarts an episode. A crowd minibatch has ~2 300 sequences of 64
decisions, so nearly every step is a cut: the trace shows ~19 GRU calls per minibatch, 966 calls
of cuDNN's persistent GRU kernel (3.75 s) per iteration, plus the slicing and concatenation in
the backward pass (0.93 s of `SliceBackward`). The cost grows with the number of episode
restarts, which also explains why the historical runs were slower than this profile at the same
config (`zombie-s0small-gpu1`: median update 4.9 s over the whole run against 2.6 s here; as
training progressed, episodes got shorter and restarts more frequent; the crowd fine-tune: 13.0 s
against 9.5 s).

The fix is exact: every episode segment becomes its own row, left-aligned and zero-padded to the
sequence length, with `h0` for a row's first segment and zeros for the later ones, and the GRU
runs once per minibatch over `B + restarts` rows. `tests/test_encoders.py` checks outputs and
gradients against the previous implementation, and the existing `sequence` vs `step` tests pass.

## After the packed GRU

| | Human, S0 | Zombies, 2-4 | Zombies, 8-10 |
|---|---|---|---|
| update | 2.78 -> 2.59 | 2.62 -> 2.58 | 9.49 -> **6.78** |
| iteration (wall) | 5.10 -> 4.89 | 5.80 -> 5.83 | 17.28 -> **15.36** |
| env steps/s | 25.7 k -> 26.8 k | 23.2 k -> 22.8 k | 7.6 k -> **8.8 k** (80 k agent steps/s) |
| GPU memory peak | | | 7.9 GB (8.7 GB at most) |

The crowd run needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (now set for the trainer
service in `compose.yaml`): without it, the variable-size GRU batches fragmented the allocator
and one run ran out of memory with ~10 GB available.

## Cost model for co-training

From the two zombie runs (3 072 and 9 215 learner agents), excluding the opponent: each learner
zombie agent costs **1.21 ms per iteration** (rollout forward plus update), on top of a fixed
**1.8 s** (simulation and glue). The human learner costs 3.75 s per iteration on 1024 envs
(policy plus update), so 2.8 s on the 75% of envs where it learns (groups A and B). Frozen
opponents in groups B and C: ~0.5 s. Per 131 k env-step iteration, with groups A/B/C at
50/25/25%:

`t = 1.8 + 2.8 + 1.21 ms x 0.75 x 1024 x mean_zombies + 0.5` (seconds, without eval)

| Stage | Mean zombies | Env steps | 4.8.3 estimate | Measured model | Time |
|---|---|---|---|---|---|
| `s0_small` | 6 | 200 M | ~8.9 k/s, ~6 h | ~12.2 k/s | ~4.5 h |
| S0 | 6 | 300 M | ~8.9 k/s, ~9.4 h | ~12.2 k/s | ~6.8 h |
| S1 | 12 | 500 M | ~5.2 k/s, ~27 h | ~8.1 k/s | ~17 h |
| S2 (M8, reference) | 18 | 1-3 B | ~3.7 k/s, 3-9 days | ~6.0 k/s | 2-6 days |

M6 (three stages) comes to about 28 h at the measured throughput with the packed GRU, against 42
h in the plan, before any of the remaining optimizations.

## Remaining optimizations, re-ranked by the measurements

1. **Factorized zombie critic first layer** (4.8.3 item 1). The large-K GEMMs of the critic's
   1396-input first layer are ~1.0 s of the crowd update (`sgemm_largek`), plus its forward
   pass; expected saving ~1-1.5 s per crowd iteration, exact.
2. **Evaluation out of the training loop** (item 4): 6-10% of the wall time today; the arena
   takes it over in M6.4.
3. **Opponent inference batched across groups** (league slots, 4.8.4): the frozen human cost
   2.4 s per crowd iteration (16%) because it runs once per zombie-count group; slots run one
   forward pass per resident snapshot over all of its rows.
4. **One Rust call for all groups and preallocated rollout tensors** (items 2 and 3): `sim` plus
   `glue` are 1.0-2.0 s per iteration (13-17%); worth it for the league's 9 groups.
5. **CUDA graphs for the rollout forward passes** (item 6): the GPU is only 46-73% busy, and the
   learner forward is 23 ms per step for 9 215 agents, much of it launch overhead in small
   encoder layers.
6. Zombie update subsampling (item 5) is not needed for M6 at these numbers.

Decision D5 (optimizations before S1) stands; with the packed GRU done, items 1-3 cover most of
what is left.
