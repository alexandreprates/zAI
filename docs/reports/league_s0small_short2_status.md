# league-s0small-short2: results and review (2026-11-28)

The second short GPU league was the validation run of step 3 after the zombie-learner and
throughput changes (commits `fc4fef1` and `c7e2578` on branch `m6`, docs/ARCHITECTURE.md
section 4.8.9). It was interrupted once because the server was needed for something else, then
resumed and finished.

## Run

- Config: `configs/train/league_s0small_gpu.yaml` with `--total-steps 40000000` (stage
  `s0_small` only, no curriculum gate). Zombie learner: lr 3e-4, 4 epochs. Guard: 0.3 / 0.7
  for 2 windows, released in [0.35, 0.65].
- First part: 2026-11-27 21:43 to 22:39, stopped after the checkpoint at iteration 220
  (28 835 840 env steps). The trainer was stopped before the iteration-240 checkpoint, so the
  resume restarted from 220.
- Resumed 2026-11-28 10:20 with
  `--resume /checkpoints/league/league-s0small-short2/league.pt` and finished at 10:42
  (iteration 306, 40 108 032 env steps, budget reached, no errors). The arena ran alongside
  and was stopped after the last rating round (10:31); no snapshot was taken after 30.3 M
  steps.
- Throughput: ~9.7 k env steps/s (7.7-10.3 k including evaluations).
- Outputs, in the `checkpoints` volume: `/checkpoints/league/league-s0small-short2/`
  (`league.pt`, `pool.json`, `pool/` with h0001-h0003 and z0001-z0003, `crossplay.jsonl`,
  `arena.json`) and the learners' `latest.pt` under `/checkpoints/{human,zombie}/`. Arena
  copies in the `data` volume under `leagues/league-s0small-short2/`; metrics in TensorBoard
  (`/runs/league-s0small-short2/`) and the server's SQLite.

## Results

Group A (both learners against each other):

| Env steps | Human survived | Zombies caught | `s_z` |
|---|---|---|---|
| 1.3 M | 0.37 | 0.62 | 0.63 |
| 14.4 M | 0.77 | 0.22 | 0.23 |
| 28.8 M (resume) | 0.81 | 0.17 | 0.18 |
| 36.7 M | 0.86 | 0.13 | 0.13 |
| 40 M (last iterations) | 0.82-0.94 | 0.17-0.27 | - |

The guard has been at -1 (favoring the zombies) since iteration 60 (~7.9 M steps), and `s_z`
kept falling, from 0.23 to 0.13.

Arena Elo (last round 10:31, scripted zombies anchored at 1000):

| Side | Starter | Snapshot 1 | Snapshot 2 | Snapshot 3 |
|---|---|---|---|---|
| Human | 1453 | h0001 1588 | h0002 1657 | h0003 1660 |
| Zombies | 1425 | z0001 1368 | z0002 1398 | z0003 1393 |

Bots: zombie-m5 1301, human-m4 1086, human-scripted 1061, human-random 1012,
zombie-random 838. Sample matchups: h0003 survives z0003 85% of the time and zombie-m5 95%.

Anchors against the scripted bots (fixed seeds):

- Zombie learner against the scripted human: caught 0.77 -> 0.83 (the scripted zombies catch
  0.20). The zombies did learn to hunt; they just cannot keep up with the learned human.
- Human learner against the scripted zombies: survived 0.93 -> 0.91 (the scripted human
  survives 0.57).

Learner statistics:

- Zombie: entropy 1.94 -> 1.37, `approx_kl` 0.005 -> 0.001, lr 3e-4 -> 2.3e-5, explained
  variance ~0.76. PFSP pool score 0.78 -> 0.39: it loses to the whole human pool.
- Human: entropy 0.93 -> 0.77, `approx_kl` 0.007 -> 0.0002, lr 1e-4 -> 3.8e-6.
- Both learning rates were close to zero for the last ~10 M steps, so the run had effectively
  stopped learning before the budget ran out.

Tactics (training episodes):

- Human: inside a house less than 1% of the time, ~0.24 door closes per episode, ~2.8 food
  eaten, walking the whole time (120 walk noises/min). It survives by running and keeping its
  distance, not by using shelters. The 10th-percentile episode length went from 76 s to 235 s.
- Zombies: coordination did not improve. Encircled fraction ~0.075 and ~2.5 catch sectors,
  the same as at the start; dispersion ~34 m. Time to catch grew from 137 s to 167 s.

## Review

The run confirms that the zombie learner now updates, but the league stays unbalanced: the
human pulls ahead and plateaus (h0002 ~ h0003), while no zombie snapshot beats the zombie
starter. The guard at -1 is too weak to recover `s_z`. Suggested changes for the next run:

1. Zombie learning-rate schedule: decay more slowly or add a floor, since the zombie needs more
   updates than the human and its lr is nearly zero for the last quarter of the run.
2. A stronger guard: for example, pause the human's updates while `s_z` is below 0.3, instead
   of only favoring the zombies.
3. Zombie reward: encirclement did not improve in 40 M steps, so a signal that rewards
   coordination (encirclement, sector coverage) is probably missing.

## Resume (no longer needed)

The run finished at its budget. A continuation with a larger budget would be:

    docker compose up -d server
    make train-league-gpu ARGS="--total-steps <N> --run-id league-s0small-short2 --resume /checkpoints/league/league-s0small-short2/league.pt"
    make arena-gpu ARGS="--run-id league-s0small-short2"

With HTTPS, `.env` holds `ZAI_TLS_SITES` and the server starts with
`docker compose -f compose.yaml -f compose.tls.yaml up -d server proxy`.
