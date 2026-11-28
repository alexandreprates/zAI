# Zombie-learner diagnostics after league-s0small-short3 (2026-11-28)

Diagnostics D1-D4 of `docs/ARCHITECTURE.md` 4.8.9 (hypotheses H1-H6 there). Run:
`league-s0small-short3` (40 M env steps, `configs/train/league_s0small_short3_gpu.yaml`), human
opponent h0002 (= the human learner's `latest.pt`, frozen from ~10.5 M steps). Scores are the
zombie side's score (caught + 0.5 starved), 128 episodes per zombie count 3/6/9 on the arena
seeds; repeated evaluations of the same policy differ by ~2 points.

## D1: cross-play (`crossplay.jsonl`, 384 episodes per pair, caught rate)

| Zombie | vs h0002 | vs scripted human | vs human-starter |
|---|---|---|---|
| zombie-starter | 0.234 | 0.776 | 0.602 |
| z0001 (10 M) | 0.219 | 0.776 | 0.458 |
| z0002 (20 M) | 0.255 | 0.818 | 0.406 |
| z0003 (30 M) | 0.221 | 0.805 | 0.365 |
| z0004 (40 M) | 0.247 | 0.807 | 0.365 |

Flat against the human it trained against, up against the scripted human, and much worse
against the starter human (forgetting). Starvation is ~3% in learner-vs-learner matchups, so
the starvation scoring mismatch (H3) is small.

## D2: the zombie learner's opponents (`pool.json`)

| Human entry | Games | Zombie learner score |
|---|---|---|
| human-scripted | 4827 | 0.884 |
| human-starter | 9153 | 0.413 |
| human-m4 | 1487 | 0.906 |
| h0001 | 5077 | 0.250 |
| h0002 | 3390 | 0.259 |

~26% of the games are against weak opponents (scripted, human-m4) that end quickly in a catch,
so they supply most of the catch signal (H1).

## D3: 10 M-step zombie fine-tunes against frozen h0002 (`probes.jsonl`)

`make probe-gpu ARGS="--run-id league-s0small-short3 --side zombie --tag=-a|-b|-c ..."`

| Probe | Init | Env (shaping) | Before | After | Gap |
|---|---|---|---|---|---|
| a | league zombie `latest.pt` | `s0_small_surround` (on) | 0.255 | 0.323 | +6.8 |
| b | league zombie `latest.pt` | `s0_small_sparse_zombie` (off) | 0.275 | 0.260 | -1.4 |
| c | zombie starter (crowd distill) | `s0_small_surround` (on) | 0.271 | 0.345 | +7.4 |

- (a) >= +5 pts: single-opponent training works, so the league's opponent mix is the problem
  (H1 confirmed).
- (b) < (a): the shaping helps; H2 rejected, change 2 of short4 (shaping off) is dropped.
- (c) ~ (a): no lost plasticity after 40 M league steps.

## D4: evaluation-only game limits (CPU, `noise.walk_radius_m: 100` / `zombie.speed_mps: 1.0`)

| Zombie vs h0002 | Baseline | Walk heard at 100 m | Zombies at 1.0 m/s |
|---|---|---|---|
| zombie-starter | 0.260 | 0.263 | 0.973 |
| z0004 | 0.271 | 0.249 | 0.986 |

Near-oracle hearing changes nothing; equal speed catches ~98%. The limit is geometry/speed
(0.7 vs 1.0 m/s), not information: learned zombie communication would not help. Against a good
learned evader the catch rate at 3/6/9 zombies on `s0_small` is probably capped near 0.25-0.35,
so the `s_z >= 0.25` balance criterion sits at the ceiling.

## Next steps (not yet decided by the user)

1. `league-s0small-short4` = short3 + change 1 (approved 2026-11-28: `league.scripted_share:
   {human: 0.2, zombie: 0.0}`, implemented in `train_league.py`) + change 3 (zombie
   `minibatches: 16`, proposed), shaping kept on. Config `league_s0small_short4_gpu.yaml` still
   to be written; go criteria in ARCHITECTURE.md 4.8.9.
2. Balance decision for the user, given D4: keep 0.7 m/s and make the balance criterion relative
   (zombie beats its starter), raise zombie counts (6/9/12), or raise zombie speed (PRD-level).
3. ml-architect to record D3/D4 and the verdict in ARCHITECTURE.md 4.8.9.

D4 script (run with `docker compose --profile train run --rm -T -e CUDA_VISIBLE_DEVICES=
trainer python - < d4.py`): `arena.probe.evaluate("zombie", <zombie ckpt>, <human latest.pt>,
deep_merge(load_config("configs/env/s0_small_surround.yaml"), override), [3, 6, 9], 128, "cpu")`
for the zombie starter and `pool/zombie/z0004.pt`, overrides `{}`,
`{"noise": {"walk_radius_m": 100}}` and `{"zombie": {"speed_mps": 1.0}}`.
