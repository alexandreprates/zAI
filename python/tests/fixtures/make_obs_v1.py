"""Writes `obs_v1.npz`: observations, states and actions of the v1 layout (M6.1 step 1,
docs/ARCHITECTURE.md section 4.8.9), recorded before `obs.rs` changes to layout v2.

After layout v2 lands, a test re-simulates the same configs, seeds and actions and checks that
`zai_env.legacy.adapt_v1` rebuilds these v1 vectors from the v2 ones, which is how the old
checkpoints (the distillation teachers) are shown to see their own input distribution.

Each stage runs `ENVS` envs for `STEPS` decisions with the scripted bots (so pickups, door
closes and noise happen), from the start of their first episode; rows after an env's first
episode ends are marked invalid in `alive`. Run from `python/`:

    python -m tests.fixtures.make_obs_v1
"""

import json
from pathlib import Path

import numpy as np
import zai_sim

from zai_env.config import config_dir, load_config
from zai_env.vec import ZaiVecEnv
from zai_rl.bots import ScriptedHuman, ScriptedZombies

ENVS = 3
STEPS = 120
SEED = 1234
# Stage config file and zombie count: the M6 stages (4.8.1) and the PRD defaults.
STAGES = {
    "s0_small": ("s0_small.yaml", 9),
    "s0": ("s0.yaml", 8),
    "default": ("default.yaml", 20),
}
OUT = Path(__file__).with_name("obs_v1.npz")


def record(config: dict, seed: int) -> dict[str, np.ndarray]:
    n = config["zombie_count"]
    venv = ZaiVecEnv(config, num_envs=ENVS, seed=seed, num_threads=1)
    human = ScriptedHuman(config, ENVS, seed=seed + 1)
    zombies = ScriptedZombies(config, ENVS, n, seed=seed + 2)
    obs = venv.reset()
    rec: dict[str, list] = {
        k: [] for k in ("human_obs", "human_mask", "zombie_obs", "state", "human_act", "zombie_act")
    }
    alive = np.ones(ENVS, dtype=bool)
    alive_rows = []
    hashes = []
    for _ in range(STEPS):
        for k in ("human_obs", "human_mask", "zombie_obs", "state"):
            rec[k].append(getattr(obs, k).copy())
        alive_rows.append(alive.copy())
        hashes.append(venv.state_hashes())
        ha = human.act(obs.human_obs, obs.human_mask)
        za = zombies.act(obs.zombie_obs)
        rec["human_act"].append(np.asarray(ha).copy())
        rec["zombie_act"].append(np.asarray(za).copy())
        step = venv.step(ha, za)
        alive &= ~(step.terminated | step.truncated)
        obs = step.obs
    out = {k: np.stack(v) for k, v in rec.items()}
    out["alive"] = np.stack(alive_rows)  # [T, E]
    out["state_hash"] = np.array(hashes, dtype=np.uint64)  # [T, E], before each step
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "obs_dims": {
            "human": zai_sim.HUMAN_OBS_DIM,
            "zombie": zai_sim.ZOMBIE_OBS_DIM,
            "state": zai_sim.STATE_DIM,
        },
        "envs": ENVS,
        "steps": STEPS,
        "seed": SEED,
        "stages": {},
    }
    for i, (name, (file, n)) in enumerate(STAGES.items()):
        cfg = load_config(config_dir() / file, overrides={"zombie_count": n}, env_overrides=False)
        rec = record(cfg, SEED + 100 * i)
        meta["stages"][name] = {"config": cfg, "seed": SEED + 100 * i}
        for k, v in rec.items():
            arrays[f"{name}/{k}"] = v
        print(f"{name}: alive rows {rec['alive'].sum()} of {rec['alive'].size}")
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
