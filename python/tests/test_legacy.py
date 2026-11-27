"""Layout v2 against the v1 observations recorded before the change (`fixtures/obs_v1.npz`).

Re-simulating the recorded seeds and actions checks three things: the dynamics did not change
(same state hashes), the v1 adapter rebuilds the recorded v1 vectors from the v2 ones, and the
scripted bots, which now read the v2 fields, choose the recorded actions.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arena.export_model import export_checkpoint
from zai_env.legacy import adapt_v1_human, adapt_v1_state, adapt_v1_zombie
from zai_env.vec import ZaiVecEnv
from zai_rl.algos.ppo_human import HumanPPO, PPOConfig
from zai_rl.bots import ScriptedHuman, ScriptedZombies
from zai_rl.models import HumanActor, ZombieActor
from zai_rl.policy import RecurrentHumanPolicy, RecurrentZombiePolicy

FIXTURE = Path(__file__).parent / "fixtures" / "obs_v1.npz"


@pytest.fixture(scope="module")
def recorded():
    data = np.load(FIXTURE)
    meta = json.loads(bytes(data["meta"]).decode())
    return data, meta


@pytest.mark.parametrize("stage", ["s0_small", "s0", "default"])
def test_v2_re_simulates_and_adapts_to_the_recorded_v1(recorded, stage):
    data, meta = recorded
    cfg, seed = meta["stages"][stage]["config"], meta["stages"][stage]["seed"]
    envs, n = meta["envs"], cfg["zombie_count"]
    rec = {k: data[f"{stage}/{k}"] for k in ("human_obs", "zombie_obs", "state", "state_hash")}
    rec.update({k: data[f"{stage}/{k}"] for k in ("human_act", "zombie_act", "alive")})

    venv = ZaiVecEnv(cfg, num_envs=envs, seed=seed, num_threads=1)
    human = ScriptedHuman(cfg, envs, seed=seed + 1)
    zombies = ScriptedZombies(cfg, envs, n, seed=seed + 2)
    obs = venv.reset()
    for t in range(meta["steps"]):
        alive = rec["alive"][t]
        assert venv.state_hashes() == rec["state_hash"][t].tolist(), f"step {t}"
        np.testing.assert_allclose(
            adapt_v1_human(obs.human_obs, cfg)[alive], rec["human_obs"][t][alive], atol=1e-5
        )
        np.testing.assert_allclose(
            adapt_v1_zombie(obs.zombie_obs, cfg)[alive], rec["zombie_obs"][t][alive], atol=1e-5
        )
        np.testing.assert_allclose(
            adapt_v1_state(obs.state, cfg)[alive], rec["state"][t][alive], atol=1e-5
        )
        ha = human.act(obs.human_obs, obs.human_mask)
        za = zombies.act(obs.zombie_obs)
        np.testing.assert_array_equal(ha[alive], rec["human_act"][t][alive], f"step {t}")
        np.testing.assert_array_equal(za[alive], rec["zombie_act"][t][alive], f"step {t}")
        obs = venv.step(ha, za).obs


def test_v1_checkpoints_play_through_the_adapter_and_cannot_train_or_export(recorded, tmp_path):
    data, meta = recorded
    cfg = meta["stages"]["s0_small"]["config"]
    n = cfg["zombie_count"]
    # Checkpoints from before M6.1 have no `obs_version`.
    human = tmp_path / "human.pt"
    torch.save(
        {
            "actor": HumanActor(hidden=32, obs_version=1).state_dict(),
            "ppo_config": {"hidden": 32},
            "env_config": cfg,
        },
        human,
    )
    zombie = tmp_path / "zombie.pt"
    torch.save(
        {
            "actor": ZombieActor(hidden=16, obs_version=1).state_dict(),
            "mappo_config": {"hidden": 16},
            "env_config": cfg,
        },
        zombie,
    )

    venv = ZaiVecEnv(cfg, num_envs=2, seed=0, num_threads=1)
    obs = venv.reset()
    hp = RecurrentHumanPolicy.from_checkpoint(human, num_envs=2, config=cfg)
    zp = RecurrentZombiePolicy.from_checkpoint(zombie, num_envs=2, num_zombies=n, config=cfg)
    assert hp.actor.obs_version == zp.actor.obs_version == 1
    for _ in range(3):
        obs = venv.step(hp.act(obs.human_obs, obs.human_mask), zp.act(obs.zombie_obs)).obs

    with pytest.raises(ValueError, match="env config"):
        RecurrentHumanPolicy.from_checkpoint(human, num_envs=2)
    algo = HumanPPO(PPOConfig(num_envs=2, minibatches=1, hidden=32), torch.device("cpu"))
    with pytest.raises(ValueError, match="layout v1"):
        algo.load_state_dict(torch.load(human, weights_only=False))
    with pytest.raises(ValueError, match="layout v1"):
        export_checkpoint(human, tmp_path / "models")
