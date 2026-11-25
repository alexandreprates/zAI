import sys

import numpy as np
import torch
import yaml

from zai_env.config import config_dir, load_config
from zai_rl import train, train_zombie
from zai_rl.algos.common import save_checkpoint
from zai_rl.algos.mappo_zombie import MAPPOConfig, ZombieMAPPO
from zai_rl.algos.ppo_human import HumanPPO, PPOConfig
from zai_rl.envs import HumanTrainingEnv
from zai_rl.models import ZombieActor
from zai_rl.policy import RecurrentZombiePolicy


def test_recurrent_zombie_opponent_forgets_finished_episodes():
    cfg = load_config(
        config_dir() / "s0.yaml", overrides={"episode_cap_s": 0.5}, env_overrides=False
    )
    opponents = []

    def factory(c, e, n, s):
        opponents.append(RecurrentZombiePolicy(ZombieActor(hidden=16), e, n))
        return opponents[-1]

    env = HumanTrainingEnv(cfg, [2], 3, factory, num_threads=1)
    env.reset()
    # 0.5 s episodes end after one decision, so memory is cleared after every step.
    env.step(np.zeros((3, 2), dtype=np.int64))
    assert opponents[0].h.abs().sum() == 0


def test_fine_tune_against_a_zombie_policy(tmp_path, monkeypatch):
    env_cfg = load_config(config_dir() / "s0.yaml", env_overrides=False)
    zombie = tmp_path / "zombie.pt"
    save_checkpoint(zombie, ZombieMAPPO(MAPPOConfig(hidden=16), torch.device("cpu")), {})
    ppo = {"num_envs": 4, "rollout": 16, "bptt": 8, "epochs": 1, "minibatches": 2, "hidden": 32}
    human = tmp_path / "human.pt"
    save_checkpoint(human, HumanPPO(PPOConfig(**ppo), torch.device("cpu")), {"steps": 1})
    config = tmp_path / "ft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "env_config": str(config_dir() / "s0.yaml"),
                "zombie_counts": [2],
                "opponent": "policy",
                "zombie_checkpoint": str(zombie),
                "init_checkpoint": str(human),
                "total_steps": 128,
                "num_threads": 1,
                "device": "cpu",
                "eval_every": 2,
                "eval_episodes": 2,
                "checkpoint_every": 2,
                "ppo": ppo,
            }
        )
    )
    monkeypatch.setenv("ZAI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("ZAI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ZAI_SERVER_URL", raising=False)
    monkeypatch.delenv("ZAI_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--run-id", "ft"])
    train.main()
    sd = torch.load(tmp_path / "ckpt" / "human" / "ft" / "latest.pt", weights_only=False)
    assert sd["steps"] == 128 and sd["env_config"] == env_cfg


def test_zombie_fine_tune_against_a_human_policy(tmp_path, monkeypatch):
    env_cfg = load_config(config_dir() / "s0.yaml", env_overrides=False)
    human = tmp_path / "human.pt"
    save_checkpoint(human, HumanPPO(PPOConfig(hidden=16), torch.device("cpu")), {})
    mappo = {"num_envs": 4, "rollout": 16, "bptt": 8, "epochs": 1, "minibatches": 2, "hidden": 32}
    zombie = tmp_path / "zombie.pt"
    save_checkpoint(zombie, ZombieMAPPO(MAPPOConfig(**mappo), torch.device("cpu")), {"steps": 1})
    config = tmp_path / "ft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "env_config": str(config_dir() / "s0.yaml"),
                "zombie_counts": [3],
                "opponent": "policy",
                "human_checkpoint": str(human),
                "init_checkpoint": str(zombie),
                "total_steps": 128,
                "num_threads": 1,
                "device": "cpu",
                "eval_every": 2,
                "eval_episodes": 2,
                "baseline_episodes": 2,
                "checkpoint_every": 2,
                "mappo": mappo,
            }
        )
    )
    monkeypatch.setenv("ZAI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("ZAI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ZAI_SERVER_URL", raising=False)
    monkeypatch.delenv("ZAI_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["train_zombie", "--config", str(config), "--run-id", "zft"])
    train_zombie.main()
    sd = torch.load(tmp_path / "ckpt" / "zombie" / "zft" / "latest.pt", weights_only=False)
    assert sd["steps"] == 128 and sd["env_config"] == env_cfg


def test_zombie_policy_opponent_needs_a_checkpoint():
    import pytest

    with pytest.raises(ValueError, match="human_checkpoint"):
        train_zombie.ZombieTrainConfig.load(None, {"opponent": "policy"})
