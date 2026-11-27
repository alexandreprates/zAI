import sys
from dataclasses import asdict

import pytest
import torch
import yaml

from zai_env.config import config_dir, load_config
from zai_rl import distill
from zai_rl.algos.common import RunningMeanStd, checkpoint_obs_version
from zai_rl.algos.mappo_zombie import MAPPOConfig
from zai_rl.algos.ppo_human import PPOConfig
from zai_rl.models import HumanActor, HumanCritic, ZombieActor, ZombieCritic
from zai_rl.policy import RecurrentHumanPolicy


def v1_checkpoint(path, side):
    """A checkpoint as written before M6.1 (no `obs_version`), with small random networks."""
    if side == "human":
        actor, critic = HumanActor(hidden=16, obs_version=1), HumanCritic(obs_version=1)
        hp = {"ppo_config": asdict(PPOConfig(num_envs=8, minibatches=2, hidden=16))}
    else:
        actor, critic = ZombieActor(hidden=16, obs_version=1), ZombieCritic(obs_version=1)
        hp = {"mappo_config": asdict(MAPPOConfig(num_envs=8, minibatches=2, hidden=16))}
    rms = RunningMeanStd()
    rms.update(torch.randn(64) * 3 + 1)
    sd = {"actor": actor.state_dict(), "critic": critic.state_dict(), "value_rms": rms.state_dict()}
    torch.save({**sd, **hp}, path)
    return path


@pytest.fixture
def tiny(tmp_path):
    env = load_config(
        config_dir() / "s0_small.yaml", overrides={"episode_cap_s": 20.0}, env_overrides=False
    )
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(env))
    return (
        env_path,
        v1_checkpoint(tmp_path / "h.pt", "human"),
        v1_checkpoint(tmp_path / "z.pt", "zombie"),
    )


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("side", ["human", "zombie"])
def test_updates_bring_the_student_closer_to_the_teacher(tiny, side, warm):
    env_path, h, z = tiny
    cfg = distill.DistillConfig(
        side=side,
        teacher=str(h if side == "human" else z),
        opponent=str(z if side == "human" else h),
        env_config=str(env_path),
        zombie_counts=[2, 3],
        num_envs=8,
        rollout=16,
        bptt=8,
        minibatches=2,
        epochs=1,
        lr=3e-3,
        num_threads=1,
        device="cpu",
        warm_start=warm,
    )
    env_cfg = load_config(env_path, env_overrides=False)
    d = distill.Distiller(cfg, env_cfg, torch.device("cpu"))
    assert d.teacher.legacy and len(d.env.groups) == 4  # scripted and policy opponents
    rows = d.env.num_envs if side == "human" else d.env.num_agents
    batch = d.env.reset()
    h_s = d.algo.actor.initial_state(rows)
    h_t = d.teacher.actor.initial_state(rows)
    start = torch.ones(rows, dtype=torch.bool)
    ep = {"outcome": [], "teacher": [], "length": []}
    rollout, *_ = d.collect(batch, h_s, h_t, start, beta=0.5, ep=ep)
    first = d.update(rollout)["train/kl"]
    if warm:
        # The transplanted student already matches the teacher on this 75 m map.
        assert first < 1e-4
        return
    for _ in range(15):
        last = d.update(rollout)["train/kl"]
    assert last < 0.5 * first
    # The student shares the teacher's value normalization.
    assert torch.equal(d.algo.value_rms.mean, d.teacher.value_rms.mean)


def test_main_writes_a_current_layout_checkpoint(tiny, tmp_path, monkeypatch):
    env_path, h, z = tiny
    cfg_path = tmp_path / "distill.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "side": "human",
                "teacher": str(h),
                "opponent": str(z),
                "env_config": str(env_path),
                "zombie_counts": [2],
                "total_steps": 2 * 16 * 8,
                "num_envs": 8,
                "rollout": 16,
                "bptt": 8,
                "minibatches": 2,
                "num_threads": 1,
                "device": "cpu",
                "eval_every": 1,
                "eval_episodes": 2,
                "final_eval_episodes": 2,
            }
        )
    )
    monkeypatch.setenv("ZAI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("ZAI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ZAI_SERVER_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["distill", "--config", str(cfg_path), "--run-id", "d1"])
    distill.main()
    final = tmp_path / "ckpt" / "human" / "d1" / "final.pt"
    sd = torch.load(final, weights_only=False)
    assert checkpoint_obs_version(sd) == 2
    assert "final/max_gap" in sd["distill_eval"]
    policy = RecurrentHumanPolicy.from_checkpoint(final, num_envs=2)
    assert policy.actor.obs_version == 2
