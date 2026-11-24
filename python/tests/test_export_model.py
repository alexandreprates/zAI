import json

import numpy as np
import torch
import zai_sim

from arena.export_model import (
    HumanStep,
    ZombieStep,
    export_checkpoint,
    export_training_model,
    stage_name,
)
from zai_env.config import config_dir, load_config
from zai_rl.algos.common import save_checkpoint
from zai_rl.algos.mappo_zombie import MAPPOConfig, ZombieMAPPO
from zai_rl.models import HumanActor, ZombieActor


def test_step_modules_match_the_actors():
    torch.manual_seed(0)
    human = HumanActor(hidden=16, entity_hidden=8).eval()
    obs = torch.rand(4, zai_sim.HUMAN_OBS_DIM)
    h = torch.randn(4, 16)
    mask = torch.ones(4, zai_sim.MOVE_ACTIONS + zai_sim.INTERACT_ACTIONS, dtype=torch.bool)
    move, interact, h_new = human.step(obs, h, mask)
    m, i, h2 = HumanStep(human)(obs, h)
    torch.testing.assert_close(torch.log_softmax(m, -1), move.logits)
    torch.testing.assert_close(torch.log_softmax(i, -1), interact.logits)
    torch.testing.assert_close(h2, h_new)

    zombie = ZombieActor(hidden=16, entity_hidden=8).eval()
    obs = torch.rand(5, zai_sim.ZOMBIE_OBS_DIM)
    h = torch.randn(5, 16)
    dist, h_new = zombie.step(obs, h)
    m, h2 = ZombieStep(zombie)(obs, h)
    torch.testing.assert_close(torch.log_softmax(m, -1), dist.logits)
    torch.testing.assert_close(h2, h_new)


def test_stage_is_the_matching_env_config_file():
    assert stage_name(load_config(config_dir() / "s0_small.yaml", env_overrides=False)) == (
        "s0_small"
    )
    cfg = load_config(config_dir() / "s0.yaml", overrides={"food_count": 7}, env_overrides=False)
    assert stage_name(cfg) == "custom"
    # Rewards do not change the map, and configs saved before a key existed get its default.
    old = load_config(config_dir() / "s0.yaml", env_overrides=False)
    old["reward"]["zombie_step_penalty"] = 0.123
    del old["food_near_spawn_count"]
    assert stage_name(old) == "s0"


def test_checkpoint_exports_onnx_with_metadata(tmp_path):
    algo = ZombieMAPPO(MAPPOConfig(hidden=16), torch.device("cpu"))
    cfg = load_config(config_dir() / "s0.yaml", env_overrides=False)
    ckpt = tmp_path / "ckpt" / "run-a" / "latest.pt"
    save_checkpoint(ckpt, algo, {"iteration": 3, "steps": 99, "run_id": "run a", "env_config": cfg})
    onnx_path = export_checkpoint(ckpt, tmp_path / "models")
    assert onnx_path.name == "run-a.onnx" and onnx_path.stat().st_size > 0
    info = json.loads(onnx_path.with_suffix(".json").read_text())
    assert info["side"] == "zombie" and info["stage"] == "s0" and info["hidden"] == 16
    assert info["obs_dim"] == zai_sim.ZOMBIE_OBS_DIM and info["iteration"] == 3
    assert info["config"] == cfg


def test_training_export_needs_the_data_dir_and_never_raises(tmp_path, monkeypatch):
    actor = HumanActor(hidden=16, entity_hidden=8)
    cfg = load_config(config_dir() / "s0.yaml", env_overrides=False)
    monkeypatch.delenv("ZAI_DATA_DIR", raising=False)
    export_training_model(actor, "r1", cfg, 1, 10)
    monkeypatch.setenv("ZAI_DATA_DIR", str(tmp_path))
    export_training_model(actor, "r1", cfg, 1, 10)
    assert (tmp_path / "models" / "r1.onnx").exists()
    assert json.loads((tmp_path / "models" / "r1.json").read_text())["side"] == "human"
    # A config that cannot be compared still must not stop training.
    export_training_model(actor, "r2", {"config": np.zeros(1)}, 1, 10)
