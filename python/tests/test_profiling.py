import numpy as np
import torch

from zai_env.config import config_dir, load_config
from zai_rl.algos.common import EpisodeStats
from zai_rl.algos.ppo_human import HumanPPO, PPOConfig
from zai_rl.bots import ScriptedZombies
from zai_rl.envs import HumanTrainingEnv
from zai_rl.profiling import PhaseTimer, maybe_trace, rollout_split, timer


def test_phase_timer_accumulates_and_clears():
    t = PhaseTimer()
    for _ in range(3):
        with t("sim"):
            pass
    with t("policy"):
        pass
    out = t.take()
    assert set(out) == {"perf/sim_s", "perf/policy_s"}
    assert all(v >= 0 for v in out.values())
    assert t.take() == {}


def test_rollout_split_derives_glue_and_update_step():
    s = {
        "perf/rollout_s": 10.0,
        "perf/sim_s": 2.0,
        "perf/opponent_s": 1.0,
        "perf/policy_s": 3.0,
        "perf/update_s": 5.0,
        "perf/update_gather_s": 1.5,
    }
    rollout_split(s)
    assert s["perf/glue_s"] == 4.0
    assert s["perf/update_step_s"] == 3.5


def test_maybe_trace_only_on_the_chosen_iteration(tmp_path):
    with maybe_trace(1, 2, tmp_path / "a"):
        torch.ones(3).sum()
    assert not (tmp_path / "a").exists()
    with maybe_trace(2, 2, tmp_path / "b"), torch.profiler.record_function("sim"):
        torch.ones(3).sum()
    assert (tmp_path / "b" / "trace.json").stat().st_size > 0
    assert "sim" in (tmp_path / "b" / "summary.txt").read_text()


def test_human_rollout_reports_every_phase():
    cfg = load_config(config_dir() / "s0.yaml", env_overrides=False)
    ppo = PPOConfig(num_envs=4, rollout=8, bptt=4, minibatches=2, epochs=1, hidden=32)
    env = HumanTrainingEnv(
        cfg, [2], 4, lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s), num_threads=1
    )
    algo = HumanPPO(ppo, torch.device("cpu"))
    batch = env.reset()
    h = algo.actor.initial_state(4, torch.device("cpu"))
    start = torch.ones(4, dtype=torch.bool)
    timer.take()
    algo.collect(env, batch, h, start, np.zeros(4), EpisodeStats())
    phases = timer.take()
    assert {"perf/sim_s", "perf/opponent_s", "perf/policy_s"} <= set(phases)
