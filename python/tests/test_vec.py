import json

import numpy as np
import pytest
import zai_sim

from zai_env.vec import ZaiVecEnv


def random_actions(rng, num_envs, num_zombies):
    human = np.stack(
        [
            rng.integers(0, zai_sim.MOVE_ACTIONS, num_envs),
            rng.integers(0, zai_sim.INTERACT_ACTIONS, num_envs),
        ],
        axis=1,
    )
    return human, rng.integers(0, zai_sim.MOVE_ACTIONS, (num_envs, num_zombies))


def test_shapes_and_dtypes(small_config):
    venv = ZaiVecEnv(small_config, num_envs=8, seed=1, num_threads=2)
    obs = venv.reset()
    assert obs.human_obs.shape == (8, zai_sim.HUMAN_OBS_DIM)
    assert obs.human_obs.dtype == np.float32
    assert obs.human_mask.shape == (8, zai_sim.MOVE_ACTIONS + zai_sim.INTERACT_ACTIONS)
    assert obs.human_mask.dtype == bool
    assert obs.zombie_obs.shape == (8, 4, zai_sim.ZOMBIE_OBS_DIM)
    assert obs.zombie_valid.all()
    assert obs.state.shape == (8, zai_sim.STATE_DIM)
    step = venv.step(*random_actions(np.random.default_rng(0), 8, 4))
    assert step.human_reward.shape == (8,)
    assert step.outcome.dtype == np.uint8


def test_vec_env_matches_single_envs_including_auto_reset(small_config):
    """Rust-vs-Python parity: the batched env equals independent single envs."""
    num_envs, base_seed = 6, 42
    venv = ZaiVecEnv(small_config, num_envs=num_envs, seed=base_seed, num_threads=3)
    venv.reset()
    raw = json.dumps(small_config)
    singles = [zai_sim.Env(raw, zai_sim.episode_seed(base_seed, i, 0)) for i in range(num_envs)]
    episodes = [0] * num_envs
    assert venv.state_hashes() == [e.state_hash() for e in singles]

    rng = np.random.default_rng(0)
    resets = 0
    for _ in range(150):
        human, zombies = random_actions(rng, num_envs, 4)
        step = venv.step(human, zombies)
        for i, env in enumerate(singles):
            hr, zr, term, trunc, outcome = env.step(
                int(human[i, 0]), int(human[i, 1]), zombies[i].tolist()
            )
            assert step.human_reward[i] == pytest.approx(hr, abs=0)
            assert step.zombie_reward[i].tolist() == pytest.approx(zr, abs=0)
            assert (step.terminated[i], step.truncated[i], step.outcome[i]) == (
                term,
                trunc,
                outcome,
            )
            if term or trunc:
                np.testing.assert_array_equal(step.final_human_obs[i], env.human_obs())
                np.testing.assert_array_equal(step.final_zombie_obs[i], env.zombie_obs())
                np.testing.assert_array_equal(step.final_state[i], env.state())
                episodes[i] += 1
                resets += 1
                env.reset(zai_sim.episode_seed(base_seed, i, episodes[i]))
            np.testing.assert_array_equal(step.obs.human_obs[i], env.human_obs())
            np.testing.assert_array_equal(step.obs.human_mask[i], env.human_mask())
            np.testing.assert_array_equal(step.obs.zombie_obs[i], env.zombie_obs())
            np.testing.assert_array_equal(step.obs.state[i], env.state())
        assert venv.state_hashes() == [e.state_hash() for e in singles]
    assert resets >= num_envs, "every env should finish at least one 30 s episode"
    assert venv.current_seeds() == [
        zai_sim.episode_seed(base_seed, i, episodes[i]) for i in range(num_envs)
    ]


def test_thread_count_does_not_change_results(small_config):
    def run(threads):
        venv = ZaiVecEnv(small_config, num_envs=16, seed=5, num_threads=threads)
        venv.reset()
        rng = np.random.default_rng(1)
        for _ in range(100):
            venv.step(*random_actions(rng, 16, 4))
        return venv.state_hashes()

    assert run(1) == run(4)


def test_invalid_actions_raise(small_config):
    venv = ZaiVecEnv(small_config, num_envs=2, seed=0, num_threads=1)
    venv.reset()
    zombies = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        venv.step(np.array([[17, 0], [0, 0]]), zombies)
    with pytest.raises(ValueError):
        venv.step(np.zeros((2, 2)), np.zeros((2, 3)))
