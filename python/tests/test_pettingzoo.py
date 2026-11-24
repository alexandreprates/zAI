from pettingzoo.test import parallel_api_test, parallel_seed_test

from zai_env.parallel import ZaiParallelEnv


def test_parallel_api(small_config):
    parallel_api_test(ZaiParallelEnv(small_config), num_cycles=200)


def test_parallel_seeding(small_config):
    parallel_seed_test(lambda: ZaiParallelEnv(small_config))


def test_human_info_carries_action_mask(small_config):
    env = ZaiParallelEnv(small_config)
    _, infos = env.reset(seed=3)
    assert infos["human"]["action_mask"].shape == (20,)
    assert set(env.agents) == {"human", "zombie_0", "zombie_1", "zombie_2", "zombie_3"}
