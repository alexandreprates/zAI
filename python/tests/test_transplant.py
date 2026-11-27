import numpy as np
import pytest
import torch

from zai_env.config import config_dir, load_config
from zai_env.legacy import adapt_v1_human, adapt_v1_state, adapt_v1_zombie
from zai_env.vec import ZaiVecEnv
from zai_rl.bots import ScriptedHuman, ScriptedZombies
from zai_rl.models import HumanActor, HumanCritic, ZombieActor, ZombieCritic
from zai_rl.transplant import transplant_actor, transplant_critic


def visited_states(cfg, steps=60, envs=6):
    """v2 observations of the scripted bots on `cfg` (every 5th decision)."""
    venv = ZaiVecEnv(cfg, num_envs=envs, seed=11, num_threads=1)
    human = ScriptedHuman(cfg, envs, seed=1)
    zombies = ScriptedZombies(cfg, envs, cfg["zombie_count"], seed=2)
    obs = venv.reset()
    out = []
    for t in range(steps):
        if t % 5 == 0:
            out.append(
                (
                    obs.human_obs.copy(),
                    obs.human_mask.copy(),
                    obs.zombie_obs.copy(),
                    obs.state.copy(),
                )
            )
        obs = venv.step(human.act(obs.human_obs, obs.human_mask), zombies.act(obs.zombie_obs)).obs
    return [np.concatenate(x) for x in zip(*out, strict=True)]


@pytest.fixture(scope="module")
def small():
    cfg = load_config(
        config_dir() / "s0_small.yaml", overrides={"zombie_count": 6}, env_overrides=False
    )
    return cfg, visited_states(cfg)


def test_transplanted_actors_and_critics_match_their_v1_source(small):
    """On the 75 m map every distance is within 100 m, so the rewrite is exact."""
    torch.manual_seed(0)
    cfg, (hobs, hmask, zobs, state) = small
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    zobs = zobs.reshape(-1, zobs.shape[-1])
    zstate = np.repeat(state, cfg["zombie_count"], axis=0)

    h1, h2 = HumanActor(hidden=32, obs_version=1), HumanActor(hidden=32)
    transplant_actor(h2, h1, cfg)
    mask = torch.as_tensor(hmask)
    with torch.no_grad():
        a = h1.step(t(adapt_v1_human(hobs, cfg)), h1.initial_state(len(hobs)), mask)
        b = h2.step(t(hobs), h2.initial_state(len(hobs)), mask)
    torch.testing.assert_close(b[0].probs, a[0].probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(b[1].probs, a[1].probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(b[2], a[2], atol=1e-5, rtol=1e-4)

    z1, z2 = ZombieActor(hidden=16, obs_version=1), ZombieActor(hidden=16)
    transplant_actor(z2, z1, cfg)
    with torch.no_grad():
        da, _ = z1.step(t(adapt_v1_zombie(zobs, cfg)), z1.initial_state(len(zobs)))
        db, _ = z2.step(t(zobs), z2.initial_state(len(zobs)))
    torch.testing.assert_close(db.probs, da.probs, atol=1e-5, rtol=1e-4)

    c1, c2 = HumanCritic(obs_version=1), HumanCritic()
    transplant_critic(c2, c1, cfg)
    with torch.no_grad():
        torch.testing.assert_close(
            c2(t(state)), c1(t(adapt_v1_state(state, cfg))), atol=1e-4, rtol=1e-4
        )
    k1, k2 = ZombieCritic(obs_version=1), ZombieCritic()
    transplant_critic(k2, k1, cfg)
    with torch.no_grad():
        va = k1(t(adapt_v1_state(zstate, cfg)), t(adapt_v1_zombie(zobs, cfg)))
        vb = k2(t(zstate), t(zobs))
    torch.testing.assert_close(vb, va, atol=1e-4, rtol=1e-4)
