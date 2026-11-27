"""CUDA graphs of the actors' step (zai_rl.models.graphs); skipped without a GPU."""

import pytest
import torch
import zai_sim

from zai_rl.models import HumanActor, ZombieActor, graphs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_unusable_on_cpu():
    assert not graphs.usable("cpu")


@cuda
def test_graph_matches_eager_and_follows_weight_updates():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    za, zb = ZombieActor(hidden=32).to(dev), ZombieActor(hidden=32).to(dev)
    o = torch.randn(64, zai_sim.ZOMBIE_OBS_DIM, device=dev)
    h = torch.randn(64, 32, device=dev)
    g = graphs.GraphedActor(za, o, h)
    with torch.no_grad():
        for _ in range(2):  # before and after loading other weights in place
            d, hn = za.step(o, h)
            logits, hg = g.run(o, h)
            torch.testing.assert_close(logits, d.logits)
            torch.testing.assert_close(hg, hn)
            za.load_state_dict(zb.state_dict())
    ha = HumanActor(hidden=32).to(dev)
    o = torch.randn(16, zai_sim.HUMAN_OBS_DIM, device=dev)
    m = torch.ones(16, 20, dtype=torch.bool, device=dev)
    h = torch.zeros(16, 32, device=dev)
    g = graphs.GraphedActor(ha, o, h, m)
    with torch.no_grad():
        move, inter, hn = ha.step(o, h, m)
        ml, il, hg = g.run(o, h, m)
    torch.testing.assert_close(ml, move.logits)
    torch.testing.assert_close(il, inter.logits)
    torch.testing.assert_close(hg, hn)
