import importlib

import pytest


def test_zai_sim_extension_loads():
    import zai_sim

    assert zai_sim.__version__ == "0.1.0"
    assert (zai_sim.HUMAN_OBS_DIM, zai_sim.ZOMBIE_OBS_DIM, zai_sim.STATE_DIM) == (584, 252, 1144)


@pytest.mark.parametrize("package", ["zai_env", "zai_rl", "zai_rl.bots", "zai_agent", "arena"])
def test_packages_import(package):
    importlib.import_module(package)


def test_layout_exported_by_rust_covers_each_observation():
    import zai_sim

    from zai_env import layout

    for name, lay, dim in [
        ("human", layout.HUMAN, zai_sim.HUMAN_OBS_DIM),
        ("zombie", layout.ZOMBIE, zai_sim.ZOMBIE_OBS_DIM),
        ("state", layout.STATE, zai_sim.STATE_DIM),
    ]:
        total = sum(b["slots"] * len(b["fields"]) for b in lay.blocks.values())
        assert total == dim, name
