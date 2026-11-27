"""Observation layout v1 and the adapter from layout v2 (docs/ARCHITECTURE.md section 4.8.2).

Layout v1 (M1-M5) divided positions and landmark deltas by the map size; v2 divides them by
constants. Checkpoints trained on v1 (the M4/M5 policies and the fine-tunes of section 4.7) keep
working in the v2 env through `adapt_v1_*`, which rebuild the v1 vectors from v2 ones and the
env config: absolute positions are rescaled, house deltas come back from the saturated delta
and the far distance, door deltas from the house center, its rotation and the configured house
depth (doors sit at `center +- axis_y * depth / 2`), and the time fields from `time_left`.
The rebuild is exact up to float rounding; `tests/test_legacy.py` checks it against v1
observations recorded from the v1 encoders (`tests/fixtures/obs_v1.npz`).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .layout import FAR_M, HUMAN, STATE, TIME_SCALE_S, ZOMBIE, Layout, house_deltas_m

_SPEC = json.loads((Path(__file__).with_name("layout_v1.json")).read_text())
HUMAN_V1 = Layout(_SPEC["human"])
ZOMBIE_V1 = Layout(_SPEC["zombie"])
STATE_V1 = Layout(_SPEC["state"])
HUMAN_OBS_DIM_V1 = _SPEC["dims"]["human"]
ZOMBIE_OBS_DIM_V1 = _SPEC["dims"]["zombie"]
STATE_DIM_V1 = _SPEC["dims"]["state"]


def _copy_block(dst: np.ndarray, src: np.ndarray, new: Layout, old: Layout, name: str) -> None:
    """Copies the fields of block `name` that both layouts have, by name."""
    a, b = new.blocks[name], old.blocks[name]
    src_block, dst_block = new.block(src, name), old.block(dst, name)
    for i, f in enumerate(b["fields"]):
        if f in a["fields"]:
            dst_block[..., i] = src_block[..., a["fields"].index(f)]


def _write(dst: np.ndarray, layout: Layout, block: str, field: str, value: np.ndarray) -> None:
    idx = layout.blocks[block]["fields"].index(field)
    layout.block(dst, block)[..., idx] = value


def adapt_v1_human(obs: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """v1 human observations `[..., 584]` from v2 ones `[..., 608]`."""
    size = float(config["map_size_m"])
    out = np.zeros((*obs.shape[:-1], HUMAN_OBS_DIM_V1), dtype=np.float32)
    for name in ("self", "zombies", "food", "houses", "rays"):
        _copy_block(out, obs, HUMAN, HUMAN_V1, name)
    for axis in ("x", "y"):
        v = HUMAN.scalar(obs, "self", axis)[..., None] * FAR_M / size
        _write(out, HUMAN_V1, "self", axis, v)
    center, front, back, _ = house_deltas_m(obs, HUMAN, config["house"]["depth_m"])
    for prefix, d in (("", center), ("front_", front), ("back_", back)):
        _write(out, HUMAN_V1, "houses", f"{prefix}dx", d[..., 0] / size)
        _write(out, HUMAN_V1, "houses", f"{prefix}dy", d[..., 1] / size)
    return out


def adapt_v1_zombie(obs: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """v1 zombie observations `[..., 252]` from v2 ones `[..., 276]`."""
    size = float(config["map_size_m"])
    cap = float(config["episode_cap_s"])
    out = np.zeros((*obs.shape[:-1], ZOMBIE_OBS_DIM_V1), dtype=np.float32)
    for name in ("self", "heard", "peers", "houses", "rays"):
        _copy_block(out, obs, ZOMBIE, ZOMBIE_V1, name)
    for axis in ("x", "y"):
        v = ZOMBIE.scalar(obs, "self", axis)[..., None] * FAR_M / size
        _write(out, ZOMBIE_V1, "self", axis, v)
    left = ZOMBIE.scalar(obs, "self", "time_left").astype(np.float64)
    _write(out, ZOMBIE_V1, "self", "time", (1.0 - left * TIME_SCALE_S / cap)[..., None])
    center, front, back, _ = house_deltas_m(obs, ZOMBIE, config["house"]["depth_m"])
    for prefix, d in (("", center), ("front_", front), ("back_", back)):
        _write(out, ZOMBIE_V1, "houses", f"{prefix}dx", d[..., 0] / size)
        _write(out, ZOMBIE_V1, "houses", f"{prefix}dy", d[..., 1] / size)
    return out


def adapt_v1_state(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """v1 critic states from v2 ones (both `[..., 1144]`)."""
    k = FAR_M / float(config["map_size_m"])
    out = np.zeros(state.shape, dtype=np.float32)
    for name in ("human", "zombies", "food", "houses", "time"):
        _copy_block(out, state, STATE, STATE_V1, name)
    positions = {
        "human": ("x", "y"),
        "zombies": ("x", "y"),
        "food": ("x", "y"),
        "houses": ("x", "y", "front_x", "front_y", "back_x", "back_y"),
    }
    for block, fields in positions.items():
        for f in fields:
            _write(out, STATE_V1, block, f, STATE.field(state, block, f) * k)
    hx = STATE.scalar(state, "human", "x")[..., None]
    hy = STATE.scalar(state, "human", "y")[..., None]
    zx, zy = STATE.field(state, "zombies", "x"), STATE.field(state, "zombies", "y")
    valid = STATE.field(state, "zombies", "valid") > 0.5
    dist = np.hypot(zx.astype(np.float64) - hx, zy.astype(np.float64) - hy) * k
    _write(out, STATE_V1, "zombies", "dist_to_human", np.where(valid, dist, 0.0))
    _write(out, STATE_V1, "time", "bias", np.ones(state.shape[:-1] + (1,)))
    return out
