"""Named access to observation vectors, using the layout exported by the Rust encoder."""

from typing import Any

import numpy as np
import zai_sim


class Layout:
    """Layout of one observation kind (`human`, `zombie` or `state`)."""

    def __init__(self, spec: dict[str, dict[str, Any]]):
        self.blocks = spec

    def block(self, obs: np.ndarray, name: str) -> np.ndarray:
        """View of a block with shape `[..., slots, fields]`."""
        b = self.blocks[name]
        width = len(b["fields"])
        flat = obs[..., b["start"] : b["start"] + b["slots"] * width]
        return flat.reshape(*obs.shape[:-1], b["slots"], width)

    def field(self, obs: np.ndarray, block: str, name: str) -> np.ndarray:
        """Values of one field for every slot, shape `[..., slots]`."""
        idx = self.blocks[block]["fields"].index(name)
        return self.block(obs, block)[..., idx]

    def scalar(self, obs: np.ndarray, block: str, name: str) -> np.ndarray:
        """Field of a single-slot block (e.g. `self`), shape `[...]`."""
        return self.field(obs, block, name)[..., 0]


_SPEC = zai_sim.obs_layout()
HUMAN = Layout(_SPEC["human"])
ZOMBIE = Layout(_SPEC["zombie"])
STATE = Layout(_SPEC["state"])


class Stats:
    """Names of the per-episode tactic counters (`VecStep.final_stats`, `stats.rs`)."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}

    def get(self, stats: np.ndarray, name: str) -> np.ndarray:
        return stats[..., self.index[name]]


STATS = Stats(zai_sim.STATS_LAYOUT)

# Layout v2 scales (crates/zai-sim/src/obs.rs): landmark deltas are `v / max(|v|, NEAR_M)`
# with a far distance `|v| / FAR_M`, absolute positions `pos / FAR_M`, time left
# `(cap - t) / TIME_SCALE_S`.
OBS_VERSION: int = zai_sim.OBS_VERSION
NEAR_M: float = zai_sim.OBS_NEAR_M
FAR_M: float = zai_sim.OBS_FAR_M
TIME_SCALE_S: float = zai_sim.OBS_TIME_SCALE_S


def house_deltas_m(
    obs: np.ndarray, layout: Layout, depth_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deltas in meters to each house center, front door and back door from a v2 observation,
    and the `valid` flags: arrays `[..., houses, 2]` (three of them) and `[..., houses]`."""
    f = lambda n: layout.field(obs, "houses", n).astype(np.float64)  # noqa: E731
    valid = f("valid") > 0.5
    scale = np.maximum(f("dist") * FAR_M, NEAR_M)
    center = np.stack([f("dx"), f("dy")], axis=-1) * scale[..., None]
    # rot = (sin, cos) of axis_x; axis_y is axis_x turned by +90 degrees.
    axis_y = np.stack([-f("rot_sin"), f("rot_cos")], axis=-1)
    front = center + axis_y * (depth_m / 2)
    back = center - axis_y * (depth_m / 2)
    zero = ~valid[..., None]
    return (
        np.where(zero, 0.0, center),
        np.where(zero, 0.0, front),
        np.where(zero, 0.0, back),
        valid,
    )
