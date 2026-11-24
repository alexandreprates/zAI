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
