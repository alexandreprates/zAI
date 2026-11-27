"""Warm start of a layout v2 network from a layout v1 one (M6.1 distillation).

Within 100 m, every v1 input is an affine function of the v2 inputs of the same block (`v1 =
A v2 + c`): landmark deltas `d / size = sat(d) * NEAR_M / size`, the absolute position from the
wall distances (`x / size = wall_w * NEAR_M / size` while `x < NEAR_M`), state positions
`(x / FAR_M) * FAR_M / size`, the zombie's `time / cap = 1 - time_left * TIME_SCALE_S / cap`,
and the state's `dist_to_human`. So a first linear layer `W v1 + b` that reads such a block
becomes `(W A) v2 + (b + W c)`, exactly on maps up to 100 m (`s0_small`) and approximately
beyond, and every other layer is copied. Distillation then only has to correct what the
affine map misses (saturated distances on larger maps).
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from zai_env.layout import FAR_M, HUMAN, NEAR_M, STATE, TIME_SCALE_S, ZOMBIE
from zai_env.legacy import HUMAN_V1, STATE_V1, ZOMBIE_V1

from .models import HumanActor, HumanCritic, ZombieActor, ZombieCritic

DELTAS = ("dx", "dy", "front_dx", "front_dy", "back_dx", "back_dy")
POSITIONS = ("x", "y", "front_x", "front_y", "back_x", "back_y")


def field_map(kind: str, block: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """`(A [f1, f2], c [f1])` with `v1 = A v2 + c` for one slot of `block` of observation
    `kind` (`human`, `zombie` or `state`)."""
    new, old = {"human": (HUMAN, HUMAN_V1), "zombie": (ZOMBIE, ZOMBIE_V1)}.get(
        kind, (STATE, STATE_V1)
    )
    f2 = new.blocks[block]["fields"]
    f1 = old.blocks[block]["fields"]
    size, cap = float(config["map_size_m"]), float(config["episode_cap_s"])
    A = np.zeros((len(f1), len(f2)))
    c = np.zeros(len(f1))
    for i, name in enumerate(f1):
        if kind != "state" and block == "self" and name in ("x", "y"):
            A[i, f2.index("wall_w" if name == "x" else "wall_s")] = NEAR_M / size
        elif kind != "state" and block == "houses" and name in DELTAS:
            A[i, f2.index(name)] = NEAR_M / size
        elif kind == "zombie" and block == "self" and name == "time":
            A[i, f2.index("time_left")] = -TIME_SCALE_S / cap
            c[i] = 1.0
        elif kind == "state" and name in POSITIONS:
            A[i, f2.index(name)] = FAR_M / size
        elif kind == "state" and name == "dist_to_human":
            A[i, f2.index(name)] = NEAR_M / size
        elif kind == "state" and name == "bias":
            c[i] = 1.0
        else:
            A[i, f2.index(name)] = 1.0
    return A, c


def vector_map(kind: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """`(A [d1, d2], c [d1])` for a whole observation or state vector."""
    new, old = {"human": (HUMAN, HUMAN_V1), "zombie": (ZOMBIE, ZOMBIE_V1)}.get(
        kind, (STATE, STATE_V1)
    )
    d1 = sum(b["slots"] * len(b["fields"]) for b in old.blocks.values())
    d2 = sum(b["slots"] * len(b["fields"]) for b in new.blocks.values())
    A, c = np.zeros((d1, d2)), np.zeros(d1)
    for name, b1 in old.blocks.items():
        b2 = new.blocks[name]
        a, cc = field_map(kind, name, config)
        w1, w2 = len(b1["fields"]), len(b2["fields"])
        for s in range(b1["slots"]):
            r, k = b1["start"] + s * w1, b2["start"] + s * w2
            A[r : r + w1, k : k + w2] = a
            c[r : r + w1] = cc
    return A, c


@torch.no_grad()
def _rewrite(dst: nn.Linear, src: nn.Linear, A: np.ndarray, c: np.ndarray, cols=None) -> None:
    """`dst = src` on inputs `A v + c`; with `cols`, only the input columns `cols` of the
    source layer read the mapped vector (the rest are copied to the remaining columns)."""
    W = src.weight.double()
    At = torch.as_tensor(A, dtype=torch.float64, device=W.device)
    ct = torch.as_tensor(c, dtype=torch.float64, device=W.device)
    if cols is None:
        dst.weight.copy_(W @ At)
        dst.bias.copy_(src.bias.double() + W @ ct)
        return
    src_cols, dst_cols = cols
    rest_src = [j for j in range(W.shape[1]) if j not in set(src_cols)]
    rest_dst = [j for j in range(dst.weight.shape[1]) if j not in set(dst_cols)]
    Ws = W[:, src_cols]
    new = torch.zeros_like(dst.weight, dtype=torch.float64)
    new[:, dst_cols] = Ws @ At
    new[:, rest_dst] = W[:, rest_src]
    dst.weight.copy_(new)
    dst.bias.copy_(src.bias.double() + Ws @ ct)


def _copy_rest(dst: nn.Module, src: nn.Module, skip: set[str]) -> None:
    sd = {k: v for k, v in src.state_dict().items() if k.rsplit(".", 1)[0] not in skip}
    missing = dst.load_state_dict(sd, strict=False)
    left = {k.rsplit(".", 1)[0] for k in missing.missing_keys}
    assert left <= skip, f"not transplanted: {left - skip}"


def transplant_actor(
    student: HumanActor | ZombieActor, teacher: HumanActor | ZombieActor, config: dict[str, Any]
) -> None:
    """Sets `student` (layout v2) to the v1 `teacher` composed with the v2 -> v1 map."""
    kind = "human" if isinstance(student, HumanActor) else "zombie"
    _copy_rest(student, teacher, {"body.0", "houses.net.0"})
    a, c = field_map(kind, "houses", config)
    _rewrite(student.houses.net[0], teacher.houses.net[0], a, c)
    # Body input: [own self fields | entity encodings | rays]; only the own part changes.
    a, c = field_map(kind, "self", config)
    a = a[teacher.own_cols.cpu().numpy()][:, student.own_cols.cpu().numpy()]
    c = c[teacher.own_cols.cpu().numpy()]
    n1, n2 = len(teacher.own_cols), len(student.own_cols)
    _rewrite(student.body[0], teacher.body[0], a, c, (list(range(n1)), list(range(n2))))


def transplant_critic(
    student: HumanCritic | ZombieCritic, teacher: HumanCritic | ZombieCritic, config: dict[str, Any]
) -> None:
    _copy_rest(student, teacher, {"net.0"})
    A, c = vector_map("state", config)
    if isinstance(student, ZombieCritic):
        Az, cz = vector_map("zombie", config)
        A = np.block(
            [[A, np.zeros((A.shape[0], Az.shape[1]))], [np.zeros((Az.shape[0], A.shape[1])), Az]]
        )
        c = np.concatenate([c, cz])
    _rewrite(student.net[0], teacher.net[0], A, c)
