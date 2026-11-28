"""Pieces shared by the human PPO and the zombie MAPPO trainers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Running mean/variance (parallel algorithm), stored as buffers for checkpoints."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.tensor(eps))

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().flatten().double()
        b_mean, b_var, b_count = x.mean(), x.var(unbiased=False), x.numel()
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += (delta * b_count / total).to(self.mean.dtype)
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var.copy_((m2 / total).to(self.var.dtype))
        self.count.copy_(total.to(self.count.dtype))

    @property
    def std(self) -> torch.Tensor:
        return self.var.clamp(min=1e-8).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


@dataclass
class EpisodeStats:
    returns: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns. `dones[t]` ends the episode after step t (no bootstrap
    across it); truncation bootstraps must already be folded into `rewards`."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def bptt_sequences(x: torch.Tensor, L: int) -> torch.Tensor:
    """`[T, B, ...] -> [chunks * B, L, ...]`, sequence `c * B + b` being chunk `c` of row `b`."""
    T, B = x.shape[:2]
    chunks = T // L
    x = x.reshape(chunks, L, B, *x.shape[2:])
    return x.transpose(1, 2).reshape(chunks * B, L, *x.shape[3:])


class Checkpointable(Protocol):
    def state_dict(self) -> dict[str, Any]: ...


def checkpoint_obs_version(sd: dict[str, Any]) -> int:
    """Observation layout a checkpoint was trained on (checkpoints before M6.1 have none: v1)."""
    return int(sd.get("obs_version", 1))


def require_obs_version(sd: dict[str, Any], expected: int, what: str) -> None:
    """Fails when `sd` was trained on another observation layout than `expected`."""
    found = checkpoint_obs_version(sd)
    if found != expected:
        raise ValueError(
            f"{what} was trained on observation layout v{found}, this code uses v{expected}; "
            "layout v1 checkpoints can only play as frozen policies (zai_env.legacy) or be "
            "distilled (python -m zai_rl.distill)"
        )


def save_checkpoint(path: Path, algo: Checkpointable, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({**algo.state_dict(), **extra}, tmp)
    tmp.replace(path)


def scheduled_lr(lr: float, progress: float, lr_floor: float = 0.0, lr_scale: float = 1.0) -> float:
    """Linear decay of `lr` over `progress` in [0, 1], stopping at `lr_floor * lr`."""
    return lr * lr_scale * max(lr_floor, 1.0 - progress)
