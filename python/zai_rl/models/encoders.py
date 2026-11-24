"""Building blocks shared by the actor networks."""

import torch
from torch import nn


def mlp(in_dim: int, hidden: int, out_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)]
    layers.append(nn.ReLU())
    if out_dim is not None:
        layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class SetEncoder(nn.Module):
    """Per-slot MLP followed by masked mean and max pooling (permutation invariant)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = mlp(in_dim, hidden)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, F], mask: [B, S] (bool)
        h = self.net(x)
        m = mask.unsqueeze(-1).to(h.dtype)
        count = m.sum(dim=1).clamp(min=1.0)
        mean = (h * m).sum(dim=1) / count
        neg = torch.finfo(h.dtype).min
        mx = h.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values
        mx = torch.where(mask.any(dim=1, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)


class RayEncoder(nn.Module):
    """1D convolution over the ray fan (circular padding), keeping direction information."""

    def __init__(self, rays: int, channels: int = 3, hidden: int = 32, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(hidden * (rays // 4), out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        # rays: [B, R, C] -> conv over R
        h = self.conv(rays.transpose(1, 2))
        return self.proj(h.flatten(1))
