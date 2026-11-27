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


def run_gru_segments(
    gru: nn.GRU, x: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor
) -> torch.Tensor:
    """Runs `gru` over `x [B, L, F]` from `h0 [B, H]`, clearing the memory of row `b` at
    every `starts[b, t]` (first decision of an episode), which is equivalent to stepping one
    decision at a time. Returns the outputs `[B, L, H]`.

    Every episode segment of a row becomes its own row, left-aligned and zero-padded to `L`:
    the first segment of row `b` starts from `h0[b]` (zeroed if `starts[b, 0]`), later ones
    from zeros. The GRU then runs once over `R = B + (restarts after t = 0)` rows, instead of
    once per time step at which any row restarts (in large minibatches that is nearly every
    step, M6.0 profiling in docs/reports/throughput_profile.md). Padding only follows a
    segment's last step, so it never changes the outputs that are read back."""
    B, L, F = x.shape
    h_first = h0 * (~starts[:, 0]).unsqueeze(1)
    later = starts[:, 1:]
    if not bool(later.any()):
        out, _ = gru(x, h_first.unsqueeze(0).contiguous())
        return out
    seg = torch.zeros(B, L, dtype=torch.long, device=x.device)
    seg[:, 1:] = later.long().cumsum(dim=1)  # segment index of (b, t) within row b
    nseg = seg[:, -1] + 1
    row_off = torch.cumsum(nseg, dim=0) - nseg
    rows = int(nseg.sum())
    t = torch.arange(L, device=x.device).expand(B, L)
    is_start = starts.clone()
    is_start[:, 0] = True
    pos = t - torch.where(is_start, t, torch.zeros_like(t)).cummax(dim=1).values
    flat = ((row_off.unsqueeze(1) + seg) * L + pos).reshape(-1)  # [B * L] into [R * L]
    xs = x.new_zeros(rows * L, F).index_copy(0, flat, x.reshape(B * L, F)).reshape(rows, L, F)
    hs = h0.new_zeros(rows, h0.shape[1]).index_copy(0, row_off, h_first)
    out, _ = gru(xs, hs.unsqueeze(0).contiguous())
    return out.reshape(rows * L, -1)[flat].reshape(B, L, -1)
