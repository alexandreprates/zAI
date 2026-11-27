"""CUDA graphs of the actors' decision step (docs/ARCHITECTURE.md section 4.8.3).

At rollout batch sizes an actor's `step` is bound by kernel launches, not by the GPU: about
100 small kernels take ~3-4 ms of CPU time whatever the batch, against ~1 ms of GPU time (P100,
measured 2026-11-27). A CUDA graph replays the whole step with one launch. The graph is
captured for fixed input shapes and reads the parameters by address, so optimizer updates
and `load_state_dict` (both in place) keep it valid.

`GraphedActor.run` returns the graph's static output tensors, which the next `run` overwrites:
consume them (sample, compute log-probabilities, clone the hidden state) before calling it
again. Work queued on the same stream in between is ordered correctly.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from .human import HumanActor


class GraphedActor:
    def __init__(
        self,
        actor: nn.Module,
        obs: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        warmup: int = 3,
    ):
        """Captures `actor.step` on inputs shaped like `obs`, `h` (and `mask` for the human)."""
        self.actor = actor
        self.human = isinstance(actor, HumanActor)
        self.obs, self.h = obs.clone(), h.clone()
        self.mask = mask.clone() if mask is not None else None
        stream = torch.cuda.Stream(obs.device)
        stream.wait_stream(torch.cuda.current_stream(obs.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(warmup):  # cuDNN and the allocator settle before capture
                self._step()
        torch.cuda.current_stream(obs.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.out = self._step()

    def _step(self) -> tuple[torch.Tensor, ...]:
        if self.human:
            move, interact, hn = self.actor.step(self.obs, self.h, self.mask)
            return move.logits, interact.logits, hn
        dist, hn = self.actor.step(self.obs, self.h)
        return dist.logits, hn

    @torch.no_grad()
    def run(
        self, obs: torch.Tensor, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """`(move logits, interact logits, h')` for the human, `(move logits, h')` for the
        zombies, as static tensors (see the module docstring)."""
        self.obs.copy_(obs)
        self.h.copy_(h)
        if self.mask is not None:
            self.mask.copy_(mask)
        self.graph.replay()
        return self.out


def dist(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits, validate_args=False)


def usable(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()
