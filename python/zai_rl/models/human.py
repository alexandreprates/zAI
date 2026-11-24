"""Human actor (recurrent, masked two-head policy) and privileged critic."""

import torch
import zai_sim
from torch import nn
from torch.distributions import Categorical

from zai_env.layout import HUMAN

from .encoders import RayEncoder, SetEncoder, mlp, run_gru_segments

MOVE = zai_sim.MOVE_ACTIONS
INTERACT = zai_sim.INTERACT_ACTIONS


def _block(name: str) -> tuple[int, int, int]:
    b = HUMAN.blocks[name]
    return b["start"], b["slots"], len(b["fields"])


class HumanActor(nn.Module):
    """Entity encoders + ray conv + MLP + GRU; heads for movement (17) and interaction (3)."""

    def __init__(self, hidden: int = 256, entity_hidden: int = 64):
        super().__init__()
        self.self_slice = _block("self")
        self.zombie_slice = _block("zombies")
        self.food_slice = _block("food")
        self.house_slice = _block("houses")
        self.ray_slice = _block("rays")
        self.zombie_valid = HUMAN.blocks["zombies"]["fields"].index("valid")
        self.food_valid = HUMAN.blocks["food"]["fields"].index("valid")

        self.zombies = SetEncoder(self.zombie_slice[2], entity_hidden)
        self.food = SetEncoder(self.food_slice[2], entity_hidden)
        self.houses = SetEncoder(self.house_slice[2], entity_hidden)
        self.rays = RayEncoder(self.ray_slice[1], self.ray_slice[2])
        in_dim = (
            self.self_slice[2]
            + self.zombies.out_dim
            + self.food.out_dim
            + self.houses.out_dim
            + self.rays.out_dim
        )
        self.body = mlp(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.move_head = nn.Linear(hidden, MOVE)
        self.interact_head = nn.Linear(hidden, INTERACT)
        self.hidden_size = hidden
        for head in (self.move_head, self.interact_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _slots(obs: torch.Tensor, s: tuple[int, int, int]) -> torch.Tensor:
        start, slots, width = s
        return obs[:, start : start + slots * width].reshape(-1, slots, width)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        own = self._slots(obs, self.self_slice)[:, 0]
        zombies = self._slots(obs, self.zombie_slice)
        food = self._slots(obs, self.food_slice)
        houses = self._slots(obs, self.house_slice)
        rays = self._slots(obs, self.ray_slice)
        x = torch.cat(
            [
                own,
                self.zombies(zombies, zombies[..., self.zombie_valid] > 0.5),
                self.food(food, food[..., self.food_valid] > 0.5),
                self.houses(houses, houses.abs().sum(-1) > 0),
                self.rays(rays),
            ],
            dim=-1,
        )
        return self.body(x)

    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def heads(self, out: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, Categorical]:
        neg = torch.finfo(out.dtype).min
        move_logits = self.move_head(out).masked_fill(~mask[:, :MOVE], neg)
        interact_logits = self.interact_head(out).masked_fill(~mask[:, MOVE:], neg)
        return Categorical(logits=move_logits), Categorical(logits=interact_logits)

    def step(
        self, obs: torch.Tensor, h: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, Categorical, torch.Tensor]:
        """One decision: returns (movement dist, interaction dist, new hidden state).

        `mask` is the env's action mask `[B, MOVE + INTERACT]` (bool).
        """
        out, hn = self.gru(self.encode(obs).unsqueeze(1), h.unsqueeze(0).contiguous())
        move, interact = self.heads(out[:, 0], mask)
        return move, interact, hn[0]

    def sequence(
        self, obs: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, Categorical]:
        """Distributions for whole sequences `[B, L, ...]` (training).

        `starts[:, t]` marks the first decision of an episode, where the memory is cleared.
        The encoder runs once over all steps; the GRU runs over segments between resets.
        """
        B, L, _ = obs.shape
        x = self.encode(obs.reshape(B * L, -1)).reshape(B, L, -1)
        out = run_gru_segments(self.gru, x, h0, starts).reshape(B * L, -1)
        return self.heads(out, mask.reshape(B * L, -1))


class HumanCritic(nn.Module):
    """Value of the privileged global state (training only)."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zai_sim.STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
