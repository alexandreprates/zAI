"""Zombie actor (recurrent, shared by every zombie) and centralized critic (M5).

The actor sees only the zombie's own observation (hearing, peers, houses, rays); it has no
agent id, so the same parameters drive any number of zombies. The critic is MAPPO's
agent-specific global state: the privileged env state concatenated with the zombie's own
observation, giving one value per zombie.
"""

import torch
import zai_sim
from torch import nn
from torch.distributions import Categorical

from zai_env.layout import OBS_VERSION, ZOMBIE, Layout
from zai_env.legacy import STATE_DIM_V1, ZOMBIE_OBS_DIM_V1, ZOMBIE_V1

from .encoders import RayEncoder, SetEncoder, mlp, run_gru_segments
from .human import own_columns

MOVE = zai_sim.MOVE_ACTIONS


def _block(layout: Layout, name: str) -> tuple[int, int, int]:
    b = layout.blocks[name]
    return b["start"], b["slots"], len(b["fields"])


class ZombieActor(nn.Module):
    """Entity encoders + ray conv + MLP + GRU; one movement head (17 actions, no mask)."""

    def __init__(self, hidden: int = 128, entity_hidden: int = 32, obs_version: int = OBS_VERSION):
        super().__init__()
        layout = ZOMBIE if obs_version == OBS_VERSION else ZOMBIE_V1
        self.obs_version = obs_version
        self.self_slice = _block(layout, "self")
        self.heard_slice = _block(layout, "heard")
        self.peer_slice = _block(layout, "peers")
        self.house_slice = _block(layout, "houses")
        self.ray_slice = _block(layout, "rays")
        self.heard_valid = layout.blocks["heard"]["fields"].index("valid")
        self.peer_valid = layout.blocks["peers"]["fields"].index("valid")
        house_fields = layout.blocks["houses"]["fields"]
        self.house_valid = house_fields.index("valid") if "valid" in house_fields else None
        self.register_buffer("own_cols", torch.tensor(own_columns(layout, obs_version)), False)

        self.heard = SetEncoder(self.heard_slice[2], entity_hidden)
        self.peers = SetEncoder(self.peer_slice[2], entity_hidden)
        self.houses = SetEncoder(self.house_slice[2], entity_hidden)
        self.rays = RayEncoder(self.ray_slice[1], self.ray_slice[2], out_dim=64)
        in_dim = (
            len(self.own_cols)
            + self.heard.out_dim
            + self.peers.out_dim
            + self.houses.out_dim
            + self.rays.out_dim
        )
        self.body = mlp(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.move_head = nn.Linear(hidden, MOVE)
        self.hidden_size = hidden
        nn.init.orthogonal_(self.move_head.weight, gain=0.01)
        nn.init.zeros_(self.move_head.bias)

    @staticmethod
    def _slots(obs: torch.Tensor, s: tuple[int, int, int]) -> torch.Tensor:
        start, slots, width = s
        return obs[:, start : start + slots * width].reshape(-1, slots, width)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        own = self._slots(obs, self.self_slice)[:, 0, self.own_cols]
        heard = self._slots(obs, self.heard_slice)
        peers = self._slots(obs, self.peer_slice)
        houses = self._slots(obs, self.house_slice)
        rays = self._slots(obs, self.ray_slice)
        if self.house_valid is None:  # layout v1 has no valid flag
            house_mask = houses.abs().sum(-1) > 0
        else:
            house_mask = houses[..., self.house_valid] > 0.5
        x = torch.cat(
            [
                own,
                self.heard(heard, heard[..., self.heard_valid] > 0.5),
                self.peers(peers, peers[..., self.peer_valid] > 0.5),
                self.houses(houses, house_mask),
                self.rays(rays),
            ],
            dim=-1,
        )
        return self.body(x)

    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def step(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """One decision for `B` zombies: returns (movement dist, new hidden state)."""
        out, hn = self.gru(self.encode(obs).unsqueeze(1), h.unsqueeze(0).contiguous())
        return Categorical(logits=self.move_head(out[:, 0])), hn[0]

    def sequence(self, obs: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor) -> Categorical:
        """Distributions for whole sequences `[B, L, ...]` (training), flattened to `B * L`."""
        B, L, _ = obs.shape
        x = self.encode(obs.reshape(B * L, -1)).reshape(B, L, -1)
        out = run_gru_segments(self.gru, x, h0, starts).reshape(B * L, -1)
        return Categorical(logits=self.move_head(out))


class ZombieCritic(nn.Module):
    """Value of `concat(env state, own observation)` per zombie (training only)."""

    def __init__(self, hidden: int = 512, obs_version: int = OBS_VERSION):
        super().__init__()
        if obs_version == OBS_VERSION:
            in_dim = zai_sim.STATE_DIM + zai_sim.ZOMBIE_OBS_DIM
        else:
            in_dim = STATE_DIM_V1 + ZOMBIE_OBS_DIM_V1
        self.obs_version = obs_version
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, obs], dim=-1)).squeeze(-1)
