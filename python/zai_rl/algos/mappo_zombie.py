"""Recurrent MAPPO for the zombie team (docs/ARCHITECTURE.md sections 4.1 and 4.6).

- Actor: `ZombieActor` (GRU) shared by every zombie, trained with truncated BPTT over chunks of
  `bptt` decisions whose initial hidden states are stored during the rollout.
- Critic: `ZombieCritic` on `concat(env state, own observation)`, one value per zombie, with
  value normalization by running return statistics.
- Rollouts use the flat agent axis of `ZombieTrainingEnv` (`[T, A, ...]`); per-env arrays
  (`state`, `done`) are stored per env and broadcast through `agent_env`.
- Per-agent rewards (team reward plus per-zombie distance shaping) and per-agent GAE; episodes
  cut by the time cap bootstrap with the critic's value of the final state.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..envs import ZombieBatch, ZombieTrainingEnv
from ..models import ZombieActor, ZombieCritic
from .common import EpisodeStats, RunningMeanStd, bptt_sequences, gae


@dataclass
class MAPPOConfig:
    num_envs: int = 256
    rollout: int = 128
    bptt: int = 64
    epochs: int = 3
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_start: float = 0.01
    ent_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128

    def __post_init__(self):
        if self.rollout % self.bptt:
            raise ValueError("rollout must be a multiple of bptt")


class ZombieMAPPO:
    def __init__(self, cfg: MAPPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = ZombieActor(hidden=cfg.hidden).to(device)
        self.critic = ZombieCritic().to(device)
        self.value_rms = RunningMeanStd().to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5
        )

    def _t(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    @torch.no_grad()
    def value(self, state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Denormalized values; `state` is already gathered per agent."""
        return self.value_rms.denormalize(self.critic(state, obs))

    # --- rollout ------------------------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        env: ZombieTrainingEnv,
        batch: ZombieBatch,
        h: torch.Tensor,
        start: torch.Tensor,
        ep_return: np.ndarray,
        stats: EpisodeStats,
    ) -> tuple[dict[str, torch.Tensor], ZombieBatch, torch.Tensor, torch.Tensor]:
        """Collects `rollout` decisions. `h [A, H]` and `start [A]` carry over between calls;
        `ep_return [A]` accumulates each zombie's return; finished episodes add the mean of
        their zombies' returns to `stats`."""
        cfg = self.cfg
        T, L = cfg.rollout, cfg.bptt
        agent_env = self._t(env.agent_env, torch.long)
        buf: dict[str, Any] = {
            k: []
            for k in (
                "obs",
                "state",
                "actions",
                "logp",
                "values",
                "rewards",
                "dones",
                "starts",
            )
        }
        h0 = []
        for t in range(T):
            obs = self._t(batch.obs)
            state = self._t(batch.state)
            h = h * (~start).unsqueeze(1)
            if t % L == 0:
                h0.append(h.clone())
            dist, h_new = self.actor.step(obs, h)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            value = self.value(state[agent_env], obs)

            step = env.step(actions.cpu().numpy())
            reward = step.reward.astype(np.float32)
            ep_return += reward
            env_done = step.terminated | step.truncated
            done = env_done[env.agent_env]
            reward_t = self._t(reward)
            if step.truncated.any():
                idx = np.nonzero(step.truncated[env.agent_env])[0]
                v_final = self.value(
                    self._t(step.final_state[env.agent_env[idx]]), self._t(step.final_obs[idx])
                )
                reward_t[idx] += cfg.gamma * v_final

            for k, v in (
                ("obs", obs),
                ("state", state),
                ("actions", actions),
                ("logp", logp),
                ("values", value),
                ("rewards", reward_t),
                ("dones", self._t(done)),
                ("starts", start),
            ):
                buf[k].append(v)

            if env_done.any():
                for e in np.nonzero(env_done)[0]:
                    agents = env.agent_env == e
                    stats.returns.append(float(ep_return[agents].mean()))
                    stats.lengths.append(float(step.episode_length[e]))
                    stats.outcomes.append(int(step.outcome[e]))
                ep_return[done] = 0.0
            start = self._t(done, torch.bool)
            h = h_new
            batch = step.next

        rollout = {k: torch.stack(v) for k, v in buf.items()}
        rollout["h0"] = torch.stack(h0)  # [T/L, A, H]
        last_value = self.value(self._t(batch.state)[agent_env], self._t(batch.obs))
        adv, ret = gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        rollout["advantages"], rollout["returns"] = adv, ret
        rollout["agent_env"] = agent_env
        return rollout, batch, h, start

    # --- update -------------------------------------------------------------------------
    def update(self, rollout: dict[str, torch.Tensor], progress: float) -> dict[str, float]:
        cfg = self.cfg
        T, A = rollout["rewards"].shape
        L = cfg.bptt
        chunks = T // L
        n = chunks * A
        lr = cfg.lr * max(0.0, 1.0 - progress)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * min(1.0, progress)

        self.value_rms.update(rollout["returns"])

        per_agent = ("obs", "actions", "logp", "advantages", "returns", "starts")
        data = {k: bptt_sequences(rollout[k], L) for k in per_agent}
        h0 = rollout["h0"].reshape(n, -1)
        # The env state stays per env ([T, E, S]); sequence `c * A + a` reads rows
        # `c * L + l` of env `agent_env[a]`.
        state, agent_env = rollout["state"], rollout["agent_env"]
        seq = torch.arange(n, device=self.device)
        seq_t0 = (seq // A) * L
        seq_env = agent_env[seq % A]
        offsets = torch.arange(L, device=self.device)

        stats: dict[str, list[float]] = {
            k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")
        }
        for _ in range(cfg.epochs):
            # The agent count depends on how envs split into groups, so minibatches may
            # differ in size by one sequence.
            for idx in torch.randperm(n, device=self.device).tensor_split(cfg.minibatches):
                dist = self.actor.sequence(data["obs"][idx], h0[idx], data["starts"][idx])
                logp = dist.log_prob(data["actions"][idx].reshape(-1)).reshape(len(idx), L)
                entropy = dist.entropy().mean()

                adv = data["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = logp - data["logp"][idx]
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()

                rows = seq_t0[idx].unsqueeze(1) + offsets  # [mb, L]
                mb_state = state[rows, seq_env[idx].unsqueeze(1)]  # [mb, L, S]
                v_pred = self.critic(
                    mb_state.reshape(-1, state.shape[-1]),
                    data["obs"][idx].reshape(-1, data["obs"].shape[-1]),
                ).reshape(len(idx), L)
                v_target = self.value_rms.normalize(data["returns"][idx])
                v_loss = 0.5 * (v_pred - v_target).pow(2).mean()

                loss = pg - ent_coef * entropy + cfg.vf_coef * v_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"].append(pg.item())
                    stats["value_loss"].append(v_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())

        ret, val = rollout["returns"], rollout["values"]
        explained = 1.0 - (ret - val).var().item() / max(ret.var().item(), 1e-8)
        out = {f"train/{k}": float(np.mean(v)) for k, v in stats.items()}
        out.update(
            {
                "train/lr": lr,
                "train/ent_coef": ent_coef,
                "train/explained_variance": explained,
                "train/value_mean": val.mean().item(),
            }
        )
        return out

    # --- checkpoints --------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_rms": self.value_rms.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "mappo_config": asdict(self.cfg),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.value_rms.load_state_dict(sd["value_rms"])
        self.optimizer.load_state_dict(sd["optimizer"])
