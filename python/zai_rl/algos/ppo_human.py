"""Recurrent PPO for the human (docs/ARCHITECTURE.md section 4.1).

- Actor: `HumanActor` (GRU), trained with truncated BPTT over chunks of `bptt` decisions whose
  initial hidden states are stored during the rollout.
- Critic: `HumanCritic` on the privileged global state (no recurrence), with value
  normalization by running return statistics.
- GAE; episodes cut by the time cap bootstrap with the critic's value of the final state
  (folded into the last reward), terminal episodes do not.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..envs import HumanBatch, HumanStep, HumanTrainingEnv
from ..models import HumanActor, HumanCritic
from ..profiling import timer
from .common import (
    EpisodeStats,
    RunningMeanStd,
    bptt_sequences,
    gae,
    require_obs_version,
    save_checkpoint,
)

__all__ = [
    "EpisodeStats",
    "HumanPPO",
    "HumanRollout",
    "PPOConfig",
    "RunningMeanStd",
    "gae",
    "save_checkpoint",
]


@dataclass
class PPOConfig:
    num_envs: int = 128
    rollout: int = 128
    bptt: int = 64
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_start: float = 0.01
    ent_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256

    def __post_init__(self):
        if self.rollout % self.bptt:
            raise ValueError("rollout must be a multiple of bptt")
        if (self.rollout // self.bptt * self.num_envs) % self.minibatches:
            raise ValueError("sequences per rollout must be divisible by minibatches")


class HumanPPO:
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = HumanActor(hidden=cfg.hidden).to(device)
        self.critic = HumanCritic().to(device)
        self.value_rms = RunningMeanStd().to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5
        )

    # --- acting -------------------------------------------------------------------------
    def _t(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_rms.denormalize(self.critic(state))

    # --- rollout ------------------------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        env: HumanTrainingEnv,
        batch: HumanBatch,
        h: torch.Tensor,
        start: torch.Tensor,
        ep_return: np.ndarray,
        stats: EpisodeStats,
    ) -> tuple[dict[str, torch.Tensor], HumanBatch, torch.Tensor, torch.Tensor]:
        ro = HumanRollout(self, h, start, ep_return)
        ro.begin()
        for _ in range(self.cfg.rollout):
            step = env.step(ro.act(batch.obs, batch.mask, batch.state))
            ro.record(step, stats)
            batch = step.next
        return ro.finish(batch.state), batch, ro.h, ro.start

    # --- update -------------------------------------------------------------------------
    def update(
        self,
        rollout: dict[str, torch.Tensor],
        progress: float,
        epochs: int | None = None,
        lr_scale: float = 1.0,
    ) -> dict[str, float]:
        """PPO update; `epochs` and `lr_scale` override the config (league imbalance guard)."""
        cfg = self.cfg
        T, E = rollout["rewards"].shape
        L = cfg.bptt
        chunks = T // L
        lr = cfg.lr * max(0.0, 1.0 - progress) * lr_scale
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * min(1.0, progress)

        self.value_rms.update(rollout["returns"])

        data = {k: bptt_sequences(rollout[k], L) for k in rollout if k != "h0"}
        h0 = rollout["h0"].reshape(chunks * E, -1)
        n = chunks * E
        mb_size = n // cfg.minibatches

        stats: dict[str, list[float]] = {
            k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")
        }
        for _ in range(cfg.epochs if epochs is None else epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.minibatches):
                idx = perm[i * mb_size : (i + 1) * mb_size]
                move, interact = self.actor.sequence(
                    data["obs"][idx], h0[idx], data["starts"][idx], data["mask"][idx]
                )
                a = data["actions"][idx].reshape(-1, 2)
                logp = (move.log_prob(a[:, 0]) + interact.log_prob(a[:, 1])).reshape(len(idx), L)
                entropy = (move.entropy() + interact.entropy()).mean()

                adv = data["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = logp - data["logp"][idx]
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()

                state = data["state"][idx].reshape(-1, data["state"].shape[-1])
                v_pred = self.critic(state).reshape(len(idx), L)
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
            "ppo_config": asdict(self.cfg),
            "obs_version": self.actor.obs_version,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        require_obs_version(sd, self.actor.obs_version, "the checkpoint")
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.value_rms.load_state_dict(sd["value_rms"])
        self.optimizer.load_state_dict(sd["optimizer"])


class HumanRollout:
    """One rollout of the human learner, decision by decision, so that a caller that steps the
    environment itself (the league trainer, where both sides act in the same envs) can drive
    it: `begin()`, then `rollout` times `act(...)` and `record(step, stats)`, then
    `finish(next_state)`. `h [E, H]`, `start [E]` and `ep_return [E]` carry over between
    rollouts."""

    def __init__(self, algo: HumanPPO, h: torch.Tensor, start: torch.Tensor, ep_return: np.ndarray):
        self.algo = algo
        self.h, self.start, self.ep_return = h, start, ep_return

    def begin(self) -> None:
        self.t = 0
        self.buf: dict[str, list[torch.Tensor]] = {
            k: []
            for k in (
                "obs",
                "mask",
                "state",
                "actions",
                "logp",
                "values",
                "rewards",
                "dones",
                "starts",
            )
        }
        self.h0: list[torch.Tensor] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, state: np.ndarray) -> np.ndarray:
        algo = self.algo
        with timer("policy"):
            obs_t = algo._t(obs)
            mask_t = algo._t(mask, torch.bool)
            state_t = algo._t(state)
            self.h = self.h * (~self.start).unsqueeze(1)
            if self.t % algo.cfg.bptt == 0:
                self.h0.append(self.h.clone())
            move, interact, self.h_new = algo.actor.step(obs_t, self.h, mask_t)
            a_move, a_inter = move.sample(), interact.sample()
            logp = move.log_prob(a_move) + interact.log_prob(a_inter)
            value = algo.value(state_t)
            actions = torch.stack([a_move, a_inter], dim=1)
            actions_np = actions.cpu().numpy()
        self.pending = (obs_t, mask_t, state_t, actions, logp, value)
        return actions_np

    @torch.no_grad()
    def record(self, step: HumanStep, stats: EpisodeStats) -> None:
        algo, cfg = self.algo, self.algo.cfg
        obs_t, mask_t, state_t, actions, logp, value = self.pending
        reward = step.reward.astype(np.float32)
        self.ep_return += reward
        done = step.terminated | step.truncated
        reward_t = algo._t(reward)
        if step.truncated.any():
            with timer("policy"):
                idx = np.nonzero(step.truncated)[0]
                v_final = algo.value(algo._t(step.final_state[idx]))
                reward_t[idx] += cfg.gamma * v_final
        for k, v in (
            ("obs", obs_t),
            ("mask", mask_t),
            ("state", state_t),
            ("actions", actions),
            ("logp", logp),
            ("values", value),
            ("rewards", reward_t),
            ("dones", algo._t(done)),
            ("starts", self.start),
        ):
            self.buf[k].append(v)
        if done.any():
            for i in np.nonzero(done)[0]:
                stats.returns.append(float(self.ep_return[i]))
                stats.lengths.append(float(step.episode_length[i]))
                stats.outcomes.append(int(step.outcome[i]))
                self.ep_return[i] = 0.0
        self.start = algo._t(done, torch.bool)
        self.h = self.h_new
        self.t += 1

    @torch.no_grad()
    def finish(self, next_state: np.ndarray) -> dict[str, torch.Tensor]:
        algo, cfg = self.algo, self.algo.cfg
        rollout = {k: torch.stack(v) for k, v in self.buf.items()}
        rollout["h0"] = torch.stack(self.h0)  # [T/L, E, H]
        last_value = algo.value(algo._t(next_state))
        adv, ret = gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        rollout["advantages"], rollout["returns"] = adv, ret
        return rollout
