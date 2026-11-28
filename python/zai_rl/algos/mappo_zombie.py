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

from zai_env.layout import FAR_M, STATE

from ..envs import ZombieBatch, ZombieStep, ZombieTrainingEnv
from ..models import ZombieActor, ZombieCritic, graphs
from ..profiling import timer
from .common import (
    EpisodeStats,
    RunningMeanStd,
    bptt_sequences,
    gae,
    require_obs_version,
    scheduled_lr,
)

NEAREST_EVERY = 8  # decisions between samples of the train/nearest_zombie_m metric


def nearest_zombie_m(state: np.ndarray) -> np.ndarray:
    """Distance from the human to the nearest zombie per env, from the positions in the global
    state `[E, S]` (`dist_to_human` saturates at 100 m in layout v2)."""
    dx = STATE.field(state, "zombies", "x") - STATE.scalar(state, "human", "x")[:, None]
    dy = STATE.field(state, "zombies", "y") - STATE.scalar(state, "human", "y")[:, None]
    valid = STATE.field(state, "zombies", "valid") > 0.5
    return np.where(valid, np.hypot(dx, dy), np.inf).min(axis=1) * FAR_M


@dataclass
class MAPPOConfig:
    num_envs: int = 256
    rollout: int = 128
    bptt: int = 64
    epochs: int = 3
    minibatches: int = 8
    lr: float = 3e-4
    # The linear lr decay stops at this fraction of `lr` (0: decays to 0).
    lr_floor: float = 0.0
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
        ro = ZombieRollout(self, env.agent_env, h, start, ep_return)
        ro.begin()
        for _ in range(self.cfg.rollout):
            step = env.step(ro.act(batch.obs, batch.state))
            ro.record(step, stats)
            batch = step.next
        return ro.finish(batch.obs, batch.state), batch, ro.h, ro.start

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
        T, A = rollout["rewards"].shape
        L = cfg.bptt
        chunks = T // L
        n = chunks * A
        lr = scheduled_lr(cfg.lr, progress, cfg.lr_floor, lr_scale)
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
        for _ in range(cfg.epochs if epochs is None else epochs):
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

                with timer("update_gather"):
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
            "obs_version": self.actor.obs_version,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        require_obs_version(sd, self.actor.obs_version, "the checkpoint")
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.value_rms.load_state_dict(sd["value_rms"])
        self.optimizer.load_state_dict(sd["optimizer"])


class ZombieRollout:
    """One rollout of the zombie learner, decision by decision (see `HumanRollout`). Agents
    `[A]` sit on a flat axis with `agent_env [A]`, the index of each agent's env in the
    learner's env axis `[E]` (per-env arrays such as `state` and `terminated` are `[E]`)."""

    def __init__(
        self,
        algo: ZombieMAPPO,
        agent_env: np.ndarray,
        h: torch.Tensor,
        start: torch.Tensor,
        ep_return: np.ndarray,
        graph: bool = False,
    ):
        """`graph`: replay the actor's step as a CUDA graph (`models.graphs`) on CUDA."""
        self.algo = algo
        self.agent_env_np = agent_env
        self.agent_env = algo._t(agent_env, torch.long)
        self.h, self.start, self.ep_return = h, start, ep_return
        self.graph = graph and graphs.usable(algo.device)
        self._graphed: graphs.GraphedActor | None = None

    def begin(self) -> None:
        self.t = 0
        self.buf: dict[str, list[torch.Tensor]] = {
            k: []
            for k in ("obs", "state", "actions", "logp", "values", "rewards", "dones", "starts")
        }
        self.h0: list[torch.Tensor] = []
        self.nearest: list[float] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        actions = self.act_device(obs, state)
        with timer("policy"):
            return actions.cpu().numpy()

    @torch.no_grad()
    def act_device(self, obs: np.ndarray, state: np.ndarray) -> torch.Tensor:
        """`act` without the transfer: the actions stay on the device (no synchronization), so
        a caller can queue more work before fetching them."""
        algo = self.algo
        if self.t % NEAREST_EVERY == 0:
            self.nearest.append(float(nearest_zombie_m(state).mean()))
        with timer("policy"):
            obs_t = algo._t(obs)
            state_t = algo._t(state)
            self.h = self.h * (~self.start).unsqueeze(1)
            if self.t % algo.cfg.bptt == 0:
                self.h0.append(self.h.clone())
            if self.graph:
                if self._graphed is None:
                    self._graphed = graphs.GraphedActor(algo.actor, obs_t, self.h)
                logits, hn = self._graphed.run(obs_t, self.h)
                dist, self.h_new = graphs.dist(logits), hn.clone()
            else:
                dist, self.h_new = algo.actor.step(obs_t, self.h)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            value = algo.value(state_t[self.agent_env], obs_t)
        self.pending = (obs_t, state_t, actions, logp, value)
        return actions

    @torch.no_grad()
    def record(self, step: ZombieStep, stats: EpisodeStats) -> None:
        algo, cfg = self.algo, self.algo.cfg
        agent_env = self.agent_env_np
        obs_t, state_t, actions, logp, value = self.pending
        reward = step.reward.astype(np.float32)
        self.ep_return += reward
        env_done = step.terminated | step.truncated
        done = env_done[agent_env]
        reward_t = algo._t(reward)
        if step.truncated.any():
            with timer("policy"):
                idx = np.nonzero(step.truncated[agent_env])[0]
                v_final = algo.value(
                    algo._t(step.final_state[agent_env[idx]]),
                    algo._t(step.final_obs[idx]),
                )
                reward_t[idx] += cfg.gamma * v_final
        for k, v in (
            ("obs", obs_t),
            ("state", state_t),
            ("actions", actions),
            ("logp", logp),
            ("values", value),
            ("rewards", reward_t),
            ("dones", algo._t(done)),
            ("starts", self.start),
        ):
            self.buf[k].append(v)
        if env_done.any():
            for e in np.nonzero(env_done)[0]:
                agents = agent_env == e
                stats.returns.append(float(self.ep_return[agents].mean()))
                stats.lengths.append(float(step.episode_length[e]))
                stats.outcomes.append(int(step.outcome[e]))
            self.ep_return[done] = 0.0
        self.start = algo._t(done, torch.bool)
        self.h = self.h_new
        self.t += 1

    @torch.no_grad()
    def finish(self, next_obs: np.ndarray, next_state: np.ndarray) -> dict[str, torch.Tensor]:
        algo, cfg = self.algo, self.algo.cfg
        rollout = {k: torch.stack(v) for k, v in self.buf.items()}
        rollout["h0"] = torch.stack(self.h0)  # [T/L, A, H]
        last_value = algo.value(algo._t(next_state)[self.agent_env], algo._t(next_obs))
        adv, ret = gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        rollout["advantages"], rollout["returns"] = adv, ret
        rollout["agent_env"] = self.agent_env
        # Mean distance from the human to the nearest zombie over the rollout (diagnostic).
        rollout["nearest_zombie_m"] = torch.tensor(float(np.mean(self.nearest)))
        return rollout
