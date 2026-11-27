"""Distillation of a trained policy into a fresh student on the current observation layout
(M6.1, docs/ARCHITECTURE.md section 4.8.2).

The teacher is any checkpoint of either side, usually one trained on layout v1, which sees its
own inputs through `zai_env.legacy`. The student has the teacher's architecture on the current
layout and is trained with on-policy DAgger: each env's episode is driven by the teacher with
probability `beta` (1 -> 0 over the first `beta_fraction` of the steps), otherwise by the
student, and the student always runs on the visited states. Losses, over BPTT chunks:

- actor: `KL(teacher || student)` of the masked action distributions (both human heads);
- critic: squared error to the teacher's normalized value; `value_rms` is the teacher's, so the
  student's values mean the same returns.

Half of the zombie-count groups play against the scripted bot of the other side, half against
`opponent` (a checkpoint of the other side, any layout) when it is given. The result is saved in
the trainer's checkpoint format (`/checkpoints/<side>/<run_id>/`), so it can be fine-tuned with
`init_checkpoint` or used as a league starter. Acceptance (4.8.2): with `final_eval_episodes`
per count on fixed seeds, the student's outcome rates against each opponent are within
`accept_points` of the teacher's, and the mean per-decision KL is below `accept_kl`.

    python -m zai_rl.distill --config configs/train/distill/human_s0small.yaml
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.distributions import Categorical, kl_divergence

from zai_env.config import deep_merge, load_config
from zai_env.layout import OBS_VERSION
from zai_env.legacy import adapt_v1_human, adapt_v1_state, adapt_v1_zombie

from .algos.common import RunningMeanStd, bptt_sequences, checkpoint_obs_version, save_checkpoint
from .algos.mappo_zombie import MAPPOConfig, ZombieMAPPO
from .algos.ppo_human import HumanPPO, PPOConfig
from .bots import ScriptedHuman, ScriptedZombies
from .envs import HumanTrainingEnv, ZombieTrainingEnv
from .metrics import MetricsLogger
from .models import HumanActor, HumanCritic, ZombieActor, ZombieCritic
from .policy import (
    RecurrentHumanPolicy,
    RecurrentZombiePolicy,
    load_human_actor,
    load_zombie_actor,
)
from .train import OUTCOME_KEYS, resolve
from .train_zombie import catch_rates

log = logging.getLogger("zai_rl.distill")


@dataclass
class DistillConfig:
    run_id: str | None = None
    side: str = "human"  # human | zombie
    teacher: str = ""  # checkpoint of `side` (any observation layout)
    opponent: str | None = None  # checkpoint of the other side for half of the groups
    env_config: str = "configs/env/s0_small.yaml"
    zombie_counts: list[int] = field(default_factory=lambda: [3, 6, 9])
    total_steps: int = 10_000_000
    num_envs: int = 1024
    rollout: int = 128
    bptt: int = 64
    epochs: int = 2
    minibatches: int = 8
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    beta_fraction: float = 0.3
    seed: int = 0
    num_threads: int | None = None
    torch_threads: int | None = None
    device: str = "auto"
    eval_every: int = 20
    eval_episodes: int = 128  # per zombie count and opponent, during the run
    final_eval_episodes: int = 512  # per zombie count and opponent, for acceptance
    accept_points: float = 0.03
    accept_kl: float = 0.05
    checkpoint_every: int = 20

    @classmethod
    def load(cls, path: Path | None, overrides: dict[str, Any]) -> "DistillConfig":
        raw: dict[str, Any] = {}
        if path is not None:
            raw = yaml.safe_load(path.read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown distill config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        if cfg.side not in ("human", "zombie"):
            raise ValueError("side must be 'human' or 'zombie'")
        if not cfg.teacher:
            raise ValueError("teacher checkpoint is required")
        if cfg.rollout % cfg.bptt:
            raise ValueError("rollout must be a multiple of bptt")
        return cfg


def kl(teacher_logits: torch.Tensor, student: Categorical) -> torch.Tensor:
    """Mean `KL(teacher || student)` over decisions; `teacher_logits` is `[B, L, actions]`."""
    return kl_divergence(Categorical(logits=teacher_logits.flatten(0, 1)), student).mean()


class Teacher:
    """Frozen actor and critic of a checkpoint with the adapters its layout needs."""

    def __init__(self, path: Path, side: str, device: torch.device):
        sd = torch.load(path, map_location=device, weights_only=False)
        self.sd = sd
        self.version = checkpoint_obs_version(sd)
        if side == "human":
            self.actor: nn.Module = HumanActor(
                hidden=sd["ppo_config"]["hidden"], obs_version=self.version
            )
            self.critic: nn.Module = HumanCritic(obs_version=self.version)
        else:
            self.actor = ZombieActor(hidden=sd["mappo_config"]["hidden"], obs_version=self.version)
            self.critic = ZombieCritic(obs_version=self.version)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.value_rms = RunningMeanStd()
        self.value_rms.load_state_dict(sd["value_rms"])
        for m in (self.actor, self.critic, self.value_rms):
            m.to(device).eval()
        self.legacy = self.version != OBS_VERSION

    def obs(self, obs: np.ndarray, config: dict[str, Any], side: str) -> np.ndarray:
        if not self.legacy:
            return obs
        return adapt_v1_human(obs, config) if side == "human" else adapt_v1_zombie(obs, config)

    def state(self, state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        return adapt_v1_state(state, config) if self.legacy else state


def student_for(cfg: DistillConfig, teacher: Teacher, device: torch.device):
    """A fresh trainer of the teacher's architecture on the current layout."""
    if cfg.side == "human":
        hp = {**teacher.sd["ppo_config"], "num_envs": cfg.num_envs, "rollout": cfg.rollout}
        hp.update(bptt=cfg.bptt, minibatches=cfg.minibatches, lr=cfg.lr)
        algo: HumanPPO | ZombieMAPPO = HumanPPO(PPOConfig(**hp), device)
    else:
        zp = {**teacher.sd["mappo_config"], "num_envs": cfg.num_envs, "rollout": cfg.rollout}
        zp.update(bptt=cfg.bptt, minibatches=cfg.minibatches, lr=cfg.lr)
        algo = ZombieMAPPO(MAPPOConfig(**zp), device)
    algo.value_rms.load_state_dict(teacher.value_rms.state_dict())
    return algo


def opponents(cfg: DistillConfig, device: torch.device) -> list[tuple[str, Any]]:
    """`(name, factory)` of the opponents: the scripted bot, plus the `opponent` checkpoint."""
    if cfg.side == "human":
        out: list[tuple[str, Any]] = [
            ("scripted", lambda c, e, n, s: ScriptedZombies(c, e, n, seed=s))
        ]
        if cfg.opponent:
            za = load_zombie_actor(resolve(cfg.opponent), device)
            out.append(
                ("policy", lambda c, e, n, s: RecurrentZombiePolicy(za, e, n, device, config=c))
            )
    else:
        out = [("scripted", lambda c, e, s: ScriptedHuman(c, e, seed=s))]
        if cfg.opponent:
            ha = load_human_actor(resolve(cfg.opponent), device)
            out.append(("policy", lambda c, e, s: RecurrentHumanPolicy(ha, e, device, config=c)))
    return out


class Distiller:
    def __init__(self, cfg: DistillConfig, env_cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.teacher = Teacher(resolve(cfg.teacher), cfg.side, device)
        self.algo = student_for(cfg, self.teacher, device)
        self.opponents = opponents(cfg, device)
        counts = cfg.zombie_counts * len(self.opponents)
        factories = [f for _, f in self.opponents for _ in cfg.zombie_counts]
        env_cls = HumanTrainingEnv if cfg.side == "human" else ZombieTrainingEnv
        self.env = env_cls(env_cfg, counts, cfg.num_envs, factories, cfg.seed, cfg.num_threads)
        # Group configs, for the teacher's adapters (map size, cap and house depth).
        self.group_cfgs = [deep_merge(env_cfg, {"zombie_count": n}) for n in counts]
        # First agent row of each group on the flat zombie axis.
        slices = self.env.group_slices
        sizes = [(sl.stop - sl.start) * n for sl, n in zip(slices, counts, strict=True)]
        self.agent_offsets = np.cumsum([0, *sizes[:-1]]).tolist()
        self.rng = np.random.default_rng(cfg.seed + 17)
        self.driver = np.ones(cfg.num_envs, dtype=bool)  # teacher drives this env's episode

    def _t(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _per_group(self, x: np.ndarray, fn, per_agent: bool = False) -> np.ndarray:
        """Applies `fn(rows, group config)` group by group (rows are envs, or agents)."""
        parts = []
        for i, sl in enumerate(self.env.group_slices):
            if per_agent:
                n = self.env.groups[i][2]
                first = self.agent_offsets[i]
                sl = slice(first, first + (sl.stop - sl.start) * n)
            parts.append(fn(x[sl], self.group_cfgs[i]))
        return np.concatenate(parts)

    # --- rollout ------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, batch, h_s, h_t, start, beta: float, ep: dict[str, list]):
        cfg, side, teacher = self.cfg, self.cfg.side, self.teacher
        T, L = cfg.rollout, cfg.bptt
        human = side == "human"
        buf: dict[str, list[torch.Tensor]] = {}
        h0 = []

        def put(k, v):
            buf.setdefault(k, []).append(v)

        agent_env = None if human else self._t(self.env.agent_env, torch.long)
        for t in range(T):
            obs = self._t(batch.obs)
            state = self._t(batch.state)
            reset = (~start).unsqueeze(1)
            h_s, h_t = h_s * reset, h_t * reset
            if t % L == 0:
                h0.append(h_s.clone())
            t_obs = self._per_group(batch.obs, lambda o, c: teacher.obs(o, c, side), not human)
            t_obs = self._t(t_obs)
            t_state = self._t(self._per_group(batch.state, teacher.state))
            if human:
                mask = self._t(batch.mask, torch.bool)
                move_s, inter_s, h_s = self.algo.actor.step(obs, h_s, mask)
                move_t, inter_t, h_t = teacher.actor.step(t_obs, h_t, mask)
                drive = self._t(self.driver, torch.bool).unsqueeze(1)
                a_s = torch.stack([move_s.sample(), inter_s.sample()], dim=1)
                a_t = torch.stack([move_t.sample(), inter_t.sample()], dim=1)
                actions = torch.where(drive, a_t, a_s)
                put("mask", mask)
                put("t_move", move_t.logits)
                put("t_inter", inter_t.logits)
                put("t_value", teacher.critic(t_state))
            else:
                dist_s, h_s = self.algo.actor.step(obs, h_s)
                dist_t, h_t = teacher.actor.step(t_obs, h_t)
                drive = self._t(self.driver, torch.bool)[agent_env]
                actions = torch.where(drive, dist_t.sample(), dist_s.sample())
                put("t_move", dist_t.logits)
                put("t_value", teacher.critic(t_state[agent_env], t_obs))
            put("obs", obs)
            put("state", state)
            put("starts", start)

            step = self.env.step(actions.cpu().numpy())
            env_done = step.terminated | step.truncated
            for e in np.nonzero(env_done)[0]:
                ep["outcome"].append(int(step.outcome[e]))
                ep["teacher"].append(bool(self.driver[e]))
                ep["length"].append(float(step.episode_length[e]))
            self.driver[env_done] = self.rng.random(int(env_done.sum())) < beta
            done = env_done if human else env_done[self.env.agent_env]
            start = self._t(done, torch.bool)
            batch = step.next

        rollout = {k: torch.stack(v) for k, v in buf.items()}
        rollout["h0"] = torch.stack(h0)
        if agent_env is not None:
            rollout["agent_env"] = agent_env
        return rollout, batch, h_s, h_t, start

    # --- update -------------------------------------------------------------------------
    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg, algo = self.cfg, self.algo
        human = cfg.side == "human"
        L = cfg.bptt
        T, R = rollout["starts"].shape
        chunks = T // L
        n = chunks * R
        keys = ["obs", "starts", "t_move", "t_value"]
        keys += ["mask", "t_inter", "state"] if human else []
        data = {k: bptt_sequences(rollout[k], L) for k in keys}
        h0 = rollout["h0"].reshape(n, -1)
        if not human:
            state, agent_env = rollout["state"], rollout["agent_env"]
            seq = torch.arange(n, device=self.device)
            seq_t0, seq_env = (seq // R) * L, agent_env[seq % R]
            offsets = torch.arange(L, device=self.device)
        params = list(algo.actor.parameters()) + list(algo.critic.parameters())
        names = ("kl", "kl_move", "kl_interact", "value_mse")
        stats: dict[str, list[float]] = {k: [] for k in names}
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n, device=self.device).tensor_split(cfg.minibatches):
                obs = data["obs"][idx]
                if human:
                    starts, mask = data["starts"][idx], data["mask"][idx]
                    move, inter = algo.actor.sequence(obs, h0[idx], starts, mask)
                    kl_m = kl(data["t_move"][idx], move)
                    kl_i = kl(data["t_inter"][idx], inter)
                    v = algo.critic(data["state"][idx].flatten(0, 1))
                else:
                    move = algo.actor.sequence(obs, h0[idx], data["starts"][idx])
                    kl_m = kl(data["t_move"][idx], move)
                    kl_i = torch.zeros((), device=self.device)
                    rows = seq_t0[idx].unsqueeze(1) + offsets
                    mb_state = state[rows, seq_env[idx].unsqueeze(1)]
                    v = algo.critic(mb_state.flatten(0, 1), obs.flatten(0, 1))
                v_loss = (v - data["t_value"][idx].flatten()).pow(2).mean()
                loss = kl_m + kl_i + cfg.vf_coef * v_loss
                algo.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                algo.optimizer.step()
                stats["kl_move"].append(kl_m.item())
                stats["kl_interact"].append(kl_i.item())
                stats["kl"].append(kl_m.item() + kl_i.item())
                stats["value_mse"].append(v_loss.item())
        return {f"train/{k}": float(np.mean(v)) for k, v in stats.items()}

    # --- evaluation ---------------------------------------------------------------------
    def evaluate(self, env_cfg: dict[str, Any], episodes: int, prefix: str) -> dict[str, float]:
        """Outcome rates of the student and of the teacher against every opponent."""
        cfg, dev = self.cfg, self.device
        out: dict[str, float] = {}
        for who, actor in (("student", self.algo.actor), ("teacher", self.teacher.actor)):
            for name, opp in self.opponents:
                if cfg.side == "human":

                    def human(c, e, s, actor=actor):
                        return RecurrentHumanPolicy(actor, e, dev, config=c)

                    r = catch_rates(env_cfg, cfg.zombie_counts, human, opp, episodes, 4242, "x")
                else:

                    def zombies(c, e, n, s, actor=actor):
                        return RecurrentZombiePolicy(actor, e, n, dev, config=c)

                    r = catch_rates(env_cfg, cfg.zombie_counts, opp, zombies, episodes, 4242, "x")
                for key in ("caught", "starved", "survived"):
                    out[f"{prefix}_{who}_vs_{name}/{key}"] = r[f"x/{key}"]
        gaps = [
            abs(out[f"{prefix}_student_vs_{name}/{k}"] - out[f"{prefix}_teacher_vs_{name}/{k}"])
            for name, _ in self.opponents
            for k in ("caught", "starved", "survived")
        ]
        out[f"{prefix}/max_gap"] = max(gaps)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a checkpoint into the current layout.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = DistillConfig.load(
        resolve(str(args.config)),
        {"run_id": args.run_id, "total_steps": args.total_steps, "device": args.device},
    )
    run_id = cfg.run_id or time.strftime(f"{cfg.side}-distill-%Y%m%d-%H%M%S")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    env_cfg = load_config(resolve(cfg.env_config), env_overrides=False)
    dist = Distiller(cfg, env_cfg, device)
    log.info(
        "distill %s: teacher %s (layout v%d) -> student layout v%d, opponents %s, run %s",
        cfg.side,
        cfg.teacher,
        dist.teacher.version,
        OBS_VERSION,
        [n for n, _ in dist.opponents],
        run_id,
    )

    ckpt_dir = Path(os.environ.get("ZAI_CHECKPOINT_DIR", "checkpoints")) / cfg.side / run_id
    metrics = MetricsLogger(
        run_id,
        cfg.side,
        server_url=os.environ.get("ZAI_SERVER_URL"),
        tensorboard_dir=Path(os.environ.get("ZAI_RUNS_DIR", "runs")),
    )
    env = dist.env
    rows = env.num_envs if cfg.side == "human" else env.num_agents
    batch = env.reset()
    h_s = dist.algo.actor.initial_state(rows, device)
    h_t = dist.teacher.actor.initial_state(rows, device)
    start = torch.ones(rows, dtype=torch.bool, device=device)
    per_iter = cfg.rollout * env.num_envs
    decision_s = env_cfg["dt_s"] * env_cfg["action_repeat"]
    steps = iteration = 0
    started = time.time()
    while steps < cfg.total_steps:
        beta = max(0.0, 1.0 - steps / (cfg.beta_fraction * cfg.total_steps))
        t0 = time.time()
        ep: dict[str, list] = {"outcome": [], "teacher": [], "length": []}
        rollout, batch, h_s, h_t, start = dist.collect(batch, h_s, h_t, start, beta, ep)
        t1 = time.time()
        scalars = dist.update(rollout)
        t2 = time.time()
        iteration += 1
        steps += per_iter
        scalars.update(
            {
                "train/beta": beta,
                "perf/steps_per_s": per_iter / (t2 - t0),
                "perf/rollout_s": t1 - t0,
                "perf/update_s": t2 - t1,
            }
        )
        student_eps = [o for o, by_t in zip(ep["outcome"], ep["teacher"], strict=True) if not by_t]
        if ep["outcome"]:
            scalars["episode/length_s"] = float(np.mean(ep["length"])) * decision_s
        if student_eps:
            outcomes = np.array(student_eps)
            scalars["episode_student/count"] = len(student_eps)
            for code, name in OUTCOME_KEYS.items():
                scalars[f"episode_student/{name}"] = float((outcomes == code).mean())
        if iteration % cfg.eval_every == 0:
            scalars.update(dist.evaluate(env_cfg, cfg.eval_episodes, "eval"))
        metrics.log(steps, scalars)
        gap = scalars.get("eval/max_gap")
        log.info(
            "it %d steps %d | beta %.2f kl %.4f (move %.4f interact %.4f) value_mse %.4f%s"
            " | %.0f steps/s",
            iteration,
            steps,
            beta,
            scalars["train/kl"],
            scalars["train/kl_move"],
            scalars["train/kl_interact"],
            scalars["train/value_mse"],
            "" if gap is None else f" | eval max gap {100 * gap:.1f} pts",
            scalars["perf/steps_per_s"],
        )
        if iteration % cfg.checkpoint_every == 0 or steps >= cfg.total_steps:
            extra = {"iteration": iteration, "steps": steps, "run_id": run_id}
            extra.update(env_config=env_cfg, distilled_from=cfg.teacher)
            save_checkpoint(ckpt_dir / "latest.pt", dist.algo, extra)

    final = dist.evaluate(env_cfg, cfg.final_eval_episodes, "final")
    final["final/kl"] = scalars["train/kl"]
    final["final/accepted"] = float(
        final["final/max_gap"] <= cfg.accept_points and final["final/kl"] < cfg.accept_kl
    )
    metrics.log(steps, final)
    for k in sorted(final):
        log.info("%s = %.4f", k, final[k])
    extra = {"iteration": iteration, "steps": steps, "run_id": run_id, "env_config": env_cfg}
    extra.update(distilled_from=cfg.teacher, distill_eval=final)
    save_checkpoint(ckpt_dir / "final.pt", dist.algo, extra)
    metrics.close()
    log.info(
        "done: %d steps in %.0f s, %s",
        steps,
        time.time() - started,
        "accepted" if final["final/accepted"] else "NOT accepted",
    )


if __name__ == "__main__":
    main()
