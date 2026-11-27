"""Frozen opponents of one side, batched per resident slot (docs/ARCHITECTURE.md 4.8.4).

The opponent side of a league group (zombies in B, the human in C) has `K` resident slots:
slot 0 is the scripted bot, slots 1.. hold pool entries as frozen actors on the device. Each
opponent env is assigned a slot when its episode starts (the scripted slot with probability
`scripted_share`, otherwise a resident entry with probability proportional to its PFSP
weight), so an opponent never changes within an episode. Every `lease_iters` iterations the
oldest resident entry starts draining: it gets no new episodes, and once its last one ends it
is replaced by a new PFSP sample. Per step there is one forward pass per resident actor over
its rows, plus the scripted bots of the side's groups.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..bots import ScriptedHuman, ScriptedZombies
from ..envs import MatchObs, MatchTrainingEnv
from ..models import graphs
from ..policy import load_human_actor, load_zombie_actor
from .pool import Entry, Pool, other


@dataclass
class Slot:
    entry: Entry | None = None
    actor: Any = None
    draining: bool = False
    loaded_iter: int = 0
    envs: set[int] = field(default_factory=set)  # opponent envs currently assigned


class OpponentSlots:
    def __init__(
        self,
        side: str,
        env: MatchTrainingEnv,
        pool: Pool,
        device: torch.device,
        rng: np.random.Generator,
        slots: int = 4,
        scripted_share: float = 0.2,
        lease_iters: int = 5,
        seed: int = 0,
        graph: bool = True,
    ):
        """`graph`: on CUDA, each resident actor runs as a CUDA graph (`models.graphs`) over all
        of this side's rows, and its own rows are picked from the result."""
        self.graph = graph and graphs.usable(device)
        self.templates: dict[int, _Template] = {}
        self.graphs: dict[tuple[int, int], graphs.GraphedActor] = {}  # (slot, batch) -> graph
        self.side, self.learner = side, other(side)
        self.env, self.pool, self.device, self.rng = env, pool, device, rng
        self.scripted_share, self.lease_iters = scripted_share, lease_iters
        kind = "B" if side == "zombie" else "C"
        self.groups = [i for i, g in enumerate(env.specs) if g.kind == kind]
        self.envs = np.nonzero(env.kind == kind)[0]  # global env indices
        if side == "human":
            self.rows = self.envs  # one row per env
            self.row_env = self.envs
        else:
            self.rows = env.opp_zombie_agents  # one row per zombie
            self.row_env = env.agent_env[self.rows]
        self.slots = [Slot(entry=pool.scripted(side))] + [Slot() for _ in range(slots - 1)]
        self.bots = {}
        for i in self.groups:
            g, cfg = env.specs[i], env.group_cfgs[i]
            if side == "human":
                self.bots[i] = ScriptedHuman(cfg, g.num_envs, seed=seed + 31 * i)
            else:
                self.bots[i] = ScriptedZombies(cfg, g.num_envs, g.zombie_count, seed=seed + 31 * i)
        # Positions, among this side's rows, of each scripted bot's group rows.
        row_pos = {int(r): i for i, r in enumerate(self.rows)}
        self.bot_pos = {}
        for i in self.groups:
            span = env.env_slices[i] if side == "human" else env.agent_slices[i]
            self.bot_pos[i] = np.array([row_pos[r] for r in range(span.start, span.stop)])
        self.hidden_size: int | None = None
        self.h: torch.Tensor | None = None
        self.env_slot = np.full(env.num_envs, -1)
        self.env_entry: dict[int, str] = {}
        for k in range(1, len(self.slots)):
            self._load(k, pool.sample(self.learner, rng, set(self.resident())), 0)
        for e in self.envs:
            self._assign(int(e))

    # --- slots --------------------------------------------------------------------------
    def _load(self, k: int, entry: Entry | None, iteration: int) -> None:
        slot = self.slots[k]
        slot.entry, slot.actor, slot.draining, slot.loaded_iter = entry, None, False, iteration
        if entry is None:
            return
        load = load_human_actor if self.side == "human" else load_zombie_actor
        slot.actor = load(entry.path, self.device)
        if self.hidden_size is None:
            self.hidden_size = slot.actor.hidden_size
            self.h = torch.zeros(len(self.rows), self.hidden_size, device=self.device)
        elif slot.actor.hidden_size != self.hidden_size:
            raise ValueError(f"pool entry {entry.id} has another hidden size")

    def _assign(self, e: int) -> None:
        """Assigns a slot to opponent env `e` for its next episode."""
        weights = self.pool.weights(self.learner)
        cands = [
            k for k, s in enumerate(self.slots) if k > 0 and s.entry is not None and not s.draining
        ]
        k = 0
        if cands and self.rng.random() >= self.scripted_share:
            w = np.array([weights.get(self.slots[c].entry.id, 0.0) for c in cands])
            w = w / w.sum() if w.sum() > 0 else np.full(len(cands), 1.0 / len(cands))
            k = cands[int(self.rng.choice(len(cands), p=w))]
        self.env_slot[e] = k
        self.env_entry[e] = self.slots[k].entry.id
        self.slots[k].envs.add(e)

    def episodes_ended(self, done: np.ndarray, outcome: np.ndarray, score) -> None:
        """Records finished opponent episodes in the pool (`score(outcome) -> learner score`),
        clears their memory and assigns slots for the next episodes."""
        ended = self.envs[done[self.envs]]
        if len(ended) == 0:
            return
        for e in ended:
            e = int(e)
            self.pool.record(self.learner, self.env_entry[e], score(int(outcome[e])))
            self.slots[self.env_slot[e]].envs.discard(e)
            self._assign(e)
        if self.h is not None:
            rows = np.isin(self.row_env, ended)
            self.h[torch.as_tensor(rows, device=self.device)] = 0.0

    def end_iteration(self, iteration: int) -> None:
        """Lease rotation: start draining the oldest resident entry every `lease_iters`
        iterations, and reload drained slots with a new PFSP sample."""
        resident = [k for k, s in enumerate(self.slots) if k > 0 and s.entry is not None]
        if iteration % self.lease_iters == 0 and resident:
            if not any(self.slots[k].draining for k in resident):
                oldest = min(resident, key=lambda k: self.slots[k].loaded_iter)
                self.slots[oldest].draining = True
        for k, s in enumerate(self.slots):
            if k > 0 and (s.entry is None or s.draining) and not s.envs:
                avoid = {e for j, e in enumerate(self._ids()) if j != k}
                self._load(k, self.pool.sample(self.learner, self.rng, avoid), iteration)

    def _ids(self) -> list[str | None]:
        return [s.entry.id if s.entry is not None else None for s in self.slots]

    def resident(self) -> list[str]:
        return [s.entry.id for s in self.slots if s.entry is not None]

    # --- acting -------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: MatchObs) -> np.ndarray:
        """Actions of every opponent row: `[envs, 2]` for humans, `[agents]` for zombies.

        The resident actors' forward passes are queued on the device first and their actions
        come back in one transfer after the scripted bots ran on the CPU, so the two overlap.
        """
        env = self.env
        human = self.side == "human"
        out = np.zeros((len(self.rows), 2) if human else len(self.rows), dtype=np.int64)
        slot_of_row = self.env_slot[self.row_env]
        sampled: list[tuple[np.ndarray, torch.Tensor]] = []
        for k, s in enumerate(self.slots):
            if k == 0 or s.actor is None:
                continue
            sel = np.nonzero(slot_of_row == k)[0]
            if len(sel) == 0:
                continue
            sel_t = torch.as_tensor(sel, device=self.device)
            if self.graph:
                sampled.append((sel, self._act_graphed(k, s.actor, obs, sel, sel_t)))
                continue
            rows = self.rows[sel]
            h = self.h[sel_t]
            if human:
                o = torch.as_tensor(obs.human_obs[rows], device=self.device)
                m = torch.as_tensor(obs.human_mask[rows], device=self.device)
                move, inter, h = s.actor.step(o, h, m)
                a = torch.stack([move.sample(), inter.sample()], 1)
            else:
                o = torch.as_tensor(obs.zombie_obs[rows], device=self.device)
                dist, h = s.actor.step(o, h)
                a = dist.sample()
            self.h[sel_t] = h
            sampled.append((sel, a))
        # Scripted bots act on their whole group; only the rows in slot 0 are used.
        for i, bot in self.bots.items():
            es, as_ = env.env_slices[i], env.agent_slices[i]
            if human:
                a = bot.act(obs.human_obs[es], obs.human_mask[es])
            else:
                g = env.specs[i]
                zo = obs.zombie_obs[as_].reshape(g.num_envs, g.zombie_count, -1)
                a = bot.act(zo).reshape(-1)
            pos = self.bot_pos[i]
            use = slot_of_row[pos] == 0
            out[pos[use]] = a[use]
        if sampled:
            actions = torch.cat([a for _, a in sampled]).cpu().numpy()
            out[np.concatenate([sel for sel, _ in sampled])] = actions
        return out

    def _act_graphed(
        self, k: int, actor: Any, obs: MatchObs, sel: np.ndarray, sel_t: torch.Tensor
    ) -> torch.Tensor:
        """Actions of slot `k`'s rows `sel` (positions among this side's rows) from a graph of
        its actor; updates their memory. The batch is padded (repeating the first row) up to a
        multiple of `GRAPH_BUCKET`, with one graph per slot and padded size, all reading the
        slot's template actor, into which a new resident's weights are copied."""
        human = self.side == "human"
        tmpl = self.templates.get(k)
        if tmpl is None or tmpl.src is not actor:
            if tmpl is not None and _compatible(tmpl.actor, actor):
                tmpl.actor.load_state_dict(actor.state_dict())
                tmpl.src = actor
            else:  # first resident of the slot, or another architecture: capture anew
                self.templates[k] = _Template(actor, actor)
                for key in [key for key in self.graphs if key[0] == k]:
                    del self.graphs[key]
            tmpl = self.templates[k]
        n = len(sel)
        b = min(-(-n // GRAPH_BUCKET) * GRAPH_BUCKET, len(self.rows))
        pad = np.concatenate([sel, np.full(b - n, sel[0])])
        rows = self.rows[pad]
        o_np = obs.human_obs[rows] if human else obs.zombie_obs[rows]
        o = torch.as_tensor(o_np, device=self.device)
        m = torch.as_tensor(obs.human_mask[rows], device=self.device) if human else None
        h = self.h[torch.as_tensor(pad, device=self.device)]
        g = self.graphs.get((k, b))
        if g is None:
            g = self.graphs[(k, b)] = graphs.GraphedActor(tmpl.actor, o, h, m)
        out = g.run(o, h, m)
        self.h[sel_t] = out[-1][:n]
        if human:
            move, inter = graphs.dist(out[0][:n]), graphs.dist(out[1][:n])
            return torch.stack([move.sample(), inter.sample()], 1)
        return graphs.dist(out[0][:n]).sample()


GRAPH_BUCKET = 128  # rows; graph batch sizes are multiples of this


@dataclass
class _Template:
    actor: Any  # the module the slot's graphs captured
    src: Any  # the resident actor whose weights it holds


def _compatible(a: Any, b: Any) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return (
        type(a) is type(b)
        and getattr(a, "obs_version", None) == getattr(b, "obs_version", None)
        and sa.keys() == sb.keys()
        and all(sa[key].shape == sb[key].shape for key in sa)
    )
