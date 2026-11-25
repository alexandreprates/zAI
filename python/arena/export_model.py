"""Export trained actors to ONNX for live matches in the game server.

Each model is a pair of files in `<ZAI_DATA_DIR>/models/`:

- `<id>.onnx`: one decision of the actor, `(obs [B, D], h [B, H]) -> (move_logits [B, 17],
  [interact_logits [B, 3],] h_new [B, H])`. Masking and sampling happen in the server, so
  the graph has no randomness; the GRU is written out as an explicit cell (plain MatMul /
  Sigmoid / Tanh) to stay within the operator set `tract` runs.
- `<id>.json`: metadata (`side`, `run_id`, `iteration`, `steps`, `stage`, `hidden`,
  `obs_dim`, `config`). `config` is the env config the model was trained on; the viewer uses
  it as the map for matches with the model.

Usage (trainer container): `python -m arena.export_model --checkpoint
/checkpoints/zombie/<run_id>/latest.pt`. Trainers also export `<run_id>` on every checkpoint
when `ZAI_DATA_DIR` is set.
"""

import argparse
import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import zai_sim
from torch import nn

from zai_env.config import config_dir, load_config
from zai_rl.models import HumanActor, ZombieActor

log = logging.getLogger("export_model")

OPSET = 17


def gru_cell(gru: nn.GRU, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One step of a single-layer `nn.GRU` with explicit ops (same math as PyTorch)."""
    gi = x @ gru.weight_ih_l0.T + gru.bias_ih_l0
    gh = h @ gru.weight_hh_l0.T + gru.bias_hh_l0
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    # Same as `(1 - z) * n + z * h`; tract 0.21 negates `1 - z` (a constant on the left of
    # Sub), so the export avoids that form.
    return n + z * (h - n)


class HumanStep(nn.Module):
    def __init__(self, actor: HumanActor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        h_new = gru_cell(self.actor.gru, self.actor.encode(obs), h)
        return self.actor.move_head(h_new), self.actor.interact_head(h_new), h_new


class ZombieStep(nn.Module):
    def __init__(self, actor: ZombieActor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        h_new = gru_cell(self.actor.gru, self.actor.encode(obs), h)
        return self.actor.move_head(h_new), h_new


def step_module(actor: HumanActor | ZombieActor) -> tuple[str, nn.Module, int, list[str]]:
    """`(side, module, obs_dim, output names)` for an actor."""
    if isinstance(actor, HumanActor):
        return "human", HumanStep(actor), zai_sim.HUMAN_OBS_DIM, ["move", "interact", "h_out"]
    return "zombie", ZombieStep(actor), zai_sim.ZOMBIE_OBS_DIM, ["move", "h_out"]


def model_id(name: str) -> str:
    """Model ids are file names and URL path parts: `[A-Za-z0-9_-]`, at most 64 chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)[:64]


def models_dir() -> Path | None:
    """`<ZAI_DATA_DIR>/models`, only when `ZAI_DATA_DIR` is set."""
    data = os.environ.get("ZAI_DATA_DIR")
    return Path(data) / "models" if data else None


def stage_name(config: dict[str, Any]) -> str:
    """Name of the env config file in `configs/env` with the same map and rules as `config`
    (`custom` if none). Rewards are ignored (they do not change a match), so is the zombie
    count (trainers mix several, and matches pick their own), and keys added to the config
    after the model was trained take their defaults."""

    def rules(c: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in c.items() if k not in ("reward", "zombie_count")}

    try:
        target = rules(load_config(overrides=config, env_overrides=False))
    except ValueError:
        return "custom"
    for path in sorted(config_dir().glob("*.yaml")):
        try:
            if rules(load_config(path, env_overrides=False)) == target:
                return path.stem
        except ValueError:
            continue
    return "custom"


def export_actor(
    actor: HumanActor | ZombieActor,
    out_dir: Path,
    name: str,
    meta: dict[str, Any],
) -> Path:
    """Writes `<id>.onnx` and `<id>.json`; returns the ONNX path. `meta` must hold `config`.
    The actor is copied to the CPU, so a training actor on the GPU is left untouched."""
    side, module, obs_dim, outputs = step_module(copy.deepcopy(actor).cpu().eval())
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = model_id(name)
    onnx_path = out_dir / f"{mid}.onnx"
    tmp = out_dir / f"{mid}.onnx.tmp"
    obs = torch.zeros(2, obs_dim)
    h = torch.zeros(2, actor.hidden_size)
    batch = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            module,
            (obs, h),
            str(tmp),
            input_names=["obs", "h"],
            output_names=outputs,
            dynamic_axes={k: batch for k in ["obs", "h", *outputs]},
            opset_version=OPSET,
            dynamo=False,
        )
    tmp.replace(onnx_path)
    info = {
        "id": mid,
        "side": side,
        "hidden": actor.hidden_size,
        "obs_dim": obs_dim,
        "created_at": int(time.time()),
        "stage": stage_name(meta["config"]),
        **meta,
    }
    json_path = out_dir / f"{mid}.json"
    json_tmp = out_dir / f"{mid}.json.tmp"
    json_tmp.write_text(json.dumps(info))
    json_tmp.replace(json_path)
    return onnx_path


def actor_from_checkpoint(sd: dict[str, Any]) -> HumanActor | ZombieActor:
    if "mappo_config" in sd:
        actor: HumanActor | ZombieActor = ZombieActor(hidden=sd["mappo_config"]["hidden"])
    else:
        actor = HumanActor(hidden=sd["ppo_config"]["hidden"])
    actor.load_state_dict(sd["actor"])
    return actor.eval()


def export_checkpoint(path: Path, out_dir: Path, name: str | None = None) -> Path:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    run_id = sd.get("run_id") or path.parent.name
    meta = {
        "run_id": run_id,
        "iteration": int(sd.get("iteration", 0)),
        "steps": int(sd.get("steps", 0)),
        "config": sd["env_config"],
    }
    return export_actor(actor_from_checkpoint(sd), out_dir, name or run_id, meta)


def export_training_model(
    actor: HumanActor | ZombieActor, run_id: str, config: dict[str, Any], iteration: int, steps: int
) -> None:
    """Called by the trainers on every checkpoint; failures are logged, never raised."""
    out_dir = models_dir()
    if out_dir is None:
        return
    meta = {"run_id": run_id, "iteration": iteration, "steps": steps, "config": config}
    try:
        export_actor(actor, out_dir, run_id, meta)
    except Exception:  # exporting must never stop a training run
        log.exception("model export failed for %s", run_id)


def write_parity_fixtures(out_dir: Path, seed: int = 0) -> None:
    """Small random actors exported next to their PyTorch outputs for two chained decisions
    (`<side>.onnx` + `<side>.parity.json`), checked by the server's parity test."""
    torch.manual_seed(seed)
    config = load_config(config_dir() / "s0.yaml", env_overrides=False)
    for actor in (HumanActor(hidden=16, entity_hidden=8), ZombieActor(hidden=16, entity_hidden=8)):
        side, module, obs_dim, outputs = step_module(actor.eval())
        meta = {"run_id": f"parity-{side}", "iteration": 0, "steps": 0, "config": config}
        export_actor(actor, out_dir, side, meta)
        obs = [torch.rand(3, obs_dim) * 2 - 0.5 for _ in range(2)]
        h = torch.zeros(3, actor.hidden_size)
        steps = []
        with torch.no_grad():
            for o in obs:
                res = module(o, h)
                h = res[-1]
                steps.append(
                    {"obs": o.flatten().tolist()}
                    | {k: v.flatten().tolist() for k, v in zip(outputs, res, strict=True)}
                )
        (out_dir / f"{side}.parity.json").write_text(json.dumps({"batch": 3, "steps": steps}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoints as ONNX server models.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--parity-fixtures", type=Path, help="write the server's parity test fixtures here"
    )
    parser.add_argument("--name", help="model id (default: the run id); one checkpoint only")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $ZAI_DATA_DIR/models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.parity_fixtures:
        write_parity_fixtures(args.parity_fixtures)
        log.info("wrote parity fixtures to %s", args.parity_fixtures)
    if args.name and len(args.checkpoint) > 1:
        parser.error("--name needs a single checkpoint")
    out_dir = args.out_dir or models_dir() or Path("/data/models")
    for ckpt in args.checkpoint:
        log.info("exported %s", export_checkpoint(ckpt, out_dir, args.name))


if __name__ == "__main__":
    main()
