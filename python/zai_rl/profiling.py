"""Per-phase wall-clock timers for the trainers (M6.0, docs/ARCHITECTURE.md section 4.8.3).

`timer` is a process-wide `PhaseTimer`: code anywhere in the training loop (envs, algorithms,
trainers) wraps a phase in `with timer("sim"):`, and the trainer reads and clears the totals
once per iteration with `timer.take()`, which returns `perf/<phase>_s` scalars.

GPU work is asynchronous, so without synchronization a phase that launches kernels is charged
only for the launch, and the wait lands in whichever phase next copies to the host. Profiling
runs set `profile_sync: true` (trainer config), which calls `torch.cuda.synchronize()` at the
end of every phase; that makes the split exact and costs some throughput, so it is off by
default. The phase names are also `torch.profiler` ranges, so a trace taken with
`profile_iteration` shows the same split.
"""

import contextlib
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import torch


class PhaseTimer:
    def __init__(self) -> None:
        self.sync = False
        self.totals: dict[str, float] = defaultdict(float)

    def configure(self, sync: bool) -> None:
        self.sync = sync and torch.cuda.is_available()

    @contextlib.contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(name):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if self.sync:
                    torch.cuda.synchronize()
                self.totals[name] += time.perf_counter() - t0

    def take(self) -> dict[str, float]:
        """`perf/<phase>_s` totals since the last call, then clears them; on CUDA also
        `perf/gpu_peak_gb`, the peak memory allocated by PyTorch since the last call."""
        out = {f"perf/{k}_s": v for k, v in self.totals.items()}
        self.totals.clear()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            out["perf/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
        return out


timer = PhaseTimer()


def rollout_split(scalars: dict[str, float]) -> None:
    """Adds `perf/glue_s` (rollout time outside the simulation, the opponent and the learner's
    forward passes) and `perf/update_step_s` (update time outside the state gather)."""
    if "perf/rollout_s" in scalars:
        inner = sum(scalars.get(f"perf/{k}_s", 0.0) for k in ("sim", "opponent", "policy"))
        scalars["perf/glue_s"] = max(0.0, scalars["perf/rollout_s"] - inner)
    if "perf/update_s" in scalars:
        scalars["perf/update_step_s"] = max(
            0.0, scalars["perf/update_s"] - scalars.get("perf/update_gather_s", 0.0)
        )


def maybe_trace(
    iteration: int, profile_iteration: int, out_dir: Path | None
) -> contextlib.AbstractContextManager:
    """A `torch.profiler` context for iteration `profile_iteration` (1-based; 0 disables),
    writing a Chrome trace and a summary table to `out_dir`; a no-op context otherwise."""
    if not profile_iteration or iteration != profile_iteration or out_dir is None:
        return contextlib.nullcontext()
    return _trace(out_dir)


@contextlib.contextmanager
def _trace(out_dir: Path) -> Iterator[None]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=activities) as prof:
        yield
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    sort = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    (out_dir / "summary.txt").write_text(prof.key_averages().table(sort_by=sort, row_limit=60))
