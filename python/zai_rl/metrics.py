"""Metrics sinks: stdout, the zai-server metrics API (optional) and TensorBoard (optional)."""

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(
        self,
        run_id: str,
        model: str,
        server_url: str | None = None,
        tensorboard_dir: Path | None = None,
    ):
        self.run_id = run_id
        self.model = model
        self.server_url = server_url.rstrip("/") if server_url else None
        self._server_failed = False
        self._tb = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(tensorboard_dir / run_id / model))
            except ImportError:
                log.warning("tensorboard is not installed; TensorBoard logging disabled")

    def log(self, step: int, scalars: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in scalars.items() if v == v}  # drop NaN
        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, step)
            self._tb.flush()
        if self.server_url and clean:
            self._post(step, clean)

    def _post(self, step: int, scalars: dict[str, float]) -> None:
        body = json.dumps(
            {
                "run_id": self.run_id,
                "model": self.model,
                "step": step,
                "wall_time": time.time(),
                "scalars": scalars,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.server_url}/api/metrics",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2):
                pass
            if self._server_failed:
                log.info("metrics server reachable again")
                self._server_failed = False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if not self._server_failed:
                log.warning("cannot post metrics to %s: %s", self.server_url, e)
                self._server_failed = True

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
