"""Background jobs for long simulations and sweeps (thread pool + polling)."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from linkstudio.api.editor import build_graph
from linkstudio.api.probes import compute_probes, scope_probes
from linkstudio.core import Simulation, StopCriteria, Sweep
from linkstudio.core.parallel import cpu_count
from linkstudio.phy import theory


def _clean(value: Any) -> Any:
    """JSON-safe numbers (NaN/inf -> None, numpy scalars -> Python)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class Job:
    id: str
    mode: str
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    detail: str | None = None  # traceback of a failed job
    started: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        return _clean({"id": self.id, "mode": self.mode, "status": self.status, "progress": self.progress,
                       "message": self.message, "records": self.records, "result": self.result,
                       "error": self.error, "elapsed_s": round(time.time() - self.started, 2)})


THEORY_CURVES = {
    "bpsk": ("BPSK/QPSK AWGN", theory.ber_bpsk),
    "psk8": ("8-PSK AWGN", lambda x: theory.ber_psk(8, x)),
    "qam16": ("16-QAM AWGN", lambda x: theory.ber_qam(16, x)),
    "qam64": ("64-QAM AWGN", lambda x: theory.ber_qam(64, x)),
    "qam256": ("256-QAM AWGN", lambda x: theory.ber_qam(256, x)),
    "rayleigh": ("BPSK Rayleigh", theory.ber_bpsk_rayleigh),
    "mrc2": ("BPSK MRC L=2", lambda x: theory.ber_bpsk_mrc(2, x)),
    "mrc4": ("BPSK MRC L=4", lambda x: theory.ber_bpsk_mrc(4, x)),
}


class JobManager:
    def __init__(self, base_dir: Path, workers: int = 2) -> None:
        self.base_dir = base_dir
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, request: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], mode=request.get("mode", "simulation"))
        self.jobs[job.id] = job
        self.pool.submit(self._run, job, request)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job:
            job.cancel_event.set()

    # ------------------------------------------------------------------ runners
    def _run(self, job: Job, request: dict[str, Any]) -> None:
        job.status = "running"
        try:
            editor = request["graph"]
            graph = build_graph(editor, self.base_dir)
            graph.validate()
            if job.mode == "sweep":
                self._run_sweep(job, graph, editor, request.get("config") or {})
            else:
                self._run_simulation(job, graph, editor, request.get("config") or {})
            job.status = "cancelled" if job.cancel_event.is_set() else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
        except Exception as exc:
            job.status = "error"
            job.error = f"{exc}"
            job.message = "failed"
            job.detail = traceback.format_exc(limit=3)

    @staticmethod
    def _stop(cfg: dict[str, Any]) -> tuple[int | None, StopCriteria | None]:
        if cfg.get("metric"):
            return None, StopCriteria(max_batches=int(cfg.get("max_batches", 100)), metric=cfg["metric"],
                                      threshold=float(cfg.get("threshold", 100)),
                                      max_time_s=cfg.get("max_time_s"))
        return int(cfg.get("n_batches", 10)), None

    @staticmethod
    def _workers(cfg: dict[str, Any]) -> int:
        return max(1, min(int(cfg.get("workers", 1)), cpu_count()))

    def _run_simulation(self, job: Job, graph, editor: dict[str, Any], cfg: dict[str, Any]) -> None:
        n_batches, stop = self._stop(cfg)
        total = n_batches or (stop.max_batches if stop else 1)
        probes = scope_probes(editor)
        sim = Simulation(graph, batch_size=int(editor.get("batch_size", 32)), seed=int(editor.get("seed", 0)))

        def on_batch(n: int, metrics: dict[str, float]) -> bool:
            job.progress = min(n / total, 0.99)
            job.message = f"batch {n}"
            job.result = {"metrics": metrics, "n_batches": n}
            return not job.cancel_event.is_set()

        keys = [p["key"] for p in probes]
        res = sim.run(n_batches=n_batches, stop=stop, capture="*" if self._workers(cfg) == 1 else keys,
                      on_batch=on_batch, workers=self._workers(cfg))
        job.result = {"metrics": res.metrics, "n_batches": res.n_batches, "elapsed_s": res.elapsed_s,
                      "stop_reason": res.stop_reason, "probes": compute_probes(res.captures, probes)}

    def _run_sweep(self, job: Job, graph, editor: dict[str, Any], cfg: dict[str, Any]) -> None:
        params = cfg.get("params") or {}
        if not params:
            raise ValueError("The sweep needs at least one parameter.")
        n_batches, stop = self._stop(cfg)
        sweep = Sweep(graph, params, mode=cfg.get("mode", "grid"), batch_size=int(editor.get("batch_size", 32)),
                      seed=int(editor.get("seed", 0)), n_batches=n_batches, stop=stop)
        total = len(sweep.points())

        def on_point(i: int, n: int, record: dict[str, Any]) -> None:
            job.records.append(record)
            job.progress = (i + 1) / n
            job.message = f"point {i + 1}/{n}"

        job.message = f"0/{total} points"
        sweep.run(on_point=on_point, should_stop=job.cancel_event.is_set, workers=self._workers(cfg))
        result: dict[str, Any] = {"param_names": list(params)}
        key = cfg.get("theory")
        if key in THEORY_CURVES and len(params) == 1:
            values = np.asarray(list(params.values())[0], dtype=float)
            fine = np.linspace(values.min(), values.max(), 80)
            label, fn = THEORY_CURVES[key]
            result["theory"] = {"label": label, "x": fine.tolist(), "y": np.asarray(fn(fine), dtype=float).tolist()}
        job.result = result
