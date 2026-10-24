"""Parameter sweeps: run a simulation for every point of a parameter grid."""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from linkstudio.core.graph import Graph
from linkstudio.core.simulation import Simulation, StopCriteria, _fmt


class SweepResult:
    """Table of sweep results: one record (dict) per point."""

    def __init__(self, records: list[dict[str, Any]], param_names: list[str]) -> None:
        self.records = records
        self.param_names = param_names

    @property
    def columns(self) -> list[str]:
        return list(self.records[0]) if self.records else []

    def column(self, name: str) -> np.ndarray:
        return np.array([r[name] for r in self.records])

    def select(self, **fixed: Any) -> SweepResult:
        """Records whose values match ``fixed``, e.g. ``select(**{"map.order": 16})``."""
        recs = [r for r in self.records if all(r.get(k) == v for k, v in fixed.items())]
        return SweepResult(recs, self.param_names)

    def table(self, columns: Sequence[str] | None = None) -> str:
        cols = list(columns) if columns else self.columns
        rows = [[_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols] for r in self.records]
        widths = [max(len(c), *(len(row[i]) for row in rows)) if rows else len(c) for i, c in enumerate(cols)]
        sep = "  "
        lines = [sep.join(c.rjust(w) for c, w in zip(cols, widths)),
                 sep.join("-" * w for w in widths)]
        lines += [sep.join(v.rjust(w) for v, w in zip(row, widths)) for row in rows]
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)


class Sweep:
    """Run ``graph`` for every combination of parameter values.

    Args:
        graph: Graph to simulate.
        params: Mapping ``"block.param" -> values``.
        mode: ``"grid"`` (cartesian product) or ``"zip"`` (values paired element-wise).
        batch_size, seed: Forwarded to :class:`Simulation`.
        n_batches, stop: Per-point stopping rule (see :meth:`Simulation.run`).
        same_seed: Re-use the same seed at every point (common random numbers:
            smoother curves and fair comparisons).
        metrics: Metric keys to keep (default: all).
    """

    def __init__(self, graph: Graph, params: Mapping[str, Sequence[Any]], *, mode: str = "grid",
                 batch_size: int = 32, seed: int = 0, n_batches: int | None = None,
                 stop: StopCriteria | None = None, same_seed: bool = True,
                 metrics: Sequence[str] | None = None) -> None:
        if mode not in ("grid", "zip"):
            raise ValueError("mode must be 'grid' or 'zip'")
        self.graph = graph
        self.params = {k: list(v) for k, v in params.items()}
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed
        self.n_batches = n_batches
        self.stop = stop
        self.same_seed = same_seed
        self.metrics = list(metrics) if metrics else None

    def points(self) -> list[dict[str, Any]]:
        names = list(self.params)
        if self.mode == "grid":
            combos = itertools.product(*self.params.values())
        else:
            lengths = {len(v) for v in self.params.values()}
            if len(lengths) > 1:
                raise ValueError("All value lists must have the same length in 'zip' mode.")
            combos = zip(*self.params.values())
        return [dict(zip(names, c)) for c in combos]

    def run(self, progress: bool = False,
            on_point: Callable[[int, int, dict[str, Any]], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> SweepResult:
        """Run every point.

        Args:
            progress: Print a line per point.
            on_point: Called with ``(index, total, record)`` after every point.
            should_stop: Polled after every batch; returning ``True`` cancels the sweep
                (records collected so far are returned).
        """
        originals = {k: self.graph.get_param(k) for k in self.params}
        records: list[dict[str, Any]] = []
        points = self.points()
        keep_going = (lambda n, m: not should_stop()) if should_stop else None
        try:
            for i, point in enumerate(points):
                if should_stop is not None and should_stop():
                    break
                for key, value in point.items():
                    self.graph.set_param(key, value)
                sim = Simulation(self.graph, self.batch_size, self.seed if self.same_seed else self.seed + i)
                res = sim.run(n_batches=self.n_batches, stop=self.stop, on_batch=keep_going)
                if res.stop_reason == "cancelled":
                    break
                metrics = res.metrics if self.metrics is None else {k: res.metrics[k] for k in self.metrics}
                record = {**point, **metrics, "n_batches": res.n_batches, "elapsed_s": round(res.elapsed_s, 3)}
                records.append(record)
                if on_point is not None:
                    on_point(i, len(points), record)
                if progress:
                    ptxt = ", ".join(f"{k}={v}" for k, v in point.items())
                    mtxt = ", ".join(f"{k}={_fmt(v)}" for k, v in list(metrics.items())[:3])
                    print(f"  [{i + 1}] {ptxt} -> {mtxt}")
        finally:
            for key, value in originals.items():
                self.graph.set_param(key, value)
        return SweepResult(records, list(self.params))
