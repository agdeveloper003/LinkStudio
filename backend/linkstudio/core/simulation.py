"""Monte-Carlo driver: runs batches through a graph until a stop criterion is met."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from linkstudio.core.context import RunContext
from linkstudio.core.graph import Graph, GraphError
from linkstudio.core.signal import Signal


@dataclass
class StopCriteria:
    """When to stop a simulation.

    The run stops at the first satisfied condition once ``min_batches`` were run:
    ``metric >= threshold`` (e.g. 100 bit errors), ``max_batches`` or ``max_time_s``.
    """

    max_batches: int = 100
    min_batches: int = 1
    max_time_s: float | None = None
    metric: str | None = None
    threshold: float | None = None

    @classmethod
    def errors(cls, metric: str = "ber.bit_errors", count: int = 100, max_batches: int = 1000,
               max_time_s: float | None = None) -> StopCriteria:
        """Stop after ``count`` errors have been observed (classic BER rule)."""
        return cls(max_batches=max_batches, metric=metric, threshold=count, max_time_s=max_time_s)

    def check(self, n_batches: int, elapsed_s: float, metrics: dict[str, float]) -> str | None:
        if n_batches < self.min_batches:
            return None
        if self.metric is not None and self.threshold is not None:
            if self.metric not in metrics:
                raise GraphError(f"Stop metric '{self.metric}' not found. Available: {sorted(metrics)}")
            if metrics[self.metric] >= self.threshold:
                return f"{self.metric} >= {self.threshold:g}"
        if n_batches >= self.max_batches:
            return "max_batches"
        if self.max_time_s is not None and elapsed_s >= self.max_time_s:
            return "max_time"
        return None


@dataclass
class SimulationResult:
    metrics: dict[str, float]
    n_batches: int
    elapsed_s: float
    stop_reason: str
    captures: dict[str, Signal] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def summary(self) -> str:
        lines = [f"{self.n_batches} batches in {self.elapsed_s:.2f} s (stop: {self.stop_reason})"]
        width = max((len(k) for k in self.metrics), default=0)
        for key, value in self.metrics.items():
            lines.append(f"  {key:<{width}}  {_fmt(value)}")
        return "\n".join(lines)


def _fmt(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or not value.is_integer()):
        return f"{value:.4e}" if value != 0 and abs(value) < 1e-2 else f"{value:.4f}"
    return f"{int(value)}"


class Simulation:
    """Runs a graph batch after batch.

    Args:
        graph: Graph to simulate.
        batch_size: Frames simulated in parallel per batch.
        seed: Master seed; identical seeds give bit-identical runs.
    """

    def __init__(self, graph: Graph, batch_size: int = 32, seed: int = 0) -> None:
        self.graph = graph
        self.batch_size = batch_size
        self.seed = seed
        self._batch_index = 0

    def reset(self) -> None:
        self.graph.validate()
        self.graph.reset(self.seed)
        self._batch_index = 0

    def step(self) -> dict[str, Signal]:
        """Run a single batch and return all produced signals (``"block.port"``)."""
        ctx = RunContext(batch_size=self.batch_size, batch_index=self._batch_index, seed=self.seed)
        values = self.graph.execute(ctx)
        self._batch_index += 1
        return values

    def run(self, n_batches: int | None = None, stop: StopCriteria | None = None,
            capture: Iterable[str] | str = (), reset: bool = True, progress: bool = False,
            on_batch: Callable[[int, dict[str, float]], bool | None] | None = None,
            workers: int = 1) -> SimulationResult:
        """Run until the stop criterion is met.

        Args:
            n_batches: Run exactly this many batches (shortcut for a StopCriteria).
            stop: Stop criteria (mutually exclusive with ``n_batches``).
            capture: Signals of the last batch to keep (``"block.port"`` keys, or ``"*"``).
            reset: Re-seed and clear block state before running.
            progress: Print a line per batch.
            on_batch: Called after every batch with ``(n_batches, metrics)``; returning
                ``False`` cancels the run.
            workers: Worker processes. Above 1, batches run on several CPU cores
                (see :mod:`linkstudio.core.parallel`); ``reset`` is then implied.
        """
        if n_batches is not None and stop is not None:
            raise ValueError("Pass either n_batches or stop, not both.")
        if stop is None:
            n = n_batches or 1
            stop = StopCriteria(max_batches=n, min_batches=n)
        if reset:
            self.reset()
        keys = [capture] if isinstance(capture, str) else list(capture)

        start = time.perf_counter()
        if workers > 1:
            from linkstudio.core.parallel import run_batches

            def done(n: int, metrics: dict[str, float]) -> str | None:
                reason = stop.check(n, time.perf_counter() - start, metrics)
                if on_batch is not None and on_batch(n, metrics) is False:
                    reason = "cancelled"
                return reason

            self.reset()
            n_done, reason, captures = run_batches(self.graph, self.batch_size, self.seed, workers, done, keys,
                                                    stop.max_batches)
            missing = [k for k in keys if k != "*" and k not in captures]
            if missing:
                raise GraphError(f"Cannot capture {missing}. Available: {sorted(captures)}")
            return SimulationResult(self.graph.metrics(), n_done, time.perf_counter() - start, reason, captures)

        n_done = 0
        while True:
            values = self.step()
            n_done += 1
            metrics = self.graph.metrics()
            elapsed = time.perf_counter() - start
            if progress:
                shown = ", ".join(f"{k}={_fmt(v)}" for k, v in list(metrics.items())[:4])
                print(f"  batch {n_done}: {shown}")
            reason = stop.check(n_done, elapsed, metrics)
            if on_batch is not None and on_batch(n_done, metrics) is False:
                reason = "cancelled"
            if reason is not None:
                break

        if "*" in keys:
            captures = dict(values)
        else:
            missing = [k for k in keys if k not in values]
            if missing:
                raise GraphError(f"Cannot capture {missing}. Available: {sorted(values)}")
            captures = {k: values[k] for k in keys}
        return SimulationResult(metrics, n_done, time.perf_counter() - start, reason, captures)
