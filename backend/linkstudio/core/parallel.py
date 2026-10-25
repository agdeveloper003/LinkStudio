"""Multi-core execution: batches and sweep points run in a pool of worker processes.

Batch ``k`` of a parallel run is seeded from ``(seed, k)``, so the result does not
depend on the number of workers or on the order in which batches finish. Workers
return the accumulator state of their blocks (:meth:`Graph.export_state`), which is
merged in batch order into a local copy of the graph; stop criteria are evaluated
after every merged batch, exactly as in a serial run.

A parallel run is reproducible, but it is a different random realization from the
serial run with the same seed (where all batches share one random stream per block).
Blocks that keep memory from one batch to the next (e.g. a slowly time-varying
channel) restart at every batch.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any

import numpy as np

from linkstudio.core.context import RunContext
from linkstudio.core.graph import Graph
from linkstudio.core.signal import Signal

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def cpu_count() -> int:
    return os.cpu_count() or 1


def get_pool() -> ProcessPoolExecutor:
    """Process pool shared by all runs, one process per CPU core (started on first use).

    A run with ``workers`` keeps at most that many tasks per worker in flight, so
    it uses about ``workers`` cores.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            # One BLAS/FFT thread per process: the parallelism comes from the processes.
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            _pool = ProcessPoolExecutor(max_workers=cpu_count())
        return _pool


def batch_seed(seed: int, index: int) -> int:
    """Independent seed of batch ``index`` of a parallel run."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


# ------------------------------------------------------------- worker side
_graph_cache: dict[str, Graph] = {}


def _graph(spec: Mapping[str, Any], key: str) -> Graph:
    graph = _graph_cache.get(key)
    if graph is None:
        import linkstudio  # noqa: F401  (registers the built-in blocks in a fresh process)
        from linkstudio.core.scenario import graph_from_dict

        _graph_cache.clear()
        graph = graph_from_dict(spec)
        graph.validate()
        _graph_cache[key] = graph
    return graph


def _run_chunk(spec: Mapping[str, Any], key: str, batch_size: int, seed: int, start: int, count: int,
               capture: Sequence[str]) -> tuple[int, list[dict[str, Any]], dict[str, Signal], float]:
    """Run batches ``start .. start+count-1``; returns one exported state per batch."""
    graph = _graph(spec, key)
    t0 = time.perf_counter()
    states = []
    values: dict[str, Signal] = {}
    for index in range(start, start + count):
        graph.reset(batch_seed(seed, index))
        values = graph.execute(RunContext(batch_size=batch_size, batch_index=index, seed=seed))
        states.append(graph.export_state())
    captured = values if "*" in capture else {k: values[k] for k in capture if k in values}
    return start, states, captured, time.perf_counter() - t0


def _run_point(spec: Mapping[str, Any], point: Mapping[str, Any], batch_size: int, seed: int,
               n_batches: int | None, stop: Any) -> dict[str, Any]:
    import linkstudio  # noqa: F401
    from linkstudio.core.scenario import graph_from_dict
    from linkstudio.core.simulation import Simulation

    graph = graph_from_dict(spec)
    for key, value in point.items():
        graph.set_param(key, value)
    res = Simulation(graph, batch_size, seed).run(n_batches=n_batches, stop=stop)
    return {"metrics": res.metrics, "n_batches": res.n_batches, "elapsed_s": res.elapsed_s}


# -------------------------------------------------------- coordinator side
def run_batches(graph: Graph, batch_size: int, seed: int, workers: int,
                done: Callable[[int, dict[str, float]], str | None],
                capture: Sequence[str] = (), max_batches: int | None = None) -> tuple[int, str, dict[str, Signal]]:
    """Run batches on ``workers`` processes until ``done`` returns a stop reason.

    ``graph`` must be reset; it receives the merged state. ``done(n, metrics)`` is
    called after every merged batch. Batches are sent in chunks sized to take about
    0.1 s each, which keeps the inter-process overhead small. Returns
    ``(n_batches, reason, captures)``.
    """
    from linkstudio.core.scenario import graph_to_dict

    spec = graph_to_dict(graph)
    key = repr(spec)
    pool = get_pool()
    limit = max_batches if max_batches is not None else 1 << 62
    in_flight: set[Future] = set()
    ready: dict[int, tuple[dict[str, Any], dict[str, Signal]]] = {}
    next_submit = next_merge = 0
    chunk = 1
    captures: dict[str, Signal] = {}
    reason: str | None = None
    try:
        while reason is None:
            while len(in_flight) < 2 * workers and next_submit < limit:
                count = min(chunk, limit - next_submit)
                in_flight.add(pool.submit(_run_chunk, spec, key, batch_size, seed, next_submit, count, tuple(capture)))
                next_submit += count
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                in_flight.discard(fut)
                start, states, captured, elapsed = fut.result()
                for k, state in enumerate(states):
                    ready[start + k] = (state, captured if k == len(states) - 1 else {})
                chunk = int(np.clip(round(0.1 * len(states) / max(elapsed, 1e-6)), 1, 64))
            while reason is None and next_merge in ready:
                state, captured = ready.pop(next_merge)
                graph.merge_state(state)
                captures = captured or captures
                next_merge += 1
                reason = done(next_merge, graph.metrics())
    finally:
        for fut in in_flight:
            fut.cancel()
    return next_merge, reason, captures


def run_points(graph: Graph, points: Sequence[Mapping[str, Any]], batch_size: int, seeds: Sequence[int],
               n_batches: int | None, stop: Any, workers: int,
               on_result: Callable[[int, dict[str, Any]], None],
               should_stop: Callable[[], bool] | None = None) -> None:
    """Simulate every sweep point in its own process; results are reported in order."""
    from linkstudio.core.scenario import graph_to_dict

    spec = graph_to_dict(graph)
    pool = get_pool()
    futures: list[Future] = []
    try:
        for i in range(len(points)):
            while len(futures) < min(len(points), i + workers):
                k = len(futures)
                futures.append(pool.submit(_run_point, spec, points[k], batch_size, seeds[k], n_batches, stop))
            while True:
                if should_stop is not None and should_stop():
                    return
                finished, _ = wait([futures[i]], timeout=0.2)
                if finished:
                    break
            on_result(i, futures[i].result())
    finally:
        for fut in futures:
            fut.cancel()
