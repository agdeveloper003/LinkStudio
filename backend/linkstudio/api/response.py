"""Frequency response of one block, for the inspector plot.

The graph is run for one frame so that the block sees its real input (sample rate,
wavelength, ...). Blocks with a model transfer function
(:meth:`Block.frequency_response`) report it exactly; other waveform-to-waveform
blocks get an estimate H = S_xy / S_xx from their input and output samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import csd, welch

from linkstudio.core import Graph, Kind, Simulation, Signal
from linkstudio.core.graph import Endpoint

N_POINTS = 400


def _io(graph: Graph, values: dict[str, Signal], name: str) -> tuple[Signal | None, Signal | None]:
    block = graph.blocks[name]
    x = None
    for pname in block.input_ports:
        src = graph.edges.get(Endpoint(name, pname))
        if src is not None:
            x = values.get(str(src))
            break
    y = next((values[f"{name}.{p}"] for p in block.output_ports if f"{name}.{p}" in values), None)
    return x, y


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n")[0]


def _estimate(x: Signal, y: Signal) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch estimate of H(f) from frame 0, or None if the signals are not comparable."""
    if x.kind != Kind.SAMPLES or y.kind != Kind.SAMPLES or not x.fs or x.fs != y.fs:
        return None
    if (x.meta.get("unit") or "") != (y.meta.get("unit") or "") or x.data.ndim != 2 or y.data.ndim != 2:
        return None
    n = min(x.length, y.length)
    a, b = x.data[0, :n], y.data[0, :n]
    nperseg = min(1024, n)
    two_sided = np.iscomplexobj(a) or np.iscomplexobj(b)
    f, pxx = welch(a, x.fs, nperseg=nperseg, return_onesided=not two_sided, detrend=False)
    _, pxy = csd(a, b, x.fs, nperseg=nperseg, return_onesided=not two_sided, detrend=False)
    h = pxy / np.maximum(pxx, 1e-300)
    h[pxx < 1e-6 * pxx.max()] = np.nan  # no input power there: H is undefined
    order = np.argsort(f)
    return f[order], h[order]


def block_response(graph: Graph, name: str, seed: int = 0) -> dict[str, Any]:
    """Magnitude, phase and group delay of block ``name`` over its sample-rate band."""
    if name not in graph.blocks:
        return {"available": False, "reason": f"Unknown block '{name}'."}
    block = graph.blocks[name]
    sim = Simulation(graph, batch_size=1, seed=seed)
    sim.reset()
    values = sim.step()
    x, y = _io(graph, values, name)
    if x is None:
        return {"available": False, "reason": "The block has no connected input."}
    fs = (y.fs if y is not None and y.fs else None) or x.fs
    h = None
    source = "model"
    if fs:
        two_sided = block.two_sided_response
        f = np.linspace(-fs / 2 if two_sided else 0.0, fs / 2, N_POINTS)
        h = block.frequency_response(f, x)
    if h is None and y is not None and block.estimate_response:
        est = _estimate(x, y)
        if est is not None:
            f, h = est
            source = "estimate"
    if h is None:
        return {"available": False, "reason": "No frequency response for this block."}
    h = np.asarray(h, dtype=complex)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    mag = np.maximum(mag, np.nanmax(mag) - 100)  # floor deep stop-band nulls
    phase = np.unwrap(np.nan_to_num(np.angle(h)))
    gd = -np.gradient(phase, 2 * np.pi * f)
    nan = np.isnan(h)
    return {
        "available": True,
        "source": source,
        "f": f.tolist(),
        "magnitude_db": [None if m else round(float(v), 3) for v, m in zip(mag, nan)],
        "phase_deg": [None if m else round(float(np.degrees(v)), 2) for v, m in zip(phase, nan)],
        "group_delay_s": [None if m else float(v) for v, m in zip(gd, nan)],
        "doc": _first_line(type(block).frequency_response.__doc__ or type(block).__doc__) if source == "model" else
        "One channel realization, estimated from the simulated input and output (S_xy / S_xx).",
    }
