"""Matplotlib helpers to inspect signals and results (optional dependency)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from linkstudio.core.signal import Signal
from linkstudio.phy.dsp.analysis import eye_traces, psd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("linkstudio.viz needs matplotlib: pip install matplotlib") from exc


def _data(x: Signal | np.ndarray) -> np.ndarray:
    return x.data if isinstance(x, Signal) else np.asarray(x)


def _ax(ax: Any) -> Any:
    return ax if ax is not None else plt.subplots(figsize=(5, 5))[1]


def plot_constellation(x: Signal | np.ndarray, ax: Any = None, max_points: int = 4000,
                       reference: np.ndarray | None = None, title: str | None = None) -> Any:
    """Scatter plot of complex symbols (optionally with ideal reference points)."""
    ax = _ax(ax)
    pts = _data(x).ravel()
    if pts.size > max_points:
        pts = pts[np.linspace(0, pts.size - 1, max_points).astype(int)]
    ax.scatter(pts.real, pts.imag, s=3, alpha=0.4, label="received")
    if reference is None and isinstance(x, Signal) and x.meta.get("constellation") is not None:
        reference = x.meta["constellation"].points
    if reference is not None:
        ax.scatter(reference.real, reference.imag, s=40, marker="x", color="red", label="ideal")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title or "Constellation")
    return ax


def plot_psd(x: Signal | np.ndarray, fs: float | None = None, ax: Any = None, label: str | None = None,
             nperseg: int = 1024, normalize: bool = True) -> Any:
    """Power spectral density in dB (Welch, averaged over the batch)."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 4))[1]
    if fs is None:
        fs = x.fs if isinstance(x, Signal) and x.fs else 1.0
    f, p = psd(_data(x), fs, nperseg)
    p_db = 10 * np.log10(np.maximum(p, 1e-30))
    if normalize:
        p_db -= p_db.max()
    ax.plot(f, p_db, label=label)
    ax.set_xlabel("Frequency [Hz]" if fs != 1.0 else "Normalized frequency")
    ax.set_ylabel("PSD [dB]")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_eye(x: Signal | np.ndarray, sps: int | None = None, delay: float | None = None, ax: Any = None,
             max_traces: int = 200, component: str = "real", skip_symbols: int = 8) -> Any:
    """Two-symbol eye diagram centred on the sampling instants.

    Args:
        x: Waveform (typically after a receive filter, see :func:`fir_filter`).
        sps: Samples per symbol (default: ``sps`` metadata).
        delay: Index of a symbol sampling instant (default: ``delay`` metadata).
        skip_symbols: Symbols skipped at the start to avoid filter transients.
    """
    if ax is None:
        ax = plt.subplots(figsize=(6, 4))[1]
    if isinstance(x, Signal):
        sps = sps or x.meta.get("sps")
        delay = x.meta.get("delay", 0.0) if delay is None else delay
    if not sps:
        raise ValueError("Eye diagram needs samples per symbol.")
    start = int(round(delay or 0)) - sps + skip_symbols * sps
    traces = eye_traces(_data(x), sps, max(start, 0), 2, max_traces)
    vals = traces.real if component == "real" else traces.imag
    t = np.arange(traces.shape[1]) / sps - 1.0
    ax.plot(t, vals.T, color="tab:blue", alpha=0.15, lw=0.8)
    ax.axvline(0.0, color="red", lw=0.8, ls=":")
    ax.set_xlabel("Time relative to sampling instant [symbols]")
    ax.set_title(f"Eye diagram ({component})")
    ax.grid(True, alpha=0.3)
    return ax


def plot_ber(curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
             theory: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
             ax: Any = None, xlabel: str = "Eb/N0 [dB]", ylabel: str = "BER", title: str | None = None) -> Any:
    """Semilog error-rate curves: simulated (markers) and theoretical (lines)."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 5))[1]
    for label, (xv, yv) in (theory or {}).items():
        ax.semilogy(xv, yv, "-", lw=1.2, label=label)
    for label, (xv, yv) in curves.items():
        y = np.asarray(yv, dtype=float)
        y = np.where(y > 0, y, np.nan)
        ax.semilogy(xv, y, "o--", ms=5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_waveform(x: Signal | np.ndarray, n: int = 200, ax: Any = None, title: str | None = None) -> Any:
    """I and Q of the first frame."""
    if ax is None:
        ax = plt.subplots(figsize=(8, 3))[1]
    d = np.atleast_2d(_data(x))[0, :n]
    ax.plot(d.real, label="I")
    ax.plot(d.imag, label="Q")
    ax.set_xlabel("Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Waveform")
    return ax


def save(fig: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path
