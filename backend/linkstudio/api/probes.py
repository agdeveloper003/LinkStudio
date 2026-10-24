"""Turn signals into compact, JSON-friendly plot data for the UI scopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from linkstudio.core.signal import Kind, Signal
from linkstudio.phy.dsp.analysis import eye_traces, psd


def available_views(sig: Signal) -> list[str]:
    """Plot types that make sense for a signal kind."""
    if sig.kind == Kind.SYMBOLS:
        return ["constellation", "time", "spectrum"]
    if sig.kind == Kind.SAMPLES:
        return ["spectrum", "constellation", "eye", "time"]
    if sig.kind == Kind.GRID:
        return ["constellation", "grid"]
    if sig.kind == Kind.CSI:
        return ["csi"]
    if sig.kind == Kind.LLR:
        return ["histogram", "time"]
    if sig.kind == Kind.BITS:
        return ["time"]
    return []


def _f(a: np.ndarray, digits: int = 4) -> list[float]:
    """Round to ``digits`` significant figures relative to the largest value (keeps tiny
    physical quantities such as photocurrents readable)."""
    a = np.asarray(a, dtype=float)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    if 0 < scale < 1e-2:
        decimals = digits - int(np.floor(np.log10(scale)))
        return np.round(a, decimals).tolist()
    return np.round(a, digits).tolist()


def _flat(sig: Signal, max_points: int) -> np.ndarray:
    d = np.asarray(sig.data).reshape(-1)
    if d.size > max_points:
        d = d[np.linspace(0, d.size - 1, max_points).astype(int)]
    return d


def compute_view(sig: Signal, view: str, max_points: int = 2000) -> dict[str, Any]:
    """Plot data for one signal.

    Returned dict has a ``type``: ``scatter`` (I/Q points), ``line`` (x + series),
    ``eye`` (overlaid traces) or ``heatmap``.
    """
    if view == "constellation":
        pts = _flat(sig, max_points)
        out: dict[str, Any] = {"type": "scatter", "i": _f(pts.real), "q": _f(pts.imag)}
        const = sig.meta.get("constellation")
        rms = float(np.sqrt(np.mean(np.abs(pts) ** 2))) if pts.size else 0.0
        # Reference points only when the signal is at symbol scale (not e.g. a photocurrent in A).
        if const is not None and const.order <= 256 and 0.25 < rms < 4:
            out.update(ref_i=_f(const.points.real), ref_q=_f(const.points.imag))
        return out

    if view == "spectrum":
        data = np.asarray(sig.data)
        data = data.reshape(-1, data.shape[-1])[:16]
        f, p = psd(data, sig.fs or 1.0, nperseg=min(512, data.shape[-1]))
        p_db = 10 * np.log10(np.maximum(p, 1e-30))
        return {"type": "line", "x": _f(f, 1), "series": {"PSD": _f(p_db - p_db.max(), 2)},
                "xlabel": "Frequency [Hz]" if sig.fs else "Normalized frequency", "ylabel": "dB"}

    if view == "time":
        d = np.asarray(sig.data).reshape(-1, sig.data.shape[-1])[0, :300]
        x = np.arange(d.size)
        if np.iscomplexobj(d):
            series = {"I": _f(d.real), "Q": _f(d.imag)}
        else:
            series = {"value": _f(d)}
        return {"type": "line", "x": x.tolist(), "series": series, "xlabel": "Index", "ylabel": "Amplitude"}

    if view == "eye":
        sps = int(sig.meta.get("sps") or 0)
        if sps < 2:
            return {"type": "message", "text": "Eye diagram needs an oversampled waveform."}
        delay = int(round(sig.meta.get("delay", 0.0)))
        d = np.asarray(sig.data).reshape(-1, sig.data.shape[-1])[0]
        traces = eye_traces(d, sps, max(delay - sps + 8 * sps, 0), 2, 80)
        scale = np.max(np.abs(traces)) or 1.0
        return {"type": "eye", "t": _f(np.arange(traces.shape[1]) / sps - 1.0),
                "traces": [_f(t.real / scale, 3) for t in traces]}

    if view == "histogram":
        d = _flat(sig, 20000)
        counts, edges = np.histogram(np.clip(d, -30, 30), bins=60)
        return {"type": "line", "x": _f((edges[:-1] + edges[1:]) / 2, 2), "series": {"count": counts.tolist()},
                "xlabel": "LLR", "ylabel": "Count"}

    if view == "grid":
        g = np.abs(np.asarray(sig.data)[0])
        return {"type": "heatmap", "z": _f(20 * np.log10(np.maximum(np.fft.fftshift(g, axes=-1), 1e-6)), 1),
                "xlabel": "Subcarrier", "ylabel": "OFDM symbol"}

    if view == "csi":
        return _csi_view(sig)

    return {"type": "message", "text": f"Unknown view '{view}'."}


def _csi_view(sig: Signal) -> dict[str, Any]:
    ctype = sig.meta.get("csi_type")
    d = np.asarray(sig.data)
    if ctype == "gain":
        h = d.reshape(d.shape[0], -1)[0]
        if h.size == 1:
            h = d.reshape(-1)[:200]
            label = "Frame"
        else:
            label = "Sample"
        h = h[:: max(1, h.size // 500)]
        return {"type": "line", "x": list(range(h.size)), "series": {"|h| dB": _f(20 * np.log10(np.abs(h) + 1e-12), 2)},
                "xlabel": label, "ylabel": "dB"}
    if ctype in ("cir", "cir_tv"):
        taps = d[0] if ctype == "cir" else d[0, :, 0]
        return {"type": "line", "x": list(range(taps.size)), "series": {"|h[n]|": _f(np.abs(taps))},
                "xlabel": "Delay [samples]", "ylabel": "Magnitude", "bars": True}
    if ctype == "freq":
        H = np.fft.fftshift(d[0, 0])
        n = H.size
        freqs = np.arange(-n // 2, n - n // 2)
        values: list[float | None] = _f(20 * np.log10(np.abs(H) + 1e-12), 2)
        layout = sig.meta.get("ofdm")
        if layout is not None:  # only the used subcarriers carry an estimate
            active = set(layout.active_freqs.tolist())
            values = [v if f in active else None for f, v in zip(freqs.tolist(), values)]
        return {"type": "line", "x": freqs.tolist(), "series": {"|H(f)| dB": values},
                "xlabel": "Subcarrier", "ylabel": "dB"}
    if ctype == "mimo":
        H = d[0] if d.ndim == 3 else d[0, 0]
        mag = np.abs(H).reshape(-1)
        return {"type": "line", "x": list(range(mag.size)), "series": {"|H_ij|": _f(mag)},
                "xlabel": "Entry (row-major)", "ylabel": "Magnitude", "bars": True}
    return {"type": "message", "text": f"Unsupported CSI type '{ctype}'."}


def compute_probes(values: dict[str, Signal], probes: list[dict[str, str]]) -> dict[str, Any]:
    """Plot data for every requested ``{"key": "block.port", "view": ...}`` probe."""
    out: dict[str, Any] = {}
    for probe in probes:
        key = probe["key"]
        sig = values.get(key)
        if sig is None:
            out[key] = {"type": "message", "text": f"No signal '{key}'."}
            continue
        views = available_views(sig)
        view = probe.get("view") or (views[0] if views else "")
        try:
            data = compute_view(sig, view)
        except Exception as exc:  # a broken probe must not stop the run
            data = {"type": "message", "text": f"{type(exc).__name__}: {exc}"}
        data.update(view=view, views=views, kind=sig.kind.label())
        out[key] = data
    return out
