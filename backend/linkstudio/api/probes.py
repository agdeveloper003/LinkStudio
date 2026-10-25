"""Turn signals into compact, JSON-friendly plot data for the UI scopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from linkstudio.core.block import BlockError
from linkstudio.core.graph import Graph
from linkstudio.core.registry import get_block_class
from linkstudio.core.signal import Kind, Signal
from linkstudio.phy.scopes import Scope
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


def _unit(sig: Signal) -> str:
    return sig.meta.get("unit") or ""


def _psd_unit(unit: str) -> tuple[str, float]:
    """Unit of 10 log10(PSD) and the offset to add, for samples in ``unit``."""
    if unit == "√W":
        return "dBm/Hz", 30.0
    if unit:
        return f"dB{unit}²/Hz", 0.0
    return "dB/Hz", 0.0


def compute_view(sig: Signal, view: str, max_points: int = 2000) -> dict[str, Any]:
    """Plot data for one signal.

    Returned dict has a ``type``: ``scatter`` (I/Q points), ``line`` (x + series),
    ``eye`` (overlaid traces) or ``heatmap``. Physical quantities are sent in base SI
    units together with their unit (``xunit``, ``yunit``, ``unit``); the UI chooses
    the prefix (ps, GHz, mA, ...).
    """
    unit = _unit(sig)
    if view == "constellation":
        pts = _flat(sig, max_points)
        out: dict[str, Any] = {"type": "scatter", "i": _f(pts.real), "q": _f(pts.imag), "unit": unit}
        const = sig.meta.get("constellation")
        rms = float(np.sqrt(np.mean(np.abs(pts) ** 2))) if pts.size else 0.0
        # Reference points only when the signal is at symbol scale (not e.g. a photocurrent in A).
        if const is not None and const.order <= 256 and 0.25 < rms < 4 and not unit:
            out.update(ref_i=_f(const.points.real), ref_q=_f(const.points.imag))
        return out

    if view == "spectrum":
        data = np.asarray(sig.data)
        data = data.reshape(-1, data.shape[-1])[:16]
        f, p = psd(data, sig.fs or 1.0, nperseg=min(512, data.shape[-1]))
        if not np.iscomplexobj(data):  # real signal: one-sided density
            keep = f >= 0
            f, p = f[keep], p[keep] * np.where(f[keep] > 0, 2.0, 1.0)
        yunit, offset = _psd_unit(unit)
        p_db = 10 * np.log10(np.maximum(p, 1e-300)) + offset
        p_db = np.maximum(p_db, p_db.max() - 120)
        return {"type": "line", "x": _f(f, 1) if sig.fs else _f(f), "series": {"PSD": _f(p_db, 2)},
                "xlabel": "Frequency" if sig.fs else "Normalized frequency", "xunit": "Hz" if sig.fs else "",
                "ylabel": "PSD", "yunit": yunit}

    if view == "time":
        d = np.asarray(sig.data).reshape(-1, sig.data.shape[-1])[0, :300]
        if np.iscomplexobj(d):
            series = {"I": _f(d.real), "Q": _f(d.imag)}
        else:
            series = {"value": _f(d)}
        if sig.fs:
            x, xlabel, xunit = _f(np.arange(d.size) / sig.fs), "Time", "s"
        else:
            x, xlabel, xunit = list(range(d.size)), "Index", ""
        return {"type": "line", "x": x, "series": series, "xlabel": xlabel, "xunit": xunit,
                "ylabel": "Amplitude", "yunit": unit}

    if view == "eye":
        sps = int(sig.meta.get("sps") or 0)
        if sps < 2:
            return {"type": "message", "text": "Eye diagram needs an oversampled waveform."}
        delay = int(round(sig.meta.get("delay", 0.0)))
        d = np.asarray(sig.data).reshape(-1, sig.data.shape[-1])[0]
        traces = eye_traces(d, sps, max(delay - sps + 8 * sps, 0), 2, 120).real
        t_ui = np.arange(traces.shape[1]) / sps - 1.0
        out = {"type": "eye", "traces": [_f(t) for t in traces], "unit": unit}
        if sig.fs:
            out.update(t=_f(t_ui * sps / sig.fs), tunit="s", ui=sps / sig.fs)
        else:
            out.update(t=_f(t_ui), tunit="UI", ui=1.0)
        return out

    if view == "histogram":
        d = _flat(sig, 20000)
        counts, edges = np.histogram(np.clip(d, -30, 30), bins=60)
        return {"type": "line", "x": _f((edges[:-1] + edges[1:]) / 2, 2), "series": {"count": counts.tolist()},
                "xlabel": "LLR", "xunit": "", "ylabel": "Count", "yunit": ""}

    if view == "grid":
        g = np.abs(np.asarray(sig.data)[0])
        return {"type": "heatmap", "z": _f(20 * np.log10(np.maximum(np.fft.fftshift(g, axes=-1), 1e-6)), 1),
                "xlabel": "Subcarrier", "ylabel": "OFDM symbol", "zunit": "dB"}

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
        return {"type": "line", "x": list(range(h.size)), "series": {"|h|": _f(20 * np.log10(np.abs(h) + 1e-12), 2)},
                "xlabel": label, "xunit": "", "ylabel": "Gain", "yunit": "dB"}
    if ctype in ("cir", "cir_tv"):
        taps = d[0] if ctype == "cir" else d[0, :, 0]
        if sig.fs:
            x, xunit = _f(np.arange(taps.size) / sig.fs), "s"
        else:
            x, xunit = list(range(taps.size)), "samples"
        return {"type": "line", "x": x, "series": {"|h[n]|": _f(np.abs(taps))},
                "xlabel": "Delay", "xunit": xunit, "ylabel": "Magnitude", "yunit": "", "bars": True}
    if ctype == "freq":
        H = np.fft.fftshift(d[0, 0])
        n = H.size
        freqs = np.arange(-n // 2, n - n // 2)
        values: list[float | None] = _f(20 * np.log10(np.abs(H) + 1e-12), 2)
        layout = sig.meta.get("ofdm")
        if layout is not None:  # only the used subcarriers carry an estimate
            active = set(layout.active_freqs.tolist())
            values = [v if f in active else None for f, v in zip(freqs.tolist(), values)]
        x: list[float] = freqs.tolist()
        xlabel, xunit = "Subcarrier", ""
        if layout is not None and sig.fs:
            x, xlabel, xunit = _f(freqs * sig.fs / n, 1), "Frequency", "Hz"
        return {"type": "line", "x": x, "series": {"|H(f)|": values},
                "xlabel": xlabel, "xunit": xunit, "ylabel": "Gain", "yunit": "dB"}
    if ctype == "mimo":
        H = d[0] if d.ndim == 3 else d[0, 0]
        mag = np.abs(H).reshape(-1)
        return {"type": "line", "x": list(range(mag.size)), "series": {"|H_ij|": _f(mag)},
                "xlabel": "Entry (row-major)", "xunit": "", "ylabel": "Magnitude", "yunit": "", "bars": True}
    return {"type": "message", "text": f"Unsupported CSI type '{ctype}'."}


def scope_probes(editor: dict[str, Any]) -> list[dict[str, str]]:
    """One probe per connected scope input of an editor graph: the output feeding it.

    Returned as ``{"id": "scope.inN", "key": "block.port", "view": ...}``.
    """
    sources = {e["target"]: e["source"] for e in editor.get("edges") or []}
    probes = []
    for name, b in (editor.get("blocks") or {}).items():
        if b.get("type") == "subsystem":
            continue
        try:
            cls = get_block_class(b["type"])
        except BlockError:
            continue
        if not issubclass(cls, Scope):
            continue
        for port in cls.inputs:
            ref = f"{name}.{port}"
            if ref in sources:
                probes.append({"id": ref, "key": sources[ref], "view": cls.view})
    return probes


def block_displays(graph: Graph, values: dict[str, Signal]) -> dict[str, Any]:
    """Live data drawn inside blocks with a custom look (beyond their metrics).

    * ``mimo_channel``: ``gains``, the magnitude |h_rt| (Nr x Nt) of the first frame,
      averaged over the frame for fast fading.
    """
    out: dict[str, Any] = {}
    for name, block in graph.blocks.items():
        if block.type_name == "mimo_channel":
            csi = values.get(f"{name}.csi")
            if csi is None:
                continue
            h = np.abs(np.asarray(csi.data)[0])
            if h.ndim == 3:  # fast fading: (n, Nr, Nt)
                h = h.mean(axis=0)
            out[name] = {"gains": _f(h, 3)}
    return out


def compute_probes(values: dict[str, Signal], probes: list[dict[str, str]]) -> dict[str, Any]:
    """Plot data for every requested ``{"key": "block.port", "view": ...}`` probe, keyed by
    the probe ``id`` (the scope name) when given, else by ``key``."""
    out: dict[str, Any] = {}
    for probe in probes:
        key = probe["key"]
        out_key = probe.get("id") or key
        sig = values.get(key)
        if sig is None:
            out[out_key] = {"type": "message", "text": f"No signal '{key}'."}
            continue
        views = available_views(sig)
        view = probe.get("view") or (views[0] if views else "")
        try:
            data = compute_view(sig, view)
        except Exception as exc:  # a broken probe must not stop the run
            data = {"type": "message", "text": f"{type(exc).__name__}: {exc}"}
        data.update(view=view, views=views, kind=sig.kind.label())
        out[out_key] = data
    return out
