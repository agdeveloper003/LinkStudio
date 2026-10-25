"""Pulse shapes and batched FIR filtering."""

from __future__ import annotations

import numpy as np
from scipy.signal import bessel, fftconvolve, freqs


def _time_axis(sps: int, span: int) -> np.ndarray:
    """Symmetric time axis in symbol periods with ``span*sps + 1`` taps."""
    n = span * sps
    return (np.arange(n + 1) - n / 2) / sps


def _unit_energy(h: np.ndarray) -> np.ndarray:
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def rrc_taps(sps: int, span: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine pulse, unit energy."""
    t = _time_axis(sps, span)
    b = rolloff
    if b == 0:
        return _unit_energy(np.sinc(t))
    h = np.empty_like(t)
    eps = 1e-9
    at_zero = np.abs(t) < eps
    at_sing = np.abs(np.abs(4 * b * t) - 1) < eps
    rest = ~(at_zero | at_sing)
    tt = t[rest]
    h[rest] = (np.sin(np.pi * tt * (1 - b)) + 4 * b * tt * np.cos(np.pi * tt * (1 + b))) / (
        np.pi * tt * (1 - (4 * b * tt) ** 2))
    h[at_zero] = 1 - b + 4 * b / np.pi
    h[at_sing] = b / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
    return _unit_energy(h)


def rc_taps(sps: int, span: int, rolloff: float) -> np.ndarray:
    """Raised-cosine (Nyquist) pulse, unit energy."""
    t = _time_axis(sps, span)
    b = rolloff
    h = np.sinc(t)
    if b > 0:
        sing = np.abs(np.abs(2 * b * t) - 1) < 1e-9
        rest = ~sing
        h[rest] *= np.cos(np.pi * b * t[rest]) / (1 - (2 * b * t[rest]) ** 2)
        h[sing] = np.pi / 4 * np.sinc(1 / (2 * b))
    return _unit_energy(h)


def gaussian_taps(sps: int, span: int, bt: float) -> np.ndarray:
    """Gaussian pulse (as in GMSK) with bandwidth-time product ``bt``, unit energy."""
    t = _time_axis(sps, span)
    h = np.exp(-2 * np.pi ** 2 * bt ** 2 * t ** 2 / np.log(2))
    return _unit_energy(h)


def rect_taps(sps: int) -> np.ndarray:
    """Rectangular (NRZ) pulse lasting one symbol, unit energy."""
    return np.full(sps, 1 / np.sqrt(sps))


def design_pulse(shape: str, sps: int, span: int = 10, rolloff: float = 0.25, bt: float = 0.3) -> np.ndarray:
    if shape == "rrc":
        return rrc_taps(sps, span, rolloff)
    if shape == "rc":
        return rc_taps(sps, span, rolloff)
    if shape == "gaussian":
        return gaussian_taps(sps, span, bt)
    if shape == "rect":
        return rect_taps(sps)
    raise ValueError(f"Unknown pulse shape '{shape}'.")


def upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Zero-stuffing along the last axis."""
    out = np.zeros((*x.shape[:-1], x.shape[-1] * factor), dtype=x.dtype)
    out[..., ::factor] = x
    return out


def fir_filter(x: np.ndarray, taps: np.ndarray, mode: str = "full") -> np.ndarray:
    """Convolve every row of ``x`` (last axis) with ``taps``.

    ``mode="full"`` returns ``n + len(taps) - 1`` samples, ``"causal"`` keeps the
    first ``n`` (same length, filter delay not compensated).
    """
    taps = np.asarray(taps).reshape((1,) * (x.ndim - 1) + (-1,))
    y = fftconvolve(x, taps, mode="full", axes=-1)
    if mode == "causal":
        y = y[..., :x.shape[-1]]
    elif mode != "full":
        raise ValueError(f"Unknown filter mode '{mode}'.")
    return y


def cubic_resample(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Lagrange interpolation of every row of ``x`` (B, N) at positions ``t`` (B, M)."""
    n = x.shape[-1]
    i = np.floor(t).astype(np.int64)
    mu = t - i
    xm1, x0, x1, x2 = (np.take_along_axis(x, np.clip(i + d, 0, n - 1), axis=-1) for d in (-1, 0, 1, 2))
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + x0


def apply_frequency_response(x: np.ndarray, response, fs: float, pad: int = 0) -> np.ndarray:
    """Filter the last axis of ``x`` with ``response(f)`` (f in Hz, two-sided FFT grid).

    The signal is zero-padded by ``pad`` samples to limit circular wrap-around; the
    output keeps the input length (and is real when ``x`` is real and the response
    Hermitian).
    """
    n = x.shape[-1]
    m = int(2 ** np.ceil(np.log2(n + pad)))
    f = np.fft.fftfreq(m, d=1.0 / fs)
    y = np.fft.ifft(np.fft.fft(x, n=m, axis=-1) * response(f), axis=-1)[..., :n]
    return y.real if not np.iscomplexobj(x) else y


def fractional_delay_taps(delay: float, half_len: int = 16) -> np.ndarray:
    """Windowed-sinc FIR delaying by ``half_len + delay`` samples (``0 <= delay < 1``)."""
    n = np.arange(2 * half_len + 1)
    h = np.sinc(n - half_len - delay) * np.blackman(2 * half_len + 1)
    return h / np.sum(h)


def dc_group_delay(response, fs: float) -> float:
    """Group delay (s) of ``response(f)`` at zero frequency, from the phase slope near f = 0."""
    df = fs * 1e-6
    return float(-np.angle(response(np.array([df]))[0]) / (2 * np.pi * df))


def bessel_thomson(f: np.ndarray, bandwidth_hz: float, order: int = 4) -> np.ndarray:
    """Bessel-Thomson low-pass response, -3 dB at ``bandwidth_hz`` (optical reference receiver)."""
    b, a = bessel(order, 2 * np.pi * bandwidth_hz, analog=True, norm="mag")
    return freqs(b, a, worN=2 * np.pi * np.asarray(f, dtype=float))[1]
