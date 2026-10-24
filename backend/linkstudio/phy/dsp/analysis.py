"""Signal analysis helpers (spectra, PAPR, EVM, eye diagrams)."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD averaged over the batch, frequency axis centred on 0."""
    x = np.atleast_2d(x)
    nperseg = min(nperseg, x.shape[-1])
    f, pxx = welch(x, fs=fs, nperseg=nperseg, return_onesided=False, axis=-1, detrend=False)
    pxx = pxx.reshape(-1, pxx.shape[-1]).mean(axis=0)
    return np.fft.fftshift(f), np.fft.fftshift(pxx)


def papr_db(x: np.ndarray) -> np.ndarray:
    """Peak-to-average power ratio of each frame (dB)."""
    p = np.abs(x) ** 2
    return 10 * np.log10(p.max(axis=-1) / np.maximum(p.mean(axis=-1), 1e-300))


def evm_rms(reference: np.ndarray, measured: np.ndarray) -> float:
    """RMS error vector magnitude (ratio, not percent)."""
    err = np.mean(np.abs(measured - reference) ** 2)
    return float(np.sqrt(err / np.mean(np.abs(reference) ** 2)))


def eye_traces(x: np.ndarray, sps: int, offset: int = 0, symbols_per_trace: int = 2,
               max_traces: int = 300) -> np.ndarray:
    """Cut a waveform into overlapping traces of ``symbols_per_trace`` symbols."""
    x = np.atleast_2d(x)[0]
    length = symbols_per_trace * sps + 1
    starts = np.arange(offset, x.size - length, sps)[:max_traces]
    return np.stack([x[s:s + length] for s in starts]) if starts.size else np.empty((0, length))
