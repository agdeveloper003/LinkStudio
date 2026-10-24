"""Constellations with Gray labelling, hard and soft (LLR) demapping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from linkstudio.phy.dsp.bits import bits_to_ints, gray_encode, ints_to_bits

_CHUNK = 1 << 16  # symbols processed at once by the demappers (bounds memory use)


@dataclass(frozen=True, eq=False)
class Constellation:
    """A set of complex points indexed by their bit label.

    ``points[label]`` is the symbol transmitted for the integer ``label`` whose binary
    representation (MSB first) is the group of ``bits_per_symbol`` bits.
    Points are normalized to unit average energy.
    """

    points: np.ndarray
    name: str = "custom"
    bit_labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.complex128)
        m = pts.size
        if m < 2 or m & (m - 1):
            raise ValueError(f"Constellation size must be a power of two >= 2, got {m}.")
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "bit_labels", ints_to_bits(np.arange(m)[:, None], self.bits_per_symbol))

    @property
    def order(self) -> int:
        return int(self.points.size)

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.order))

    def map(self, bits: np.ndarray) -> np.ndarray:
        """Bits ``(..., n*k)`` -> symbols ``(..., n)``."""
        return self.points[bits_to_ints(bits, self.bits_per_symbol)]

    def nearest(self, y: np.ndarray) -> np.ndarray:
        """Label of the closest point for every received sample."""
        flat = y.reshape(-1)
        out = np.empty(flat.size, dtype=np.int64)
        for s in range(0, flat.size, _CHUNK):
            d2 = np.abs(flat[s:s + _CHUNK, None] - self.points[None, :]) ** 2
            out[s:s + _CHUNK] = np.argmin(d2, axis=1)
        return out.reshape(y.shape)

    def demap_hard(self, y: np.ndarray) -> np.ndarray:
        """Minimum-distance decisions: symbols ``(..., n)`` -> bits ``(..., n*k)``."""
        return ints_to_bits(self.nearest(y), self.bits_per_symbol)

    def demap_soft(self, y: np.ndarray, noise_var: float | np.ndarray, method: str = "maxlog") -> np.ndarray:
        """Bit LLRs ``log P(b=0|y) / P(b=1|y)`` for an AWGN observation ``y = s + n``.

        Args:
            y: Received symbols ``(..., n)``.
            noise_var: Complex noise variance, scalar or broadcastable to ``y``.
            method: ``"exact"`` (log-sum-exp) or ``"maxlog"`` (max approximation).
        """
        if method not in ("exact", "maxlog"):
            raise ValueError(f"Unknown soft demapping method '{method}'.")
        k = self.bits_per_symbol
        flat = y.reshape(-1)
        nv = np.broadcast_to(np.asarray(noise_var, dtype=np.float64), y.shape).reshape(-1)
        llr = np.empty((flat.size, k))
        zero = self.bit_labels == 0  # (M, k)
        for s in range(0, flat.size, _CHUNK):
            metric = -np.abs(flat[s:s + _CHUNK, None] - self.points[None, :]) ** 2
            metric /= np.maximum(nv[s:s + _CHUNK, None], 1e-300)
            for i in range(k):
                m0, m1 = metric[:, zero[:, i]], metric[:, ~zero[:, i]]
                if method == "exact":
                    llr[s:s + _CHUNK, i] = logsumexp(m0, axis=1) - logsumexp(m1, axis=1)
                else:
                    llr[s:s + _CHUNK, i] = m0.max(axis=1) - m1.max(axis=1)
        return llr.reshape(*y.shape[:-1], -1)


def psk(order: int, phase_offset: float | None = None) -> Constellation:
    """Gray-labelled M-PSK. Default offset is pi/4 for QPSK and 0 otherwise."""
    if phase_offset is None:
        phase_offset = np.pi / 4 if order == 4 else 0.0
    pos = np.arange(order)
    points = np.empty(order, dtype=np.complex128)
    points[gray_encode(pos)] = np.exp(1j * (2 * np.pi * pos / order + phase_offset))
    return Constellation(points, name=f"{order}-PSK")


def _pam_levels(order: int) -> np.ndarray:
    """Amplitude levels of Gray-labelled M-PAM, indexed by label."""
    pos = np.arange(order)
    levels = np.empty(order)
    levels[gray_encode(pos)] = 2 * pos - (order - 1)
    return levels


def pam(order: int) -> Constellation:
    """Gray-labelled M-PAM (real-valued points)."""
    return Constellation(_pam_levels(order).astype(np.complex128), name=f"{order}-PAM")


def qam(order: int) -> Constellation:
    """Square Gray-labelled M-QAM (M = 4, 16, 64, 256, 1024, ...).

    The first half of each bit label selects the in-phase level, the second half
    the quadrature level, each with an independent Gray PAM labelling.
    """
    side = int(round(np.sqrt(order)))
    if side * side != order or side & (side - 1):
        raise ValueError(f"Square QAM needs M = 4^n, got {order}.")
    half = int(np.log2(side))
    levels = _pam_levels(side)
    labels = np.arange(order)
    points = levels[labels >> half] + 1j * levels[labels & (side - 1)]
    return Constellation(points, name=f"{order}-QAM")


def make_constellation(scheme: str, order: int, phase_offset_deg: float | None = None,
                       points: list[tuple[float, float]] | None = None) -> Constellation:
    """Factory used by the mapper/demapper blocks."""
    if scheme == "psk":
        return psk(order, None if phase_offset_deg is None else np.deg2rad(phase_offset_deg))
    if scheme == "qam":
        return qam(order)
    if scheme == "pam":
        return pam(order)
    if scheme == "custom":
        if not points:
            raise ValueError("A custom constellation needs 'points'.")
        return Constellation(np.array([complex(re, im) for re, im in points]), name="custom")
    raise ValueError(f"Unknown modulation scheme '{scheme}'.")
