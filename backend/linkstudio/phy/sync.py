"""Synchronization: preamble insertion, symbol timing recovery, frame/CFO sync, carrier tracking.

A typical synchronized single-carrier receiver chain::

    mapper -> preamble -> pulse_shaper -> [channel: delay, CFO, phase noise, AWGN]
           -> matched_filter(decimate=false) -> timing_recovery -> frame_sync
           -> carrier_recovery -> demapper

* ``timing_recovery`` tracks the fractional sampling instant (Gardner loop),
* ``frame_sync`` finds the preamble (integer symbol offset), removes the coarse
  carrier frequency offset and the phase, and strips preamble and guard symbols,
* ``carrier_recovery`` tracks the residual phase (CFO residue, phase noise).
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation


def loop_gains(bandwidth: float, damping: float, detector_gain: float = 1.0) -> tuple[float, float]:
    """Proportional and integral gains of a second-order loop.

    ``bandwidth`` is the noise bandwidth normalized to the update rate.
    """
    theta = bandwidth / (damping + 1 / (4 * damping))
    denom = 1 + 2 * damping * theta + theta ** 2
    return 4 * damping * theta / denom / detector_gain, 4 * theta ** 2 / denom / detector_gain


def cubic_interpolate(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Lagrange (Farrow) interpolation of every row of ``x`` at fractional index ``t``."""
    n = x.shape[-1]
    i = np.floor(t).astype(np.int64)
    mu = t - i
    rows = np.arange(x.shape[0])
    xm1, x0, x1, x2 = (x[rows, np.clip(i + d, 0, n - 1)] for d in (-1, 0, 1, 2))
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + x0


@register_block("preamble", category="framing", subcategory="Preamble", label="Preamble Inserter")
class Preamble(Block):
    """Prepend a known preamble and append guard symbols to every frame.

    The preamble is made of two identical halves of pseudo-random QPSK symbols
    (Schmidl & Cox structure): the repetition gives a CFO-robust timing metric and
    a frequency estimate. Guard symbols give the receiver slack for unknown delays.
    """

    class Params(BlockParams):
        length: int = Field(64, ge=8, description="Preamble length in symbols (even).")
        guard: int = Field(16, ge=0, description="Zero symbols appended after the payload.")
        seed: int = Field(2024, description="Seed of the preamble sequence.")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def setup(self) -> None:
        if self.params.length % 2:
            raise BlockError("Preamble length must be even.")
        half = np.random.default_rng(self.params.seed).integers(0, 4, self.params.length // 2)
        seq = np.exp(1j * (np.pi / 4 + np.pi / 2 * half))
        self.sequence = np.concatenate([seq, seq])

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        B = x.data.shape[0]
        out = np.concatenate([np.broadcast_to(self.sequence, (B, self.sequence.size)), x.data,
                              np.zeros((B, self.params.guard), dtype=np.complex128)], axis=-1)
        return {"out": x.evolve(out, preamble=self.sequence, n_payload_symbols=x.length,
                                n_symbols=out.shape[-1])}


@register_block("timing_recovery", category="synchronization", subcategory="Timing", label="Timing Recovery")
class TimingRecovery(Block):
    """Symbol timing recovery: Gardner detector, cubic interpolator, second-order loop.

    Input: matched-filtered waveform at ``sps >= 2`` (``matched_filter`` with
    ``decimate: false``). Output: one symbol per symbol period, starting at the
    nominal delay. Integer-symbol ambiguities are left to ``frame_sync``.
    """

    class Params(BlockParams):
        loop_bandwidth: float = Field(0.01, gt=0, lt=0.5, description="Loop noise bandwidth (x symbol rate).")
        damping: float = Field(0.707, gt=0, description="Loop damping factor.")
        detector_gain: float = Field(1.0, gt=0, description="Gardner S-curve slope (~1 for RRC 0.35).")
        initial_estimate: bool = Field(True, description="Initialize with an Oerder & Meyr feedforward "
                                                          "estimate (needs sps >= 3).")
        estimate_symbols: int = Field(128, ge=8, description="Symbols used by the initial estimate.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def on_reset(self) -> None:
        self._last_offset = np.zeros(0)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        sps = x.meta.get("sps")
        if not sps or sps < 2:
            raise BlockError("Timing recovery needs an oversampled input (sps >= 2).")
        data = x.data
        B, L = data.shape
        start = float(x.meta.get("delay", 0.0))
        n_out = int((L - 3 - start) // sps) - 1
        if n_out <= 1:
            raise BlockError("Waveform too short for timing recovery.")
        p = self.params
        kp, ki = loop_gains(p.loop_bandwidth, p.damping, p.detector_gain)
        tau = np.zeros(B)  # timing correction in symbols
        if p.initial_estimate and sps >= 3:
            # Oerder & Meyr: phase of the symbol-rate spectral line of |x|^2.
            seg = slice(int(start), int(start) + p.estimate_symbols * sps)
            k = np.arange(seg.start, min(seg.stop, L))
            line = np.sum(np.abs(data[:, k]) ** 2 * np.exp(-2j * np.pi * k / sps), axis=1)
            eps = -np.angle(line) / (2 * np.pi)  # peak position modulo one symbol
            tau = np.mod(eps - start / sps + 0.5, 1.0) - 0.5
        integ = np.zeros(B)
        out = np.empty((B, n_out), dtype=np.complex128)
        prev = cubic_interpolate(data, np.clip(start + tau * sps, 1, L - 3))
        out[:, 0] = prev
        for m in range(1, n_out):
            t = np.clip(start + (m + tau) * sps, 1, L - 3)
            y = cubic_interpolate(data, t)
            mid = cubic_interpolate(data, np.clip(t - sps / 2, 1, L - 3))
            err = np.real(np.conj(mid) * (y - prev))  # > 0 when sampling late
            integ += ki * err
            tau -= kp * err + integ
            out[:, m] = y
            prev = y
        self._last_offset = tau
        return {"out": Signal(out, Kind.SYMBOLS, fs=x.fs / sps if x.fs else None,
                              meta={**x.meta, "sps": 1, "delay": 0.0, "n_symbols": n_out,
                                    "power": x.meta.get("power", 1.0)})}

    def metrics(self) -> dict[str, float]:
        if not self._last_offset.size:
            return {}
        return {"timing_offset_sym": float(np.mean(self._last_offset))}


@register_block("frame_sync", category="synchronization", subcategory="Frame", label="Frame Synchronizer")
class FrameSync(Block):
    """Preamble-based frame synchronization.

    1. Schmidl & Cox metric over ``search`` candidate offsets: coarse CFO estimate
       from the phase between the two preamble halves (range +-1/length cycles per
       symbol), insensitive to the exact timing,
    2. cross-correlation with the known preamble over all candidates on the
       CFO-corrected signal (timing and phase),
    3. CFO refinement at the exact preamble position,
    4. extraction of the payload symbols.
    """

    class Params(BlockParams):
        search: int = Field(32, ge=0, description="Maximum frame start offset searched (symbols).")
        correct_cfo: bool = Field(True, description="Estimate and remove the frequency offset.")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def on_reset(self) -> None:
        self._frames = 0
        self._cfo_sum = 0.0
        self._offset_sum = 0.0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        pre = x.meta.get("preamble")
        n_pay = x.meta.get("n_payload_symbols")
        if pre is None or n_pay is None:
            raise BlockError("frame_sync needs a preamble block upstream.")
        data = x.data
        B, N = data.shape
        L, h = pre.size, pre.size // 2
        D = min(self.params.search, N - L - n_pay)
        if D < 0:
            raise BlockError("Frame too short: increase the preamble guard.")
        rows = np.arange(B)[:, None]

        b_idx = np.arange(B)
        n_idx = np.arange(N)

        eps = np.zeros(B)
        if self.params.correct_cfo:
            cand = np.arange(D + 1)[:, None] + np.arange(h)  # (D+1, h)
            P = np.sum(np.conj(data[:, cand]) * data[:, cand + h], axis=-1)  # (B, D+1)
            R = np.sum(np.abs(data[:, cand + h]) ** 2, axis=-1)
            d0 = np.argmax(np.abs(P) ** 2 / np.maximum(R, 1e-12) ** 2, axis=1)
            eps = np.angle(P[b_idx, d0]) / (2 * np.pi * h)  # cycles per symbol
            data = data * np.exp(-2j * np.pi * eps[:, None] * n_idx)

        win = np.arange(D + 1)[:, None] + np.arange(L)  # (D+1, L)
        corr = np.sum(data[:, win] * np.conj(pre), axis=-1)  # (B, D+1)
        d = np.argmax(np.abs(corr), axis=1)

        if self.params.correct_cfo:  # refine with the exact preamble position
            first = d[:, None] + np.arange(h)
            fine = np.angle(np.sum(np.conj(data[rows, first]) * data[rows, first + h], axis=-1)) / (2 * np.pi * h)
            data = data * np.exp(-2j * np.pi * fine[:, None] * n_idx)
            eps = eps + fine
        pwin = d[:, None] + np.arange(L)
        phase = np.angle(np.sum(data[rows, pwin] * np.conj(pre), axis=-1))

        idx = d[:, None] + L + np.arange(n_pay)
        out = data[rows, idx] * np.exp(-1j * phase)[:, None]
        self._frames += B
        self._cfo_sum += float(np.sum(eps)) * (x.fs or 1.0)
        self._offset_sum += float(np.sum(d))
        meta = {k: v for k, v in x.meta.items() if k not in ("preamble", "n_payload_symbols")}
        return {"out": Signal(out, Kind.SYMBOLS, fs=x.fs, meta={**meta, "n_symbols": n_pay})}

    def metrics(self) -> dict[str, float]:
        if not self._frames:
            return {}
        return {"cfo_estimate": self._cfo_sum / self._frames, "frame_offset": self._offset_sum / self._frames}


@register_block("carrier_recovery", category="synchronization", subcategory="Carrier", label="Carrier Recovery (PLL)")
class CarrierRecovery(Block):
    """Decision-directed second-order PLL tracking residual phase and frequency offsets."""

    class Params(BlockParams):
        loop_bandwidth: float = Field(0.005, gt=0, lt=0.5, description="Loop noise bandwidth (x symbol rate).")
        damping: float = Field(0.707, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        const: Constellation | None = x.meta.get("constellation")
        if const is None:
            raise BlockError("carrier_recovery needs the constellation (mapper upstream).")
        kp, ki = loop_gains(self.params.loop_bandwidth, self.params.damping)
        B, N = x.data.shape
        phi = np.zeros(B)
        integ = np.zeros(B)
        out = np.empty_like(x.data, dtype=np.complex128)
        for n in range(N):
            y = x.data[:, n] * np.exp(-1j * phi)
            d = const.points[const.nearest(y)]
            err = np.imag(y * np.conj(d)) / np.maximum(np.abs(d) ** 2, 1e-12)
            integ += ki * err
            phi += kp * err + integ
            out[:, n] = y
        return {"out": x.evolve(out)}
