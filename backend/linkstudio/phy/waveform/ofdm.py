"""OFDM: resource mapping, (I)FFT with cyclic prefix, pilot-based channel estimation and equalization.

Subcarrier layout: ``n_active`` subcarriers centred around DC (DC itself unused).
Every ``pilot_spacing``-th active subcarrier carries a known pilot (the first and the
last active subcarriers are always pilots so that interpolation never extrapolates).
Pilots are a fixed pseudo-random QPSK sequence across frequency: identical pilots
would produce a peaky, non-stationary time-domain signal.
The IFFT/FFT pair is unitary (``norm="ortho"``), so a symbol of energy Es on a
subcarrier stays Es in frequency domain and the noise variance per subcarrier equals
the noise variance per time-domain sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block


@dataclass(frozen=True, eq=False)
class OFDMLayout:
    n_fft: int
    cp_len: int
    active_freqs: np.ndarray  # signed subcarrier indices, ascending
    pilot_freqs: np.ndarray
    data_freqs: np.ndarray
    pilot_values: np.ndarray  # one unit-modulus value per pilot subcarrier

    @classmethod
    def build(cls, n_fft: int, cp_len: int, n_active: int, pilot_spacing: int,
              pilot_seed: int = 0) -> OFDMLayout:
        if not 1 <= n_active < n_fft:
            raise ValueError(f"n_active must be in [1, n_fft - 1] (DC is unused), got {n_active}.")
        if cp_len < 0 or cp_len > n_fft:
            raise ValueError("cp_len must be in [0, n_fft].")
        freqs = np.concatenate([np.arange(-(n_active // 2), 0), np.arange(1, n_active - n_active // 2 + 1)])
        is_pilot = np.zeros(n_active, dtype=bool)
        if pilot_spacing > 0:
            is_pilot[::pilot_spacing] = True
            is_pilot[-1] = True
        phases = np.random.default_rng(pilot_seed).integers(0, 4, int(is_pilot.sum()))
        pilots = np.exp(1j * (np.pi / 4 + np.pi / 2 * phases))
        return cls(n_fft, cp_len, freqs, freqs[is_pilot], freqs[~is_pilot], pilots)

    def bins(self, freqs: np.ndarray) -> np.ndarray:
        return np.mod(freqs, self.n_fft)

    @property
    def n_data(self) -> int:
        return int(self.data_freqs.size)

    @property
    def symbol_len(self) -> int:
        return self.n_fft + self.cp_len


class _LayoutParams(BlockParams):
    n_fft: int = Field(64, ge=2, description="FFT size.")
    cp_len: int = Field(16, ge=0, description="Cyclic prefix length (samples).")
    n_active: int = Field(52, ge=1, description="Used subcarriers (excluding DC).")
    pilot_spacing: int = Field(6, ge=0, description="Pilot every N active subcarriers (0: no pilots).")


@register_block("ofdm_modulator", category="modulator", subcategory="Multicarrier", label="OFDM Modulator")
class OFDMModulator(Block):
    """Map data symbols and pilots on subcarriers, IFFT and add the cyclic prefix.

    The last OFDM symbol is zero-padded when needed. The ``sps`` metadata is set to
    ``n_fft / n_active`` so that Es/N0 in the AWGN block refers to the per-subcarrier
    SNR (cyclic-prefix and pilot overhead are not counted).
    """

    class Params(_LayoutParams):
        subcarrier_spacing: float = Field(15e3, gt=0, description="Subcarrier spacing in Hz.")
        tail_zeros: int = Field(0, ge=0, description="Zero samples appended to the frame (slack for timing sync).")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def setup(self) -> None:
        p = self.params
        self.layout = OFDMLayout.build(p.n_fft, p.cp_len, p.n_active, p.pilot_spacing)
        if self.layout.n_data == 0:
            raise BlockError("No data subcarriers left: reduce pilot density.")

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        lay = self.layout
        B, n = x.data.shape[0], x.length
        n_sym = -(-n // lay.n_data)
        data = np.zeros((B, n_sym * lay.n_data), dtype=np.complex128)
        data[:, :n] = x.data
        grid = np.zeros((B, n_sym, lay.n_fft), dtype=np.complex128)
        grid[..., lay.bins(lay.data_freqs)] = data.reshape(B, n_sym, lay.n_data)
        grid[..., lay.bins(lay.pilot_freqs)] = lay.pilot_values
        time = np.fft.ifft(grid, axis=-1, norm="ortho")
        with_cp = np.concatenate([time[..., lay.n_fft - lay.cp_len:], time], axis=-1)
        fs = self.params.subcarrier_spacing * lay.n_fft
        data_power = (x.meta.get("power") or 1.0)
        n_pilots = lay.pilot_freqs.size
        meta = {**x.meta, "ofdm": lay, "n_data_symbols": n, "sps": lay.n_fft / lay.active_freqs.size,
                "delay": 0.0, "data_power": data_power,
                "power": (lay.n_data * data_power + n_pilots) / lay.n_fft}
        frame = with_cp.reshape(B, -1)
        if self.params.tail_zeros:
            frame = np.concatenate([frame, np.zeros((B, self.params.tail_zeros), dtype=frame.dtype)], axis=-1)
        return {"out": Signal(frame, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("ofdm_sync", category="synchronization", subcategory="Frame", label="OFDM Synchronizer")
class OFDMSync(Block):
    """Blind OFDM timing and fractional CFO synchronization from the cyclic prefix (van de Beek).

    The correlation between the cyclic prefix and the end of each OFDM symbol, accumulated
    over the frame, peaks at the symbol start; its phase gives the frequency offset in
    subcarrier spacings (range +-0.5). The frame is realigned and the offset removed.
    Integer CFO is not estimated. Use ``tail_zeros`` in the modulator as timing slack.
    """

    class Params(BlockParams):
        max_delay: int | None = Field(None, ge=0, description="Largest delay searched (default: one OFDM symbol).")
        backoff: int | None = Field(None, ge=0, description="Start the FFT window this many samples early, inside "
                                                            "the CP, to absorb delay spread (default: cp_len / 4).")
        correct_cfo: bool = True

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def on_reset(self) -> None:
        self._cfo: list[float] = []
        self._delay: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        lay: OFDMLayout | None = x.meta.get("ofdm")
        if lay is None:
            raise BlockError("ofdm_sync needs an ofdm_modulator upstream.")
        if lay.cp_len == 0:
            raise BlockError("CP-based synchronization needs a cyclic prefix.")
        r = x.data
        B, n = r.shape
        L, N, cp = lay.symbol_len, lay.n_fft, lay.cp_len
        D = min(self.params.max_delay if self.params.max_delay is not None else L - 1, n - L)
        prod = r[:, :n - N] * np.conj(r[:, N:])  # r[k] r*[k+N]
        csum = np.concatenate([np.zeros((B, 1)), np.cumsum(prod, axis=1)], axis=1)
        n_sym = (n - N - cp - D) // L + 1
        theta = np.arange(D + 1)
        gamma = np.zeros((B, D + 1), dtype=np.complex128)
        for m in range(max(n_sym, 1)):
            start = theta + m * L
            gamma += csum[:, start + cp] - csum[:, start]
        d = np.argmax(np.abs(gamma), axis=1)
        eps = -np.angle(gamma[np.arange(B), d]) / (2 * np.pi) if self.params.correct_cfo else np.zeros(B)
        backoff = self.params.backoff if self.params.backoff is not None else cp // 4
        d = np.maximum(d - backoff, 0)
        out = np.zeros_like(r)
        k = np.arange(n)
        for b in range(B):
            seg = r[b, d[b]:] * np.exp(-2j * np.pi * eps[b] * (k[: n - d[b]] + d[b]) / N)
            out[b, : seg.size] = seg
        self._cfo.extend((eps * (x.fs or N) / N).tolist())
        self._delay.extend(d.astype(float).tolist())
        return {"out": x.evolve(out)}

    def metrics(self) -> dict[str, float]:
        if not self._cfo:
            return {}
        return {"cfo_estimate": float(np.mean(self._cfo)), "timing_offset": float(np.mean(self._delay))}


@register_block("ofdm_demodulator", category="demodulator", subcategory="Multicarrier", label="OFDM Demodulator")
class OFDMDemodulator(Block):
    """Remove the cyclic prefix and FFT every OFDM symbol into a resource grid.

    The layout is taken from the upstream modulator unless ``n_fft`` is given.
    ``fft_offset`` starts the FFT window that many samples early (inside the CP).
    """

    class Params(BlockParams):
        n_fft: int | None = Field(None, ge=2, description="Override layout (else from metadata).")
        cp_len: int = 16
        n_active: int = 52
        pilot_spacing: int = 6
        fft_offset: int = Field(0, ge=0, description="Samples to start the FFT window early.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.GRID, doc="Resource grid (batch, n_ofdm_symbols, n_fft).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        lay = (OFDMLayout.build(p.n_fft, p.cp_len, p.n_active, p.pilot_spacing) if p.n_fft
               else x.meta.get("ofdm"))
        if lay is None:
            raise BlockError("No OFDM layout: set n_fft or use an ofdm_modulator upstream.")
        if p.fft_offset > lay.cp_len:
            raise BlockError("fft_offset cannot exceed the cyclic prefix length.")
        B = x.data.shape[0]
        n_sym = x.length // lay.symbol_len
        frames = x.data[:, :n_sym * lay.symbol_len].reshape(B, n_sym, lay.symbol_len)
        start = lay.cp_len - p.fft_offset
        grid = np.fft.fft(frames[..., start:start + lay.n_fft], axis=-1, norm="ortho")
        return {"out": Signal(grid, Kind.GRID, fs=x.fs, meta={**x.meta, "ofdm": lay})}


@register_block("ofdm_channel_estimator", category="detector", label="OFDM Channel Estimator")
class OFDMChannelEstimator(Block):
    """Estimate the channel frequency response on every subcarrier.

    * ``ls``: least squares on pilots + linear interpolation across frequency
      (optionally averaged over the OFDM symbols of a frame, good for block fading).
    * ``perfect``: genie CSI from the channel block (``csi`` input), for benchmarks.
    """

    class Params(BlockParams):
        method: Literal["ls", "perfect"] = "ls"
        average_symbols: bool = Field(False, description="Average LS estimates over the frame.")

    inputs = {"in": Port(Kind.GRID), "csi": Port(Kind.CSI, required=False, doc="True channel (perfect mode).")}
    outputs = {"out": Port(Kind.CSI, doc="Frequency response (batch, n_ofdm_symbols, n_fft).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        grid = inputs["in"]
        lay: OFDMLayout = grid.meta["ofdm"]
        B, n_sym, n_fft = grid.data.shape
        if self.params.method == "perfect":
            H = self._perfect(inputs.get("csi"), lay, B, n_sym)
        else:
            if lay.pilot_freqs.size == 0:
                raise BlockError("LS estimation needs pilots (pilot_spacing > 0).")
            hp = grid.data[..., lay.bins(lay.pilot_freqs)] / lay.pilot_values
            if self.params.average_symbols:
                hp = np.broadcast_to(hp.mean(axis=1, keepdims=True), hp.shape)
            # Remove the linear phase of the dominant delay (timing offset inside the CP)
            # before interpolating, then restore it.
            pf = lay.pilot_freqs
            gaps = np.diff(pf)
            if gaps.size:
                step = np.bincount(gaps).argmax()
                pairs = np.nonzero(gaps == step)[0]
                slope = np.angle(np.sum(hp[..., pairs + 1] * np.conj(hp[..., pairs]), axis=-1)) / step
            else:
                slope = np.zeros(hp.shape[:-1])
            hp = hp * np.exp(-1j * slope[..., None] * pf)
            interp = self._interp_matrix(lay)
            H = np.ones((B, n_sym, n_fft), dtype=np.complex128)
            H[..., lay.bins(lay.active_freqs)] = (hp @ interp.T) * np.exp(1j * slope[..., None] * lay.active_freqs)
        return {"out": Signal(H, Kind.CSI, meta={**grid.meta, "csi_type": "freq"})}

    @staticmethod
    def _interp_matrix(lay: OFDMLayout) -> np.ndarray:
        """Linear interpolation pilots -> active subcarriers as a matrix (n_active, n_pilots)."""
        eye = np.eye(lay.pilot_freqs.size)
        return np.stack([np.interp(lay.active_freqs, lay.pilot_freqs, e) for e in eye], axis=1)

    @staticmethod
    def _perfect(csi: Signal | None, lay: OFDMLayout, B: int, n_sym: int) -> np.ndarray:
        if csi is None:
            raise BlockError("Perfect CSI needs the 'csi' input connected to the channel.")
        ctype = csi.meta.get("csi_type")
        if ctype == "cir":
            H = np.fft.fft(csi.data, n=lay.n_fft, axis=-1)[:, None, :]
        elif ctype == "cir_tv":  # impulse response sampled in the middle of every FFT window
            mid = np.arange(n_sym) * lay.symbol_len + lay.cp_len + lay.n_fft // 2
            cir = csi.data[..., np.clip(mid, 0, csi.data.shape[-1] - 1)]  # (B, L, n_sym)
            H = np.fft.fft(np.moveaxis(cir, -1, 1), n=lay.n_fft, axis=-1)
        elif ctype == "gain":
            h = csi.data
            if h.shape[-1] == 1:
                H = h[:, None, :]
            else:  # time-varying gain: average over each OFDM symbol
                usable = (h.shape[-1] // lay.symbol_len) * lay.symbol_len
                H = h[:, :usable].reshape(B, -1, lay.symbol_len).mean(axis=-1)[..., None]
        else:
            raise BlockError(f"Unsupported CSI type '{ctype}'.")
        return np.broadcast_to(H, (B, n_sym, lay.n_fft))


@register_block("ofdm_equalizer", category="equalizer", subcategory="Channel-Based", label="OFDM Equalizer")
class OFDMEqualizer(Block):
    """Extract the data subcarriers and equalize them (one-tap ZF per subcarrier).

    Without ``csi`` the data symbols are only extracted (AWGN channel). The
    per-symbol post-equalization noise variance is written to ``noise_var``.
    """

    inputs = {"in": Port(Kind.GRID), "csi": Port(Kind.CSI, required=False)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        grid = inputs["in"]
        lay: OFDMLayout = grid.meta["ofdm"]
        B = grid.data.shape[0]
        bins = lay.bins(lay.data_freqs)
        y = grid.data[..., bins]
        nv = grid.meta.get("noise_var")
        if "csi" in inputs:
            H = inputs["csi"].data[..., bins]
            y = y / H
            if nv is not None:
                nv = np.broadcast_to(nv / np.abs(H) ** 2, y.shape).reshape(B, -1)
        n = grid.meta.get("n_data_symbols", y.shape[1] * y.shape[2])
        out = y.reshape(B, -1)[:, :n]
        if isinstance(nv, np.ndarray):
            nv = nv[:, :n]
        meta = {**grid.meta, "noise_var": nv, "sps": 1, "n_symbols": n,
                "power": grid.meta.get("data_power", 1.0)}
        return {"out": Signal(out, Kind.SYMBOLS, meta=meta)}
