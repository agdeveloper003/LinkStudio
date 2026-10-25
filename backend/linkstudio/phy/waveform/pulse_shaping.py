"""Single-carrier waveform generation: pulse shaping, matched filtering, rate changes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.filters import design_pulse, fir_filter, upsample

PulseShape = Literal["rrc", "rc", "gaussian", "rect"]


@register_block("pulse_shaper", category="modulator", subcategory="Pulse Shaping", label="Pulse Shaper")
class PulseShaper(Block):
    """Upsample symbols by ``sps`` and filter them with a unit-energy pulse.

    Output length is ``n_symbols * sps + len(taps) - 1`` (full convolution); the
    nominal filter delay is recorded in the ``delay`` metadata for the matched filter.
    """

    class Params(BlockParams):
        shape: PulseShape = Field("rrc", description="Pulse shape.")
        sps: int = Field(8, ge=1, description="Samples per symbol.")
        span: int = Field(10, ge=1, description="Filter span in symbols.")
        rolloff: float = Field(0.25, ge=0.0, le=1.0, description="Roll-off of RRC/RC pulses.")
        bt: float = Field(0.3, gt=0.0, description="BT product of the Gaussian pulse.")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def setup(self) -> None:
        p = self.params
        self.taps = design_pulse(p.shape, p.sps, p.span, p.rolloff, p.bt)

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray | None:
        """DTFT of the pulse taps at the output sample rate (sps x symbol rate)."""
        if not x.fs:
            return None
        n = np.arange(len(self.taps))
        h = np.exp(-2j * np.pi * np.multiply.outer(np.asarray(f) / (x.fs * self.params.sps), n)) @ self.taps
        return h / (np.abs(h[np.argmin(np.abs(f))]) or 1.0)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        sps = self.params.sps
        y = fir_filter(upsample(x.data, sps), self.taps)
        return {"out": Signal(y, Kind.SAMPLES, fs=x.fs * sps if x.fs else None, meta={
            **x.meta, "sps": sps, "pulse_taps": self.taps, "n_symbols": x.length,
            "power": (x.meta.get("power") or 1.0) / sps,  # unit-energy pulse spreads Es over sps samples
            "delay": x.meta.get("delay", 0.0) + (len(self.taps) - 1) / 2})}


@register_block("matched_filter", category="demodulator", subcategory="Matched Filtering", label="Matched Filter")
class MatchedFilter(Block):
    """Receive filter + symbol-rate sampling.

    ``shape="matched"`` uses the time-reversed conjugate of the transmit pulse
    (``pulse_taps`` metadata). The output is scaled so that the overall pulse peak
    is 1 (symbols come out at their nominal amplitude) and ``noise_var`` metadata
    is updated accordingly. Sampling instants follow the known filter delays; there
    is no timing recovery, so timing offsets show up as ISI.
    """

    class Params(BlockParams):
        shape: Literal["matched", "rrc", "rc", "gaussian", "rect", "none"] = "matched"
        sps: int | None = Field(None, ge=1, description="Samples per symbol (default: from metadata).")
        span: int = Field(10, ge=1)
        rolloff: float = Field(0.25, ge=0.0, le=1.0)
        bt: float = Field(0.3, gt=0.0)
        normalize: bool = Field(True, description="Scale so that the end-to-end pulse peak is 1.")
        decimate: bool = Field(True, description="Sample at symbol rate (false: keep the filtered "
                                                 "waveform for timing_recovery).")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SYMBOLS | Kind.SAMPLES, doc="Symbols, or samples when decimate=false.")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        sps = p.sps or x.meta.get("sps")
        if not sps:
            raise BlockError("Unknown samples per symbol: set 'sps' or use a pulse shaper upstream.")
        tx_taps = x.meta.get("pulse_taps")
        if p.shape == "matched":
            if tx_taps is None:
                raise BlockError("'matched' needs the transmit pulse (pulse_shaper upstream).")
            rx_taps = np.conj(tx_taps[::-1])
        elif p.shape == "none":
            rx_taps = np.ones(1)
        else:
            rx_taps = design_pulse(p.shape, sps, p.span, p.rolloff, p.bt)

        y = fir_filter(x.data, rx_taps)
        n_sym = x.meta.get("n_symbols", x.length // sps)
        start = int(round(x.meta.get("delay", 0.0) + (len(rx_taps) - 1) / 2))

        noise_gain = float(np.sum(np.abs(rx_taps) ** 2))
        if p.normalize and tx_taps is not None:
            overall = np.convolve(tx_taps, rx_taps)
            peak = overall[int(round((len(tx_taps) - 1) / 2 + (len(rx_taps) - 1) / 2))]
            y = y / peak
            noise_gain /= abs(peak) ** 2
        noise = {"noise_var": x.meta["noise_var"] * noise_gain} if x.meta.get("noise_var") is not None else {}

        if not p.decimate:
            meta = {**x.meta, "delay": float(start), "pulse_taps": None, "n_symbols": n_sym, **noise}
            return {"out": Signal(y, Kind.SAMPLES, fs=x.fs, meta=meta)}

        idx = start + np.arange(n_sym) * sps
        if idx[-1] >= y.shape[-1]:
            raise BlockError("Waveform too short for the requested number of symbols.")
        z = y[..., idx]
        meta = {**x.meta, "sps": 1, "delay": 0.0, "pulse_taps": None, "n_symbols": n_sym,
                "power": (x.meta.get("power") or 1.0 / sps) * sps, **noise}
        return {"out": Signal(z, Kind.SYMBOLS, fs=x.fs / sps if x.fs else None, meta=meta)}


@register_block("upsampler", category="filter", label="Upsampler")
class Upsampler(Block):
    """Insert ``factor - 1`` zeros between samples."""

    class Params(BlockParams):
        factor: int = Field(2, ge=1)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        f = self.params.factor
        return {"out": Signal(upsample(x.data, f), Kind.SAMPLES, fs=x.fs * f if x.fs else None,
                              meta={**x.meta, "sps": x.meta.get("sps", 1) * f,
                                    "power": (x.meta.get("power") or 1.0) / f})}


@register_block("downsampler", category="filter", label="Downsampler")
class Downsampler(Block):
    """Keep one sample out of ``factor`` starting at ``offset``.

    The ``sps`` and ``delay`` metadata are rescaled, so a matched-filtered waveform can be
    brought from a high simulation rate down to e.g. 2 samples/symbol for an equalizer.
    """

    class Params(BlockParams):
        factor: int = Field(2, ge=1)
        offset: int = Field(0, ge=0)
        output_kind: Literal["samples", "symbols"] = "samples"

    inputs = {"in": Port(Kind.SAMPLES | Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SAMPLES | Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        f = self.params.factor
        kind = Kind.SYMBOLS if self.params.output_kind == "symbols" else Kind.SAMPLES
        sps = max(1, x.meta.get("sps", f) // f)
        delay = (x.meta.get("delay", 0.0) - self.params.offset) / f
        return {"out": Signal(x.data[..., self.params.offset::f], kind, fs=x.fs / f if x.fs else None,
                              meta={**x.meta, "sps": sps, "delay": delay, "pulse_taps": None})}


@register_block("fir_filter", category="filter", label="FIR Filter")
class FIRFilter(Block):
    """Generic FIR filter with user-supplied taps (causal, output length preserved)."""

    class Params(BlockParams):
        taps: list[float] = Field([1.0], description="Real filter coefficients.")

    inputs = {"in": Port(Kind.SAMPLES | Kind.SYMBOLS)}
    outputs = {"out": Port.like("in")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        taps = np.asarray(self.params.taps)
        return {"out": x.evolve(fir_filter(x.data, taps, mode="causal"),
                                delay=x.meta.get("delay", 0.0) + (len(taps) - 1) / 2)}
