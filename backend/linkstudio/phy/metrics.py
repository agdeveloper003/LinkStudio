"""Measurement (sink) blocks. Their results are read through ``metrics()``."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.analysis import papr_db
from linkstudio.phy.dsp.bits import hard_decision


def wilson_interval(errors: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score confidence interval of an error probability."""
    if trials == 0:
        return float("nan"), float("nan")
    p = errors / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))


@register_block("ber_meter", category="measurement", label="BER Meter")
class BERMeter(Block):
    """Count bit, symbol and frame errors between transmitted and received bits.

    Metrics: ``ber`` with its 95 % confidence interval (``ber_ci_low`` / ``ber_ci_high``,
    Wilson score), ``bit_errors``, ``bits``, ``fer``, ``frame_errors``, ``frames`` and,
    when ``bits_per_symbol`` is set, ``ser``, ``symbol_errors``, ``symbols``.
    """

    class Params(BlockParams):
        bits_per_symbol: int | None = Field(None, ge=1, description="Also count symbol errors.")
        skip_bits: int = Field(0, ge=0, description="Ignore the first bits of every frame.")

    inputs = {"tx": Port(Kind.BITS, doc="Reference bits."), "rx": Port(Kind.BITS | Kind.LLR, doc="Decisions.")}
    outputs = {}

    def on_reset(self) -> None:
        self._bit_err = self._bits = self._frame_err = self._frames = 0
        self._sym_err = self._syms = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        tx = inputs["tx"].data
        rx = inputs["rx"]
        rx_bits = hard_decision(rx.data) if rx.kind == Kind.LLR else rx.data
        n = min(tx.shape[-1], rx_bits.shape[-1])
        s = self.params.skip_bits
        errors = tx[..., s:n] != rx_bits[..., s:n]
        self._bit_err += int(np.count_nonzero(errors))
        self._bits += errors.size
        frame_err = errors.reshape(errors.shape[0], -1).any(axis=1)
        self._frame_err += int(np.count_nonzero(frame_err))
        self._frames += frame_err.size
        k = self.params.bits_per_symbol
        if k:
            m = (errors.shape[-1] // k) * k
            sym_err = errors[..., :m].reshape(*errors.shape[:-1], -1, k).any(axis=-1)
            self._sym_err += int(np.count_nonzero(sym_err))
            self._syms += sym_err.size
        return {}

    def metrics(self) -> dict[str, float]:
        nan = float("nan")
        lo, hi = wilson_interval(self._bit_err, self._bits)
        out: dict[str, float] = {
            "ber": self._bit_err / self._bits if self._bits else nan,
            "ber_ci_low": lo,
            "ber_ci_high": hi,
            "bit_errors": self._bit_err,
            "bits": self._bits,
            "fer": self._frame_err / self._frames if self._frames else nan,
            "frame_errors": self._frame_err,
            "frames": self._frames,
        }
        if self.params.bits_per_symbol:
            out.update(ser=self._sym_err / self._syms if self._syms else nan,
                       symbol_errors=self._sym_err, symbols=self._syms)
        return out


@register_block("evm_meter", category="measurement", label="EVM Meter")
class EVMMeter(Block):
    """Error vector magnitude and modulation error ratio between reference and received symbols."""

    class Params(BlockParams):
        skip_symbols: int = Field(0, ge=0, description="Ignore the first symbols (e.g. equalizer training).")

    inputs = {"ref": Port(Kind.SYMBOLS), "in": Port(Kind.SYMBOLS)}
    outputs = {}

    def on_reset(self) -> None:
        self._err = self._ref = 0.0
        self._n = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        ref, meas = inputs["ref"].data, inputs["in"].data
        n = min(ref.shape[-1], meas.shape[-1])
        s = min(self.params.skip_symbols, n)
        self._err += float(np.sum(np.abs(meas[..., s:n] - ref[..., s:n]) ** 2))
        self._ref += float(np.sum(np.abs(ref[..., s:n]) ** 2))
        self._n += ref[..., s:n].size
        return {}

    def metrics(self) -> dict[str, float]:
        if not self._n:
            return {"evm_pct": float("nan"), "mer_db": float("nan")}
        ratio = self._err / self._ref
        return {"evm_pct": 100 * float(np.sqrt(ratio)), "mer_db": float(-10 * np.log10(max(ratio, 1e-300)))}


@register_block("power_meter", category="measurement", label="Power Meter")
class PowerMeter(Block):
    """Average power and peak-to-average power ratio of a waveform."""

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {}

    def on_reset(self) -> None:
        self._power = 0.0
        self._papr: list[float] = []
        self._count = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"].data
        self._power += float(np.sum(np.abs(x) ** 2))
        self._count += x.size
        self._papr.extend(papr_db(x).ravel().tolist())
        return {}

    def metrics(self) -> dict[str, float]:
        if not self._count:
            return {"power_db": float("nan"), "papr_db_mean": float("nan"), "papr_db_max": float("nan")}
        return {"power_db": 10 * float(np.log10(self._power / self._count)),
                "papr_db_mean": float(np.mean(self._papr)), "papr_db_max": float(np.max(self._papr))}
