"""RF front-end and propagation impairments."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.filters import fir_filter, fractional_delay_taps

_IO = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
_OUT = {"out": Port.like("in")}


def _require_fs(x: Signal, what: str) -> float:
    if not x.fs:
        raise BlockError(f"{what} needs the sample rate (fs) of the signal.")
    return x.fs


@register_block("gain", category="amplifier", subcategory="Electrical", label="Gain")
class Gain(Block):
    """Scale the signal by ``gain_db`` (e.g. path loss). Noise metadata is scaled too."""

    class Params(BlockParams):
        gain_db: float = 0.0

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        g = 10 ** (self.params.gain_db / 20)
        meta = {}
        for key in ("noise_var", "power"):
            if x.meta.get(key) is not None:
                meta[key] = x.meta[key] * g ** 2
        return {"out": x.evolve(x.data * g, **meta)}


@register_block("carrier_offset", category="impairment", subcategory="RF Front-End", label="Carrier Frequency Offset")
class CarrierOffset(Block):
    """Carrier frequency offset and phase rotation: y = x * exp(j(2 pi cfo t + phi))."""

    class Params(BlockParams):
        cfo_hz: float = Field(0.0, description="Frequency offset in Hz (needs fs).")
        phase_deg: float = Field(0.0, description="Static phase offset.")
        random_phase: bool = Field(False, description="Uniform random phase per frame.")

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        B, n = x.data.shape[0], x.length
        phase = np.full((B, 1), np.deg2rad(p.phase_deg))
        if p.random_phase:
            phase = phase + self.rng.uniform(0, 2 * np.pi, (B, 1))
        if p.cfo_hz:
            phase = phase + 2 * np.pi * p.cfo_hz / _require_fs(x, "A frequency offset") * np.arange(n)
        return {"out": x.evolve(x.data * np.exp(1j * phase))}


@register_block("phase_noise", category="impairment", subcategory="RF Front-End", label="Phase Noise")
class PhaseNoise(Block):
    """Oscillator phase noise modelled as a Wiener process with 3-dB ``linewidth_hz``."""

    class Params(BlockParams):
        linewidth_hz: float = Field(100.0, ge=0, description="Oscillator 3-dB linewidth.")

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "Phase noise")
        var = 2 * np.pi * self.params.linewidth_hz / fs
        phi = np.cumsum(self.rng.standard_normal(x.data.shape) * np.sqrt(var), axis=-1)
        return {"out": x.evolve(x.data * np.exp(1j * phi))}


@register_block("timing_offset", category="impairment", subcategory="Timing", label="Timing Offset")
class TimingOffset(Block):
    """Delay the signal by a (fractional) number of samples, keeping its length."""

    class Params(BlockParams):
        delay_samples: float = Field(0.0, ge=0, description="Delay in samples (can be fractional).")

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        d = self.params.delay_samples
        whole, frac = int(np.floor(d)), d - np.floor(d)
        y = x.data
        if frac > 1e-12:
            half = 16
            full = fir_filter(y, fractional_delay_taps(frac, half))
            y = full[..., half:half + x.length]
        if whole:
            y = np.concatenate([np.zeros((*y.shape[:-1], whole), y.dtype), y[..., :-whole]], axis=-1)
        return {"out": x.evolve(y)}


@register_block("iq_imbalance", category="impairment", subcategory="RF Front-End", label="IQ Imbalance")
class IQImbalance(Block):
    """Receiver IQ imbalance and DC offset: y = K1 x + K2 conj(x) + dc."""

    class Params(BlockParams):
        amplitude_db: float = Field(0.0, description="Gain mismatch between I and Q branches.")
        phase_deg: float = Field(0.0, description="Quadrature phase error.")
        dc_i: float = Field(0.0, description="DC offset on I.")
        dc_q: float = Field(0.0, description="DC offset on Q.")

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        g = 10 ** (p.amplitude_db / 20)
        phi = np.deg2rad(p.phase_deg)
        k1 = (1 + g * np.exp(-1j * phi)) / 2
        k2 = (1 - g * np.exp(1j * phi)) / 2
        return {"out": x.evolve(k1 * x.data + k2 * np.conj(x.data) + complex(p.dc_i, p.dc_q))}


@register_block("power_amplifier", category="impairment", subcategory="RF Front-End", label="Power Amplifier Nonlinearity")
class PowerAmplifier(Block):
    """Memoryless PA nonlinearity (Rapp, Saleh or ideal clipper).

    The input is scaled so that its mean power sits ``input_backoff_db`` below the
    saturation power (1), passed through the AM/AM (and AM/PM for Saleh) curves and,
    if ``normalize_output``, scaled back to the original level.
    """

    class Params(BlockParams):
        model: Literal["rapp", "saleh", "clip"] = "rapp"
        input_backoff_db: float = Field(6.0, description="Mean input power below saturation (dB).")
        smoothness: float = Field(2.0, gt=0, description="Rapp smoothness factor p.")
        saleh_beta_a: float = 1.1517
        saleh_alpha_p: float = 4.0033
        saleh_beta_p: float = 9.1040
        normalize_output: bool = True

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        p_in = float(np.mean(np.abs(x.data) ** 2))
        scale = np.sqrt(10 ** (-p.input_backoff_db / 10) / max(p_in, 1e-300))
        u = x.data * scale
        r, theta = np.abs(u), np.angle(u)
        if p.model == "rapp":
            a = r / (1 + r ** (2 * p.smoothness)) ** (1 / (2 * p.smoothness))
            y = a * np.exp(1j * theta)
        elif p.model == "saleh":
            # Divided by alpha_a so that the small-signal gain is 1, like the other models.
            a = r / (1 + p.saleh_beta_a * r ** 2)
            dphi = p.saleh_alpha_p * r ** 2 / (1 + p.saleh_beta_p * r ** 2)
            y = a * np.exp(1j * (theta + dphi))
        else:
            y = np.minimum(r, 1.0) * np.exp(1j * theta)
        if p.normalize_output:
            y = y / scale
        return {"out": x.evolve(y)}


@register_block("quantizer", category="impairment", subcategory="Data Converters", label="ADC Quantizer")
class Quantizer(Block):
    """Uniform mid-rise ADC on I and Q with clipping.

    ``full_scale`` is the clipping level; when unset it is ``loading_factor`` times
    the RMS value of each component (measured over the batch).
    """

    class Params(BlockParams):
        bits: int = Field(8, ge=1, le=24, description="ADC resolution.")
        full_scale: float | None = Field(None, gt=0, description="Clipping amplitude per component.")
        loading_factor: float = Field(4.0, gt=0, description="full_scale / rms when full_scale is unset.")

    inputs, outputs = _IO, _OUT

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = p.full_scale
        if fs is None:
            rms = np.sqrt(np.mean(np.abs(x.data) ** 2) / 2)
            fs = p.loading_factor * rms
        levels = 1 << p.bits
        step = 2 * fs / levels

        def q(v: np.ndarray) -> np.ndarray:
            idx = np.clip(np.floor(v / step), -levels // 2, levels // 2 - 1)
            return (idx + 0.5) * step

        return {"out": x.evolve(q(x.data.real) + 1j * q(x.data.imag))}
