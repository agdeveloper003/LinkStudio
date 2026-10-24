"""Fading channels: flat Rayleigh/Rician and frequency-selective tapped delay lines."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.signal import fftconvolve

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block

# 3GPP TS 36.104 Annex B extended models: (delays in ns, relative powers in dB).
TDL_PRESETS: dict[str, tuple[list[float], list[float]]] = {
    "EPA": ([0, 30, 70, 90, 110, 190, 410], [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]),
    "EVA": ([0, 30, 150, 310, 370, 710, 1090, 1730, 2510],
            [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]),
    "ETU": ([0, 50, 120, 200, 230, 500, 1600, 2300, 5000],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]),
}


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """CN(0, 1) samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def sum_of_sinusoids(rng: np.random.Generator, batch: int, n: int, doppler_norm: float,
                     n_sinusoids: int = 16) -> np.ndarray:
    """Time-varying Rayleigh process with Jakes (Clarke) Doppler spectrum.

    Random-phase sum-of-sinusoids model; ``doppler_norm`` is f_D / f_s.
    """
    t = np.arange(n)
    h = np.zeros((batch, n), dtype=np.complex128)
    alpha = rng.uniform(0, 2 * np.pi, (batch, n_sinusoids))
    phi = rng.uniform(0, 2 * np.pi, (batch, n_sinusoids))
    for i in range(n_sinusoids):
        h += np.exp(1j * (2 * np.pi * doppler_norm * np.cos(alpha[:, i, None]) * t + phi[:, i, None]))
    return h / np.sqrt(n_sinusoids)


@register_block("flat_fading", category="channel")
class FlatFading(Block):
    """Frequency-flat Rayleigh or Rician fading, E|h|^2 = 1.

    * ``block``: one coefficient per frame (quasi-static).
    * ``fast``: independent coefficient per sample.
    * ``jakes``: time-correlated process with maximum Doppler ``doppler_hz`` (needs fs).

    The ``csi`` output carries the true gains (``csi_type="gain"``) for coherent
    equalization or benchmarking with perfect channel knowledge.
    """

    class Params(BlockParams):
        model: Literal["rayleigh", "rician"] = "rayleigh"
        k_factor_db: float = Field(0.0, description="Rician K factor (LOS / scattered power).")
        variation: Literal["block", "fast", "jakes"] = "block"
        doppler_hz: float = Field(100.0, ge=0, description="Maximum Doppler shift (jakes).")
        n_sinusoids: int = Field(16, ge=4, description="Sinusoids of the Jakes generator.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in"), "csi": Port(Kind.CSI, doc="Channel gains (batch, 1 or n).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        B, n = x.data.shape[0], x.length
        if p.variation == "block":
            h = complex_normal(self.rng, (B, 1))
        elif p.variation == "fast":
            h = complex_normal(self.rng, (B, n))
        else:
            if not x.fs:
                raise BlockError("Jakes fading needs the sample rate (fs) of the signal.")
            h = sum_of_sinusoids(self.rng, B, n, p.doppler_hz / x.fs, p.n_sinusoids)
        if p.model == "rician":
            k = 10 ** (p.k_factor_db / 10)
            los = np.exp(1j * self.rng.uniform(0, 2 * np.pi, (B, 1)))
            h = np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * h
        csi = Signal(h, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "gain"})
        return {"out": x.evolve(x.data * h), "csi": csi}


@register_block("multipath", category="channel")
class MultipathChannel(Block):
    """Tapped-delay-line channel (frequency selective).

    Taps are given either as ``delays`` (samples, or seconds with ``delay_unit="s"``)
    + ``powers_db``, or by a 3GPP ``preset`` (EPA/EVA/ETU, needs fs). Delays are
    rounded to the sample grid. With ``fading=false`` the taps are deterministic
    (amplitude sqrt(power)), useful for static ISI experiments.

    ``variation="block"`` keeps the taps constant over a frame (``csi_type="cir"``,
    shape ``(batch, L)``); ``"jakes"`` gives every tap an independent Doppler process
    (``csi_type="cir_tv"``, shape ``(batch, L, n)``).
    Output length equals input length (the channel tail is truncated).
    """

    class Params(BlockParams):
        preset: Literal["EPA", "EVA", "ETU"] | None = Field(None, description="3GPP TDL model.")
        delays: list[float] = Field([0.0, 1.0, 2.0], description="Tap delays.")
        powers_db: list[float] = Field([0.0, -3.0, -6.0], description="Tap powers in dB.")
        delay_unit: Literal["samples", "s"] = "samples"
        fading: bool = Field(True, description="Rayleigh taps (else static real taps).")
        variation: Literal["block", "jakes"] = Field("block", description="Tap time variation.")
        doppler_hz: float = Field(50.0, ge=0, description="Maximum Doppler shift (jakes).")
        normalize: bool = Field(True, description="Normalize total power to 1.")

        @model_validator(mode="after")
        def _check(self) -> "MultipathChannel.Params":
            if self.preset is None and len(self.delays) != len(self.powers_db):
                raise ValueError("delays and powers_db must have the same length.")
            return self

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in"), "csi": Port(Kind.CSI, doc="Channel impulse response (batch, L).")}

    def tap_profile(self, fs: float | None) -> tuple[np.ndarray, np.ndarray]:
        """Integer sample delays and linear powers (taps falling on the same sample are merged)."""
        p = self.params
        if p.preset:
            delays_ns, powers_db = TDL_PRESETS[p.preset]
            delays, unit = np.asarray(delays_ns) * 1e-9, "s"
        else:
            delays, powers_db, unit = np.asarray(p.delays), p.powers_db, p.delay_unit
        if unit == "s":
            if not fs:
                raise BlockError("Delays in seconds need the sample rate (fs) of the signal.")
            delays = delays * fs
        idx = np.round(delays).astype(int)
        if np.any(idx < 0):
            raise BlockError("Tap delays must be non-negative.")
        power = np.zeros(idx.max() + 1)
        np.add.at(power, idx, 10 ** (np.asarray(powers_db) / 10))
        if p.normalize:
            power /= power.sum()
        return np.nonzero(power)[0], power[power > 0]

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        B, n = x.data.shape[0], x.length
        delays, power = self.tap_profile(x.fs)
        if self.params.variation == "jakes" and self.params.fading:
            if not x.fs:
                raise BlockError("Time-varying multipath needs the sample rate (fs).")
            cir = np.zeros((B, delays.max() + 1, n), dtype=np.complex128)
            y = np.zeros_like(x.data, dtype=np.complex128)
            for d, pw in zip(delays, power):
                h = sum_of_sinusoids(self.rng, B, n, self.params.doppler_hz / x.fs) * np.sqrt(pw)
                cir[:, d] = h
                y[..., d:] += h[..., d:] * x.data[..., :n - d]
            csi = Signal(cir, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "cir_tv"})
            meta = {"power": x.meta["power"] * power.sum()} if x.meta.get("power") is not None else {}
            return {"out": x.evolve(y, **meta), "csi": csi}
        cir = np.zeros((B, delays.max() + 1), dtype=np.complex128)
        if self.params.fading:
            cir[:, delays] = complex_normal(self.rng, (B, delays.size)) * np.sqrt(power)
        else:
            cir[:, delays] = np.sqrt(power)
        y = fftconvolve(x.data, cir, mode="full", axes=-1)[..., :x.length]
        csi = Signal(cir, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "cir"})
        meta = {"power": x.meta["power"] * power.sum()} if x.meta.get("power") is not None else {}
        return {"out": x.evolve(y, **meta), "csi": csi}
