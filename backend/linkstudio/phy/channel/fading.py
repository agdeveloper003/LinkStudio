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

# 3GPP TR 38.901 Tables 7.7.2-1..5: (normalized delays, powers in dB, LOS tap index or None).
# Delays are scaled by the wanted RMS delay spread (TR 38.901 eq. 7.7-1).
NR_TDL_PRESETS: dict[str, tuple[list[float], list[float], int | None]] = {
    "TDL-A": ([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242,
               2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586],
              [-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3,
               -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7], None),
    "TDL-B": ([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283,
               1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834],
              [0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3, -8.9, -9, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2,
               -9.8, -11.4, -14.9, -9.2, -11.3], None),
    "TDL-C": ([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285,
               1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523],
              [-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2,
               -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8], None),
    "TDL-D": ([0, 0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525],
              [-0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0, -27.7], 0),
    "TDL-E": ([0, 0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426, 3.7136, 5.4524,
               12.0034, 20.6519],
              [-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6, -22.3, -25.6, -20.2, -29.8,
               -29.2], 0),
}


def delay_kernel(delay: float, half_width: int = 8) -> tuple[int, np.ndarray]:
    """Band-limited fractional delay: Hann-windowed sinc taps, unit energy, causal part only.

    Returns the index of the first tap and the taps. An integer delay gives a single unit tap.
    """
    whole = int(np.floor(delay))
    frac = delay - whole
    if frac < 1e-9:
        return whole, np.ones(1)
    k = np.arange(-half_width + 1, half_width + 1)
    taps = np.sinc(k - frac) * (0.5 + 0.5 * np.cos(np.pi * (k - frac) / half_width))
    first = whole + k[0]
    if first < 0:
        taps, first = taps[-first:], 0
    return first, taps / np.linalg.norm(taps)


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """CN(0, 1) samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def sum_of_sinusoids(rng: np.random.Generator, batch: int, n: int, doppler_norm: float,
                     n_sinusoids: int = 16) -> np.ndarray:
    """Time-varying Rayleigh process with Jakes (Clarke) Doppler spectrum, E|h|^2 = 1.

    Zheng & Xiao (2003) sum-of-sinusoids generator: M sinusoids with arrival angles
    alpha_n = (2 pi n - pi + theta) / (4 M) and independent random phases. Unlike the
    original Jakes model it is wide-sense stationary, and its autocorrelation approaches
    J0(2 pi f_D tau) per realization with few sinusoids. ``doppler_norm`` is f_D / f_s.
    """
    m = n_sinusoids
    wd = 2 * np.pi * doppler_norm * np.arange(n)
    theta = rng.uniform(-np.pi, np.pi, (batch, 1))
    phi = rng.uniform(-np.pi, np.pi, (batch, 1))
    psi = rng.uniform(-np.pi, np.pi, (batch, m))
    alpha = (2 * np.pi * np.arange(1, m + 1) - np.pi + theta) / (4 * m)  # (batch, m)
    arg = wd[None, None, :] * np.cos(alpha)[:, :, None] + phi[:, :, None]
    xc = np.sum(np.cos(psi)[:, :, None] * np.cos(arg), axis=1)
    xs = np.sum(np.sin(psi)[:, :, None] * np.cos(arg), axis=1)
    return (xc + 1j * xs) * np.sqrt(2 / m)


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
    + ``powers_db``, or by a 3GPP ``preset``: EPA/EVA/ETU (TS 36.104) or TDL-A...TDL-E
    (TR 38.901, scaled to ``delay_spread_ns``); presets need fs. Fractional sample delays
    become band-limited (windowed-sinc) taps unless ``fractional_delays`` is off, in which
    case they are rounded. The LOS tap of TDL-D/E is a fixed-amplitude path with random
    phase (Doppler 0.7 f_D in ``jakes`` mode), giving the Ricean first tap of the standard.
    With ``fading=false`` the taps are deterministic (amplitude sqrt(power)), useful for
    static ISI experiments.

    ``variation="block"`` keeps the taps constant over a frame (``csi_type="cir"``,
    shape ``(batch, L)``); ``"jakes"`` gives every tap an independent Doppler process
    (``csi_type="cir_tv"``, shape ``(batch, L, n)``).
    Output length equals input length (the channel tail is truncated).
    """

    estimate_response = True

    class Params(BlockParams):
        preset: Literal["EPA", "EVA", "ETU", "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"] | None = Field(
            None, description="3GPP TDL model.")
        delay_spread_ns: float = Field(100.0, gt=0, description="RMS delay spread for TDL-A...E.")
        fractional_delays: bool = Field(True, description="Band-limited fractional taps (else round).")
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

    def tap_profile(self, fs: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Delays in samples (float), linear powers and a LOS flag per tap (powers normalized if asked)."""
        p = self.params
        los = None
        if p.preset in NR_TDL_PRESETS:
            norm_delays, powers_db, los = NR_TDL_PRESETS[p.preset]
            delays, unit = np.asarray(norm_delays) * p.delay_spread_ns * 1e-9, "s"
        elif p.preset:
            delays_ns, powers_db = TDL_PRESETS[p.preset]
            delays, unit = np.asarray(delays_ns) * 1e-9, "s"
        else:
            delays, powers_db, unit = np.asarray(p.delays, dtype=float), p.powers_db, p.delay_unit
        if unit == "s":
            if not fs:
                raise BlockError("Delays in seconds need the sample rate (fs) of the signal.")
            delays = delays * fs
        if np.any(delays < 0):
            raise BlockError("Tap delays must be non-negative.")
        if not p.fractional_delays:
            delays = np.round(delays)
        power = 10 ** (np.asarray(powers_db, dtype=float) / 10)
        if p.normalize:
            power = power / power.sum()
        is_los = np.zeros(power.size, dtype=bool)
        if los is not None:
            is_los[los] = True
        return delays, power, is_los

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        B, n = x.data.shape[0], x.length
        delays, power, is_los = self.tap_profile(x.fs)
        kernels = [delay_kernel(d) for d in delays]
        length = max(first + taps.size for first, taps in kernels)
        jakes = p.variation == "jakes" and p.fading
        if jakes and not x.fs:
            raise BlockError("Time-varying multipath needs the sample rate (fs).")
        gains = []
        for pw, los in zip(power, is_los):
            phase = self.rng.uniform(0, 2 * np.pi, (B, 1))
            if los:
                g = np.exp(1j * phase) * np.ones((B, n if jakes else 1))
                if jakes:
                    g = g * np.exp(2j * np.pi * 0.7 * p.doppler_hz / x.fs * np.arange(n))
            elif not p.fading:
                g = np.ones((B, 1), dtype=np.complex128)
            elif jakes:
                g = sum_of_sinusoids(self.rng, B, n, p.doppler_hz / x.fs)
            else:
                g = complex_normal(self.rng, (B, 1))
            gains.append(g * np.sqrt(pw))
        meta = {"power": x.meta["power"] * power.sum()} if x.meta.get("power") is not None else {}
        if jakes:
            cir = np.zeros((B, length, n), dtype=np.complex128)
            y = np.zeros_like(x.data, dtype=np.complex128)
            for (first, taps), g in zip(kernels, gains):
                k = np.zeros(first + taps.size)
                k[first:] = taps
                y += g * fftconvolve(x.data, k[None, :], mode="full", axes=-1)[..., :n]
                cir[:, first:first + taps.size] += g[:, None, :] * taps[None, :, None]
            csi = Signal(cir, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "cir_tv"})
            return {"out": x.evolve(y, **meta), "csi": csi}
        cir = np.zeros((B, length), dtype=np.complex128)
        for (first, taps), g in zip(kernels, gains):
            cir[:, first:first + taps.size] += g * taps[None, :]
        y = fftconvolve(x.data, cir, mode="full", axes=-1)[..., :x.length]
        csi = Signal(cir, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "cir"})
        return {"out": x.evolve(y, **meta), "csi": csi}
