"""Additive white Gaussian noise channel."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block


@register_block("awgn", category="channel")
class AWGN(Block):
    """Add circularly-symmetric complex Gaussian noise.

    ``snr_db`` is interpreted according to ``mode``:

    * ``snr``  - per-sample SNR: N0 = P / SNR
    * ``esn0`` - energy per symbol over N0: N0 = P * sps / (Es/N0)
    * ``ebn0`` - energy per information bit: N0 = P * sps / (Eb/N0 * k * R * Ns)
    * ``thermal`` - absolute receiver noise: N0 = k_B T F (W/Hz) over the sampled bandwidth,
      noise variance N0 fs per complex sample; the signal must be in watts (link budgets).

    (Ns = ``spatial_rate`` metadata: streams per channel use with spatial multiplexing.)

    where P is the reference signal power: ``signal_power`` if given, otherwise the
    nominal power tracked in the ``power`` metadata (``power_reference="nominal"``,
    the textbook definition of average SNR) or the power measured over the batch
    (``"measured"``). ``sps`` / ``k`` (bits per symbol) / ``R`` (code rate) come from
    the signal metadata unless overridden. The noise variance per complex sample (N0)
    is published in the ``noise_var`` metadata for soft demappers.
    """

    class Params(BlockParams):
        snr_db: float = Field(10.0, description="SNR value in dB (meaning set by 'mode').")
        mode: Literal["snr", "esn0", "ebn0", "thermal"] = Field("ebn0", description="SNR definition.")
        power_reference: Literal["nominal", "measured"] = Field(
            "nominal", description="Use the nominal power from metadata or measure it.")
        signal_power: float | None = Field(None, gt=0, description="Explicit reference power.")
        bits_per_symbol: int | None = Field(None, ge=1, description="Override metadata.")
        code_rate: float | None = Field(None, gt=0, le=1, description="Override metadata.")
        noise_figure_db: float = Field(5.0, ge=0, description="Receiver noise figure (thermal mode).")
        temperature_k: float = Field(290.0, gt=0, description="Reference temperature (thermal mode).")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def noise_variance(self, x: Signal) -> float:
        p = self.params
        if p.mode == "thermal":
            if not x.fs:
                raise BlockError("Thermal noise needs the sample rate (fs) of the signal.")
            return 1.380649e-23 * p.temperature_k * 10 ** (p.noise_figure_db / 10) * x.fs
        if p.signal_power is not None:
            power = p.signal_power
        elif p.power_reference == "nominal" and x.meta.get("power") is not None:
            power = float(x.meta["power"])
        else:
            power = float(np.mean(np.abs(x.data) ** 2))
        snr = 10 ** (p.snr_db / 10)
        if p.mode == "snr":
            return power / snr
        sps = x.meta.get("sps", 1)
        if p.mode == "esn0":
            return power * sps / snr
        k = p.bits_per_symbol or x.meta.get("bits_per_symbol")
        if k is None:
            raise BlockError("Eb/N0 mode needs bits_per_symbol (mapper upstream or parameter).")
        rate = p.code_rate or x.meta.get("code_rate", 1.0)
        # Spatial multiplexing carries several symbols per channel use.
        return power * sps / (snr * k * rate * x.meta.get("spatial_rate", 1))

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        n0 = self.noise_variance(x)
        if np.iscomplexobj(x.data):
            noise = self.rng.standard_normal((*x.data.shape, 2)).view(np.complex128)[..., 0]
        else:  # real baseband (wireline, IM/DD): N0/2 per real dimension
            noise = self.rng.standard_normal(x.data.shape)
        y = x.data + np.sqrt(n0 / 2) * noise
        return {"out": x.evolve(y, noise_var=x.meta.get("noise_var", 0.0) + n0)}
