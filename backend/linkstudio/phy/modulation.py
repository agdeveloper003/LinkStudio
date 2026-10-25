"""Digital modulation: bits <-> constellation symbols."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation, make_constellation

Scheme = Literal["psk", "qam", "pam", "custom"]


@register_block("mapper", category="mapper", label="Constellation Mapper")
class Mapper(Block):
    """Map groups of bits onto Gray-labelled constellation points (unit average energy)."""

    class Params(BlockParams):
        scheme: Scheme = Field("qam", description="Modulation family.")
        order: int = Field(4, ge=2, description="Constellation size M (power of two).")
        phase_offset_deg: float | None = Field(None, description="PSK rotation (default: 45 deg for QPSK).")
        points: list[tuple[float, float]] | None = Field(None, description="Custom points (I, Q), by label.")
        symbol_rate: float = Field(1e6, gt=0, description="Symbol rate in Hz.")

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def setup(self) -> None:
        p = self.params
        self.constellation = make_constellation(p.scheme, p.order, p.phase_offset_deg, p.points)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        k = self.constellation.bits_per_symbol
        if x.length % k:
            raise BlockError(f"Frame length {x.length} is not a multiple of {k} bits/symbol "
                             f"({self.constellation.name}).")
        symbols = self.constellation.map(x.data)
        meta = {**x.meta, "bits_per_symbol": k, "constellation": self.constellation,
                "n_symbols": symbols.shape[-1], "sps": 1, "power": 1.0}
        return {"out": Signal(symbols, Kind.SYMBOLS, fs=self.params.symbol_rate, meta=meta)}


@register_block("demapper", category="demapper", label="Constellation Demapper")
class Demapper(Block):
    """Hard or soft (LLR) demapping.

    When ``scheme`` is not set, the constellation is taken from the upstream mapper
    (``constellation`` metadata). Soft demapping needs the noise variance, read from
    the ``noise_var`` metadata (set by the AWGN block and updated by equalizers)
    unless given explicitly.
    """

    class Params(BlockParams):
        method: Literal["hard", "maxlog", "exact"] = Field("maxlog", description="Decision type.")
        scheme: Scheme | None = Field(None, description="Override the upstream constellation.")
        order: int = Field(4, ge=2)
        phase_offset_deg: float | None = None
        points: list[tuple[float, float]] | None = None
        noise_var: float | None = Field(None, gt=0, description="Override the noise variance.")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.BITS | Kind.LLR, doc="Bits (hard) or LLRs (soft).")}

    def setup(self) -> None:
        p = self.params
        self._constellation = (make_constellation(p.scheme, p.order, p.phase_offset_deg, p.points)
                               if p.scheme else None)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        const: Constellation | None = self._constellation or x.meta.get("constellation")
        if const is None:
            raise BlockError("No constellation: set 'scheme' or place a mapper upstream.")
        if self.params.method == "hard":
            return {"out": Signal(const.demap_hard(x.data), Kind.BITS, meta=x.meta)}
        nv = self.params.noise_var if self.params.noise_var is not None else x.meta.get("noise_var")
        if nv is None:
            raise BlockError("Soft demapping needs a noise variance (AWGN upstream or 'noise_var').")
        llr = const.demap_soft(x.data, nv, self.params.method)
        return {"out": Signal(llr, Kind.LLR, meta=x.meta)}
