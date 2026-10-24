"""Bit sources."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.lfsr import PRBS_TAPS, lfsr_sequence


@register_block("random_bits", category="source")
class RandomBits(Block):
    """Independent Bernoulli bits, a new random frame for every batch row."""

    class Params(BlockParams):
        n_bits: int = Field(1024, ge=1, description="Bits per frame.")
        p_one: float = Field(0.5, ge=0.0, le=1.0, description="Probability of a one.")

    outputs = {"out": Port(Kind.BITS, doc="Bits (batch, n_bits).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        shape = (ctx.batch_size, self.params.n_bits)
        if self.params.p_one == 0.5:
            bits = self.rng.integers(0, 2, size=shape, dtype=np.uint8)
        else:
            bits = (self.rng.random(shape) < self.params.p_one).astype(np.uint8)
        return {"out": Signal(bits, Kind.BITS)}


@register_block("prbs", category="source")
class PRBS(Block):
    """Pseudo-random binary sequence (PRBS-7/9/11/15/20/23/31), continuous across frames."""

    class Params(BlockParams):
        order: Literal[7, 9, 11, 15, 20, 23, 31] = Field(9, description="PRBS order (period 2^order - 1).")
        n_bits: int = Field(1024, ge=1, description="Bits per frame.")

    outputs = {"out": Port(Kind.BITS, doc="Bits (batch, n_bits).")}

    def on_reset(self) -> None:
        self._state: np.ndarray | None = None

    def setup(self) -> None:
        self._state = None

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        n = ctx.batch_size * self.params.n_bits
        seq, self._state = lfsr_sequence(PRBS_TAPS[self.params.order], n, self._state)
        return {"out": Signal(seq.reshape(ctx.batch_size, -1), Kind.BITS)}
