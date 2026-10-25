"""Additive (synchronous) scrambler."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.lfsr import int_to_state, lfsr_sequence


@register_block("scrambler", category="bit_processing", label="Scrambler")
class Scrambler(Block):
    """XOR every frame with the same LFSR sequence.

    The operation is its own inverse: use the same block (same parameters) at the
    receiver. On LLR inputs the descrambling flips the sign of the affected LLRs.
    Default polynomial ``x^7 + x^4 + 1`` (IEEE 802.11).
    """

    class Params(BlockParams):
        taps: list[int] = Field([7, 4], description="Feedback polynomial exponents.")
        seed: int = Field(0b1011101, ge=1, description="Initial register state.")

    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port.like("in")}

    def setup(self) -> None:
        self._state = int_to_state(self.params.seed, max(self.params.taps))
        self._cache: dict[int, np.ndarray] = {}

    def sequence(self, n: int) -> np.ndarray:
        if n not in self._cache:
            self._cache[n] = lfsr_sequence(self.params.taps, n, self._state)[0]
        return self._cache[n]

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        seq = self.sequence(x.length)
        if x.kind == Kind.LLR:
            return {"out": x.evolve(x.data * (1.0 - 2.0 * seq))}
        return {"out": x.evolve(x.data ^ seq)}
