"""Puncturing (rate matching) for any mother code."""

from __future__ import annotations

import numpy as np
from pydantic import Field, field_validator

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block


class _PunctureParams(BlockParams):
    pattern: list[int] = Field([1, 1, 1, 0, 0, 1],
                               description="Keep (1) / drop (0) pattern, repeated over the frame. "
                                           "Default: rate 3/4 from a rate-1/2 code (IEEE 802.11).")

    @field_validator("pattern")
    @classmethod
    def _binary(cls, v: list[int]) -> list[int]:
        if not v or any(b not in (0, 1) for b in v) or not any(v):
            raise ValueError("pattern must be a non-empty list of 0/1 with at least one 1.")
        return v


def _keep_mask(pattern: list[int], n: int) -> np.ndarray:
    return np.resize(np.asarray(pattern, dtype=bool), n)


@register_block("puncturer", category="coding")
class Puncturer(Block):
    """Remove coded bits according to a periodic pattern (raises the code rate)."""

    Params = _PunctureParams
    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        keep = _keep_mask(self.params.pattern, x.length)
        out = x.data[..., keep]
        rate = x.meta.get("code_rate", 1.0) * x.length / out.shape[-1]
        return {"out": x.evolve(out, code_rate=rate, punctured_length=x.length)}


@register_block("depuncturer", category="coding")
class Depuncturer(Block):
    """Re-insert punctured positions as erasures (LLR = 0) before decoding."""

    Params = _PunctureParams
    inputs = {"in": Port(Kind.LLR | Kind.BITS)}
    outputs = {"out": Port(Kind.LLR)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        llr = x.data if x.kind == Kind.LLR else 1.0 - 2.0 * x.data
        m = llr.shape[-1]
        n = x.meta.get("punctured_length")
        if n is None:  # shortest mother length whose pattern keeps exactly m positions
            period = len(self.params.pattern)
            big = _keep_mask(self.params.pattern, (m // sum(self.params.pattern) + 1) * period)
            n = int(np.nonzero(np.cumsum(big) == m)[0][0]) + 1
        keep = _keep_mask(self.params.pattern, n)
        out = np.zeros((*llr.shape[:-1], n))
        out[..., keep] = llr
        return {"out": Signal(out, Kind.LLR, meta=x.meta)}
