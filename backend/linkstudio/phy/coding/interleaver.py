"""Bit interleavers (block and pseudo-random) and their inverses."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block


@register_block("interleaver", category="coding")
class Interleaver(Block):
    """Permute the bits (or LLRs) of every frame.

    * ``block``: write row by row into a ``rows x (n / rows)`` matrix, read column by column.
    * ``random``: pseudo-random permutation drawn from ``seed`` (same at TX and RX).

    Set ``inverse: true`` for the de-interleaver.
    """

    class Params(BlockParams):
        kind: Literal["block", "random"] = "random"
        rows: int = Field(16, ge=1, description="Rows of the block interleaver.")
        seed: int = Field(1234, description="Seed of the random permutation.")
        inverse: bool = Field(False, description="Apply the inverse permutation.")

    inputs = {"in": Port(Kind.BITS | Kind.LLR | Kind.SYMBOLS)}
    outputs = {"out": Port.like("in")}

    def setup(self) -> None:
        self._cache: dict[int, np.ndarray] = {}

    def permutation(self, n: int) -> np.ndarray:
        if n not in self._cache:
            p = self.params
            if p.kind == "random":
                perm = np.random.default_rng(p.seed).permutation(n)
            else:
                if n % p.rows:
                    raise BlockError(f"Frame length {n} is not a multiple of rows={p.rows}.")
                perm = np.arange(n).reshape(p.rows, -1).T.reshape(-1)
            self._cache[n] = np.argsort(perm) if p.inverse else perm
        return self._cache[n]

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        return {"out": x.evolve(x.data[..., self.permutation(x.length)])}
