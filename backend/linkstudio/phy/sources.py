"""Bit sources."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.lfsr import PRBS_TAPS, lfsr_sequence


@register_block("random_bits", category="source", subcategory="Bit Sources", label="Random Bits")
class RandomBits(Block):
    """Random bits, a new random frame for every batch row.

    Without ``transition_probability`` the bits are independent (Bernoulli ``p_one``).
    With it they form a two-state Markov chain: ``transition_probability`` is the share of
    bits that differ from the previous one, ``p_one`` the stationary share of ones, so
    P(0 -> 1) = t / (2 (1 - p_one)) and P(1 -> 0) = t / (2 p_one). Independent bits are the
    special case t = 2 p_one (1 - p_one); t must not exceed 2 min(p_one, 1 - p_one).
    """

    class Params(BlockParams):
        n_bits: int = Field(1024, ge=1, description="Bits per frame.")
        p_one: float = Field(0.5, ge=0.0, le=1.0, description="Probability of a one.")
        transition_probability: float | None = Field(
            None, ge=0.0, le=1.0, description="Probability that a bit differs from the previous one (None = independent bits).")

        @model_validator(mode="after")
        def _check_transitions(self) -> "RandomBits.Params":
            t = self.transition_probability
            if t is not None and t > 2 * min(self.p_one, 1 - self.p_one) + 1e-12:
                limit = 2 * min(self.p_one, 1 - self.p_one)
                raise ValueError(f"transition_probability must be at most {limit:g} with p_one = {self.p_one:g}.")
            return self

    outputs = {"out": Port(Kind.BITS, doc="Bits (batch, n_bits).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        p = self.params
        shape = (ctx.batch_size, p.n_bits)
        if p.transition_probability is not None and p.transition_probability != 2 * p.p_one * (1 - p.p_one):
            bits = markov_bits(self.rng, *shape, p.p_one, p.transition_probability)
        elif p.p_one == 0.5:
            bits = self.rng.integers(0, 2, size=shape, dtype=np.uint8)
        else:
            bits = (self.rng.random(shape) < p.p_one).astype(np.uint8)
        return {"out": Signal(bits, Kind.BITS)}


def markov_bits(rng: np.random.Generator, rows: int, n: int, p_one: float, t: float) -> np.ndarray:
    """Rows of a stationary two-state Markov chain with share of ones ``p_one`` and transition rate ``t``.

    Runs of zeros and ones have geometric lengths (exit probabilities P(0 -> 1), P(1 -> 0)),
    drawn in blocks and laid out alternately, so the cost does not grow with a Python loop per bit.
    """
    out = np.empty((rows, n), dtype=np.uint8)
    leave = (t / (2 * (1 - p_one)) if p_one < 1 else 1.0, t / (2 * p_one) if p_one > 0 else 1.0)
    runs_per_draw = int(n * t) // 2 + 16
    for r in range(rows):
        state = int(rng.random() < p_one)
        row: list[np.ndarray] = []
        filled = 0
        while filled < n:
            pairs = []
            for k in range(2):
                q = leave[(state + k) % 2]
                pairs.append(np.full(runs_per_draw, n, dtype=np.int64) if q <= 0 else rng.geometric(min(q, 1.0), runs_per_draw))
            lengths = np.column_stack(pairs).ravel()
            values = np.tile(np.array([state, 1 - state], dtype=np.uint8), runs_per_draw)
            row.append(np.repeat(values, lengths))
            filled += int(lengths.sum())
        out[r] = np.concatenate(row)[:n]
    return out


@register_block("prbs", category="source", subcategory="Bit Sources", label="PRBS Generator")
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
