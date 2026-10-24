"""Simple linear block codes: repetition and Hamming."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.bits import hard_decision


def _scale_rate(x: Signal, n_in: int, n_out: int) -> float:
    return x.meta.get("code_rate", 1.0) * n_in / n_out


# ----------------------------------------------------------------- repetition
@register_block("repetition_encoder", category="coding")
class RepetitionEncoder(Block):
    """Repeat every bit ``n`` times (rate 1/n)."""

    class Params(BlockParams):
        n: int = Field(3, ge=1, description="Repetition factor.")

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        out = np.repeat(x.data, self.params.n, axis=-1)
        return {"out": x.evolve(out, code_rate=_scale_rate(x, x.length, out.shape[-1]))}


@register_block("repetition_decoder", category="coding")
class RepetitionDecoder(Block):
    """Soft (LLR sum) or hard (majority vote) repetition decoding."""

    class Params(BlockParams):
        n: int = Field(3, ge=1, description="Repetition factor.")

    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        n = self.params.n
        groups = x.data.reshape(*x.data.shape[:-1], -1, n)
        if x.kind == Kind.LLR:
            bits = hard_decision(groups.sum(axis=-1))
        else:
            bits = (groups.sum(axis=-1) * 2 > n).astype(np.uint8)
        return {"out": Signal(bits, Kind.BITS, meta=x.meta)}


# -------------------------------------------------------------------- Hamming
class HammingCode:
    """Systematic Hamming code (2^m - 1, 2^m - 1 - m), single-error correcting."""

    def __init__(self, m: int) -> None:
        if m < 2:
            raise ValueError("Hamming code needs m >= 2.")
        self.m = m
        self.n = (1 << m) - 1
        self.k = self.n - m
        cols = np.arange(1, self.n + 1)
        weight_ge2 = cols[(cols & (cols - 1)) != 0]
        unit = 1 << np.arange(m - 1, -1, -1)
        order = np.concatenate([weight_ge2, unit])  # parity part last -> systematic
        self.H = ((order[None, :] >> np.arange(m - 1, -1, -1)[:, None]) & 1).astype(np.int32)  # (m, n)
        p = self.H[:, :self.k].T  # (k, m)
        self.G = np.concatenate([np.eye(self.k, dtype=np.int32), p], axis=1)  # (k, n)
        # Syndrome value -> position of the single erroneous bit (-1: no error).
        self.syndrome_table = np.full(1 << m, -1, dtype=np.int64)
        syndromes = (self.H.T * (1 << np.arange(m - 1, -1, -1))).sum(axis=1)
        self.syndrome_table[syndromes] = np.arange(self.n)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        words = bits.reshape(*bits.shape[:-1], -1, self.k).astype(np.int32)
        return ((words @ self.G) & 1).astype(np.uint8).reshape(*bits.shape[:-1], -1)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        words = bits.reshape(*bits.shape[:-1], -1, self.n).astype(np.int32)
        syn = ((words @ self.H.T) & 1) @ (1 << np.arange(self.m - 1, -1, -1))
        pos = self.syndrome_table[syn]
        idx = np.nonzero(pos >= 0)
        words[(*idx, pos[idx])] ^= 1
        return words[..., :self.k].astype(np.uint8).reshape(*bits.shape[:-1], -1)


class _HammingParams(BlockParams):
    m: int = Field(3, ge=2, le=10, description="Parity bits; code is (2^m-1, 2^m-1-m).")


@register_block("hamming_encoder", category="coding")
class HammingEncoder(Block):
    """Systematic Hamming encoder (frame length must be a multiple of k)."""

    Params = _HammingParams
    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        self.code = HammingCode(self.params.m)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        if x.length % self.code.k:
            raise BlockError(f"Frame length {x.length} is not a multiple of k={self.code.k}.")
        out = self.code.encode(x.data)
        return {"out": x.evolve(out, code_rate=_scale_rate(x, x.length, out.shape[-1]))}


@register_block("hamming_decoder", category="coding")
class HammingDecoder(Block):
    """Syndrome decoding (hard decision; LLR inputs are sliced first)."""

    Params = _HammingParams
    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        self.code = HammingCode(self.params.m)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        bits = hard_decision(x.data) if x.kind == Kind.LLR else x.data
        if bits.shape[-1] % self.code.n:
            raise BlockError(f"Frame length {bits.shape[-1]} is not a multiple of n={self.code.n}.")
        return {"out": Signal(self.code.decode(bits), Kind.BITS, meta=x.meta)}
