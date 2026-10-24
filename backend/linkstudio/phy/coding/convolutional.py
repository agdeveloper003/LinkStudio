"""Rate-1/n convolutional codes with a soft/hard-decision Viterbi decoder."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.bits import bits_to_llr


class Trellis:
    """Trellis of a feed-forward rate-1/n convolutional code.

    Args:
        generators: Generator polynomials in octal notation, e.g. ``(171, 133)``
            for the industry-standard K=7 code. The MSB taps the current input bit.
        constraint_length: K (memory K-1).

    State convention: the state holds the previous K-1 input bits, the most
    recent one in the MSB.
    """

    def __init__(self, generators: tuple[int, ...], constraint_length: int) -> None:
        if constraint_length < 2:
            raise ValueError("Constraint length must be >= 2.")
        self.K = constraint_length
        self.n = len(generators)
        self.n_states = 1 << (self.K - 1)
        self.gens = [int(str(g), 8) for g in generators]
        if any(g >= 1 << self.K for g in self.gens):
            raise ValueError(f"Generators {generators} do not fit constraint length {self.K}.")
        # Generator taps as arrays, index i multiplies u[t - i].
        self.taps = np.array([[(g >> (self.K - 1 - i)) & 1 for i in range(self.K)] for g in self.gens],
                             dtype=np.uint8)
        s = np.arange(self.n_states)
        self.output = np.zeros((self.n_states, 2, self.n), dtype=np.uint8)
        for u in (0, 1):
            reg = (u << (self.K - 1)) | s
            for j, g in enumerate(self.gens):
                self.output[:, u, j] = np.array([bin(r & g).count("1") & 1 for r in reg])
        # Predecessors of every next state and the input bit leading to it.
        ns = np.arange(self.n_states)
        self.prev0 = (ns << 1) & (self.n_states - 1)
        self.prev1 = self.prev0 | 1
        self.input_bit = ns >> (self.K - 2)
        self.sign0 = 1.0 - 2.0 * self.output[self.prev0, self.input_bit]  # (S, n)
        self.sign1 = 1.0 - 2.0 * self.output[self.prev1, self.input_bit]

    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Bits ``(B, k)`` -> coded bits ``(B, (k + tail) * n)`` interleaved per time step."""
        if terminate:
            bits = np.concatenate([bits, np.zeros((*bits.shape[:-1], self.K - 1), np.uint8)], axis=-1)
        T = bits.shape[-1]
        out = np.zeros((*bits.shape[:-1], T, self.n), dtype=np.uint8)
        for i in range(self.K):  # c_j[t] = sum_i taps[j, i] * u[t - i]  (mod 2)
            shifted = np.zeros_like(bits)
            shifted[..., i:] = bits[..., :T - i]
            out ^= shifted[..., None] * self.taps[:, i]
        return out.reshape(*bits.shape[:-1], -1)

    def decode(self, llr: np.ndarray, terminated: bool = True) -> np.ndarray:
        """Viterbi decoding of LLRs ``(B, T*n)`` (LLR > 0 means bit 0)."""
        B = llr.shape[0]
        T = llr.shape[-1] // self.n
        llr = llr[:, :T * self.n].reshape(B, T, self.n)
        S = self.n_states
        pm = np.full((B, S), -np.inf)
        pm[:, 0] = 0.0
        decisions = np.empty((T, B, S), dtype=bool)
        for t in range(T):
            bm0 = 0.5 * llr[:, t, :] @ self.sign0.T  # (B, S) correlation metrics
            bm1 = 0.5 * llr[:, t, :] @ self.sign1.T
            c0 = pm[:, self.prev0] + bm0
            c1 = pm[:, self.prev1] + bm1
            d = c1 > c0
            decisions[t] = d
            pm = np.where(d, c1, c0)
            pm -= pm.max(axis=1, keepdims=True)
        state = np.zeros(B, dtype=np.int64) if terminated else np.argmax(pm, axis=1)
        bits = np.empty((B, T), dtype=np.uint8)
        rows = np.arange(B)
        for t in range(T - 1, -1, -1):
            bits[:, t] = self.input_bit[state]
            state = np.where(decisions[t, rows, state], self.prev1[state], self.prev0[state])
        return bits[:, :T - (self.K - 1)] if terminated else bits


class _ConvParams(BlockParams):
    generators: tuple[int, ...] = Field((171, 133), description="Octal generator polynomials.")
    constraint_length: int = Field(7, ge=2, le=16, description="Constraint length K.")
    terminate: bool = Field(True, description="Append K-1 zero tail bits (zero-terminated trellis).")


@register_block("conv_encoder", category="coding")
class ConvEncoder(Block):
    """Feed-forward convolutional encoder of rate 1/n."""

    Params = _ConvParams
    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        self.trellis = Trellis(self.params.generators, self.params.constraint_length)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        out = self.trellis.encode(x.data, self.params.terminate)
        rate = x.meta.get("code_rate", 1.0) * x.length / out.shape[-1]
        return {"out": x.evolve(out, code_rate=rate)}


@register_block("viterbi_decoder", category="coding")
class ViterbiDecoder(Block):
    """Maximum-likelihood sequence decoder; soft decisions with LLR input, hard with bits."""

    Params = _ConvParams
    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        self.trellis = Trellis(self.params.generators, self.params.constraint_length)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        llr = x.data if x.kind == Kind.LLR else bits_to_llr(x.data)
        flat = llr.reshape(-1, llr.shape[-1])
        bits = self.trellis.decode(flat, self.params.terminate)
        return {"out": Signal(bits.reshape(*llr.shape[:-1], -1), Kind.BITS, meta=x.meta)}
