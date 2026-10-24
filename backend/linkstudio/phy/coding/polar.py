"""Polar codes (Arikan) with successive-cancellation decoding.

Encoding convention: x = u G_N with G_N = F^{(x)n}, F = [[1, 0], [1, 1]] (no bit
reversal), so that for u = [u_a, u_b]: x = [enc(u_a) XOR enc(u_b), enc(u_b)].
Information positions are the K most reliable synthetic channels, ranked with the
Bhattacharyya-parameter recursion at a design SNR.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pydantic import Field, field_validator

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.bits import bits_to_llr


@lru_cache(maxsize=32)
def frozen_mask(n_log2: int, k: int, design_snr_db: float) -> np.ndarray:
    """Boolean mask, True for frozen positions."""
    z = reliability(n_log2, design_snr_db)
    mask = np.ones(1 << n_log2, dtype=bool)
    mask[np.argsort(z, kind="stable")[:k]] = False
    return mask


def reliability(n_log2: int, design_snr_db: float) -> np.ndarray:
    """Bhattacharyya parameter of every synthetic channel (lower = more reliable).

    A channel with parameter z splits into a worse one (2z - z^2) feeding the first
    half of the indices (u_a) and a better one (z^2) feeding the second half (u_b).
    """
    def rec(z: float, size: int) -> list[float]:
        if size == 1:
            return [z]
        return rec(2 * z - z * z, size // 2) + rec(z * z, size // 2)
    return np.array(rec(float(np.exp(-10 ** (design_snr_db / 10))), 1 << n_log2))


def polar_encode(u: np.ndarray) -> np.ndarray:
    """x = u G_N along the last axis (length must be a power of two)."""
    x = u.copy()
    n = x.shape[-1]
    half = 1
    while half < n:
        x = x.reshape(*x.shape[:-1], -1, 2, half)
        x[..., 0, :] ^= x[..., 1, :]
        x = x.reshape(*x.shape[:-3], n)
        half *= 2
    return x


def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def sc_decode(llr: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Successive-cancellation decoding. Returns the estimated u ``(B, N)``."""
    u_hat = np.zeros(llr.shape, dtype=np.uint8)

    def rec(L: np.ndarray, lo: int) -> np.ndarray:
        n = L.shape[-1]
        mask = frozen[lo:lo + n]
        if mask.all():
            return np.zeros(L.shape, dtype=np.uint8)
        if not mask.any():  # rate-1 node: direct hard decision
            x = (L < 0).astype(np.uint8)
            u_hat[:, lo:lo + n] = polar_encode(x)
            return x
        if n == 1:
            x = (L < 0).astype(np.uint8)
            u_hat[:, lo:lo + 1] = x
            return x
        h = n // 2
        L1, L2 = L[:, :h], L[:, h:]
        xa = rec(_f(L1, L2), lo)
        xb = rec(L2 + (1 - 2.0 * xa) * L1, lo + h)
        return np.concatenate([xa ^ xb, xb], axis=1)

    rec(llr, 0)
    return u_hat


class _PolarParams(BlockParams):
    block_length: int = Field(256, ge=2, description="Code length N (power of two).")
    k: int = Field(128, ge=1, description="Information bits per codeword.")
    design_snr_db: float = Field(0.0, description="Design SNR of the code construction.")

    @field_validator("block_length")
    @classmethod
    def _pow2(cls, v: int) -> int:
        if v & (v - 1):
            raise ValueError("block_length must be a power of two.")
        return v


class _PolarBase(Block):
    Params = _PolarParams

    def setup(self) -> None:
        p = self.params
        if p.k > p.block_length:
            raise BlockError("k cannot exceed block_length.")
        self.n_log2 = int(np.log2(p.block_length))
        self.frozen = frozen_mask(self.n_log2, p.k, p.design_snr_db)
        self.info = np.nonzero(~self.frozen)[0]


@register_block("polar_encoder", category="coding")
class PolarEncoder(_PolarBase):
    """Polar encoder (frame length must be a multiple of k)."""

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        if x.length % p.k:
            raise BlockError(f"Frame length {x.length} is not a multiple of k={p.k}.")
        B = x.data.shape[0]
        u = np.zeros((x.data.size // p.k, p.block_length), dtype=np.uint8)
        u[:, self.info] = x.data.reshape(-1, p.k)
        cw = polar_encode(u).reshape(B, -1)
        return {"out": x.evolve(cw, code_rate=x.meta.get("code_rate", 1.0) * p.k / p.block_length)}


@register_block("polar_decoder", category="coding")
class PolarDecoder(_PolarBase):
    """Successive-cancellation polar decoder (soft LLR or hard input)."""

    inputs = {"in": Port(Kind.LLR | Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        llr = x.data if x.kind == Kind.LLR else bits_to_llr(x.data, 2.0)
        if llr.shape[-1] % p.block_length:
            raise BlockError(f"Frame length {llr.shape[-1]} is not a multiple of N={p.block_length}.")
        B = llr.shape[0]
        u = sc_decode(llr.reshape(-1, p.block_length), self.frozen)
        return {"out": Signal(u[:, self.info].reshape(B, -1), Kind.BITS, meta=x.meta)}
