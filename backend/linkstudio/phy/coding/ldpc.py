"""Quasi-cyclic LDPC codes with belief-propagation decoding.

Construction (IRA / repeat-accumulate style, as in IEEE 802.11n/5G base graphs):

    H = [ H_info | H_parity ]

* ``H_info``: ``m_blocks x k_blocks`` array of ``z x z`` circulant permutations,
  ``col_weight`` non-zero blocks per column with random shifts (seeded);
* ``H_parity``: block dual-diagonal (identity on the diagonal and sub-diagonal),
  which makes H full rank and encoding a simple accumulation.

Rate = k_blocks / (k_blocks + m_blocks); n = (k_blocks + m_blocks) * z.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import scipy.sparse as sp
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.bits import bits_to_llr


class QCLDPCCode:
    def __init__(self, k_blocks: int, m_blocks: int, z: int, col_weight: int = 3, seed: int = 1) -> None:
        if col_weight > m_blocks:
            raise ValueError("col_weight cannot exceed m_blocks.")
        self.z, self.k, self.m = z, k_blocks * z, m_blocks * z
        self.n = self.k + self.m
        rng = np.random.default_rng(seed)
        rows, cols = [], []
        eye = np.arange(z)
        for cb in range(k_blocks):  # information part
            for rb in rng.choice(m_blocks, col_weight, replace=False):
                shift = rng.integers(z)
                rows.append(rb * z + eye)
                cols.append(cb * z + (eye + shift) % z)
        for rb in range(m_blocks):  # dual-diagonal parity part
            rows.append(rb * z + eye)
            cols.append(self.k + rb * z + eye)
            if rb:
                rows.append(rb * z + eye)
                cols.append(self.k + (rb - 1) * z + eye)
        r, c = np.concatenate(rows), np.concatenate(cols)
        order = np.lexsort((c, r))
        self.check_of_edge, self.var_of_edge = r[order], c[order]
        E = r.size
        self.H = sp.csr_matrix((np.ones(E, np.int32), (r, c)), shape=(self.m, self.n))
        self.H_info = self.H[:, :self.k]
        # Edge <-> node incidence matrices for vectorized message passing.
        self.edge_to_var = sp.csr_matrix((np.ones(E), (np.arange(E), self.var_of_edge)), shape=(E, self.n))
        self.edge_to_check = sp.csr_matrix((np.ones(E), (np.arange(E), self.check_of_edge)), shape=(E, self.m))
        # Padded table: edges of every check node (row weights may differ).
        counts = np.bincount(self.check_of_edge, minlength=self.m)
        dmax = counts.max()
        self.check_table = np.full((self.m, dmax), -1, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.arange(E) - starts[self.check_of_edge]
        self.check_table[self.check_of_edge, pos] = np.arange(E)

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, u: np.ndarray) -> np.ndarray:
        """Information bits ``(B, k)`` -> systematic codewords ``(B, n)``."""
        s = (self.H_info @ u.T.astype(np.int32)) % 2  # (m, B)
        blocks = s.reshape(self.m // self.z, self.z, -1).astype(np.uint8)
        parity = np.bitwise_xor.accumulate(blocks, axis=0).reshape(self.m, -1)
        return np.concatenate([u, parity.T.astype(np.uint8)], axis=1)

    def syndrome_ok(self, bits: np.ndarray) -> np.ndarray:
        return np.all(((self.H @ bits.T.astype(np.int32)) % 2) == 0, axis=0)

    def decode(self, llr: np.ndarray, max_iter: int = 30, method: str = "min_sum",
               alpha: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        """Flooding belief propagation. Returns (information bits, iterations per frame)."""
        B = llr.shape[0]
        valid = self.check_table >= 0
        table = np.where(valid, self.check_table, 0)
        v2c = llr[:, self.var_of_edge]
        c2v = np.zeros_like(v2c)
        iters = np.full(B, max_iter)
        done = np.zeros(B, dtype=bool)
        bits = (llr < 0).astype(np.uint8)
        for it in range(1, max_iter + 1):
            V = v2c[:, table]  # (B, m, dmax)
            if method == "min_sum":
                sign = np.where(V < 0, -1.0, 1.0)
                sign[:, ~valid] = 1.0
                mag = np.abs(V)
                mag[:, ~valid] = np.inf
                i1 = np.argmin(mag, axis=-1)[..., None]
                min1 = np.take_along_axis(mag, i1, -1)
                np.put_along_axis(mag, i1, np.inf, -1)
                min2 = mag.min(axis=-1, keepdims=True)
                pos = np.arange(V.shape[-1])
                out_mag = np.where(pos == i1, min2, min1) * alpha
                out = np.prod(sign, axis=-1, keepdims=True) * sign * out_mag
            else:  # sum-product (tanh rule)
                T = np.tanh(np.clip(V, -30, 30) / 2)
                T[:, ~valid] = 1.0
                T = np.where(np.abs(T) < 1e-12, 1e-12, T)
                ratio = np.clip(np.prod(T, axis=-1, keepdims=True) / T, -0.999999999, 0.999999999)
                out = 2 * np.arctanh(ratio)
            c2v[:, table[valid]] = out[:, valid]
            total = llr + (self.edge_to_var.T @ c2v.T).T
            new_bits = (total < 0).astype(np.uint8)
            bits = np.where(done[:, None], bits, new_bits)
            ok = self.syndrome_ok(bits) & ~done
            iters[ok] = it
            done |= ok
            if done.all():
                break
            v2c = total[:, self.var_of_edge] - c2v
        return bits[:, :self.k], iters


@lru_cache(maxsize=16)
def get_code(k_blocks: int, m_blocks: int, z: int, col_weight: int, seed: int) -> QCLDPCCode:
    return QCLDPCCode(k_blocks, m_blocks, z, col_weight, seed)


class _LDPCParams(BlockParams):
    k_blocks: int = Field(8, ge=1, description="Information block columns.")
    m_blocks: int = Field(8, ge=1, description="Parity block columns (rate = k/(k+m)).")
    z: int = Field(48, ge=2, description="Lifting (circulant) size.")
    col_weight: int = Field(3, ge=1, description="Non-zero blocks per information column.")
    seed: int = Field(1, description="Seed of the code construction.")


@register_block("ldpc_encoder", category="coding")
class LDPCEncoder(Block):
    """Systematic quasi-cyclic LDPC encoder (frame length must be a multiple of k)."""

    Params = _LDPCParams
    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        p = self.params
        self.code = get_code(p.k_blocks, p.m_blocks, p.z, p.col_weight, p.seed)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        k = self.code.k
        if x.length % k:
            raise BlockError(f"Frame length {x.length} is not a multiple of k={k}.")
        B = x.data.shape[0]
        cw = self.code.encode(x.data.reshape(-1, k)).reshape(B, -1)
        return {"out": x.evolve(cw, code_rate=x.meta.get("code_rate", 1.0) * self.code.rate)}


@register_block("ldpc_decoder", category="coding")
class LDPCDecoder(Block):
    """Belief-propagation LDPC decoder (normalized min-sum or sum-product) with early stopping."""

    class Params(_LDPCParams):
        max_iter: int = Field(30, ge=1, description="Maximum BP iterations.")
        method: Literal["min_sum", "sum_product"] = "min_sum"
        alpha: float = Field(0.8, gt=0, le=1, description="Min-sum normalization factor.")

    inputs = {"in": Port(Kind.LLR | Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        p = self.params
        self.code = get_code(p.k_blocks, p.m_blocks, p.z, p.col_weight, p.seed)

    def on_reset(self) -> None:
        self._iters = 0
        self._words = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        llr = x.data if x.kind == Kind.LLR else bits_to_llr(x.data, 2.0)
        n = self.code.n
        if llr.shape[-1] % n:
            raise BlockError(f"Frame length {llr.shape[-1]} is not a multiple of n={n}.")
        B = llr.shape[0]
        bits, iters = self.code.decode(llr.reshape(-1, n), self.params.max_iter, self.params.method,
                                       self.params.alpha)
        self._iters += int(iters.sum())
        self._words += iters.size
        return {"out": Signal(bits.reshape(B, -1), Kind.BITS, meta=x.meta)}

    def metrics(self) -> dict[str, float]:
        return {"avg_iterations": self._iters / self._words if self._words else float("nan")}
