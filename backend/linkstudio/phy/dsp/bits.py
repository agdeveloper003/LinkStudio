"""Bit-level helpers shared by coding and modulation."""

from __future__ import annotations

import numpy as np


def bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    """Group the last axis in words of ``k`` bits (MSB first): ``(..., n*k) -> (..., n)``."""
    if bits.shape[-1] % k:
        raise ValueError(f"Number of bits ({bits.shape[-1]}) is not a multiple of {k}.")
    words = bits.reshape(*bits.shape[:-1], -1, k).astype(np.int64)
    weights = 1 << np.arange(k - 1, -1, -1, dtype=np.int64)
    return words @ weights


def ints_to_bits(ints: np.ndarray, k: int) -> np.ndarray:
    """Expand integers into ``k`` bits each (MSB first): ``(..., n) -> (..., n*k)``."""
    shifts = np.arange(k - 1, -1, -1)
    bits = (np.asarray(ints)[..., None] >> shifts) & 1
    return bits.reshape(*bits.shape[:-2], -1).astype(np.uint8)


def gray_encode(n: np.ndarray | int) -> np.ndarray | int:
    """Binary-reflected Gray code."""
    return n ^ (n >> 1)


def hard_decision(llr: np.ndarray) -> np.ndarray:
    """LLR -> bits with the convention LLR = log P(0)/P(1) (negative means 1)."""
    return (llr < 0).astype(np.uint8)


def bits_to_llr(bits: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
    """Map hard bits to LLRs of fixed magnitude (0 -> +m, 1 -> -m)."""
    return magnitude * (1.0 - 2.0 * bits.astype(np.float64))
