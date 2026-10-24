"""Cyclic redundancy check: attach parity bits at the transmitter, verify at the receiver."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.bits import hard_decision

# Generator polynomials including the x^r term.
CRC_POLYNOMIALS: dict[str, int] = {
    "crc8": 0x107,  # x^8 + x^2 + x + 1
    "crc16": 0x11021,  # CRC-16-CCITT
    "crc24a": 0x1864CFB,  # LTE / NR CRC24A
    "crc32": 0x104C11DB7,  # IEEE 802.3
}


@lru_cache(maxsize=32)
def crc_parity_matrix(poly: int, k: int) -> np.ndarray:
    """Binary matrix ``P (k, r)`` such that ``parity = bits @ P mod 2``.

    CRC (with zero initial value, no reflection) is linear, so row ``i`` is the
    remainder of ``x^(k-1-i+r)`` modulo the generator.
    """
    r = poly.bit_length() - 1
    rems = np.empty(k, dtype=np.int64)
    rem = poly ^ (1 << r)  # x^r mod g
    for j in range(k):
        rems[k - 1 - j] = rem
        rem <<= 1
        if rem >> r & 1:
            rem ^= poly
    shifts = np.arange(r - 1, -1, -1)
    return ((rems[:, None] >> shifts) & 1).astype(np.int32)


def crc_parity(bits: np.ndarray, poly: int) -> np.ndarray:
    return ((bits.astype(np.int32) @ crc_parity_matrix(poly, bits.shape[-1])) & 1).astype(np.uint8)


class _CRCParams(BlockParams):
    polynomial: Literal["crc8", "crc16", "crc24a", "crc32"] = Field("crc16", description="CRC generator.")


@register_block("crc_encoder", category="coding")
class CRCEncoder(Block):
    """Append CRC parity bits to every frame."""

    Params = _CRCParams
    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS, doc="Payload followed by r parity bits.")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        parity = crc_parity(x.data, CRC_POLYNOMIALS[self.params.polynomial])
        out = np.concatenate([x.data, parity], axis=-1)
        rate = x.meta.get("code_rate", 1.0) * x.length / out.shape[-1]
        return {"out": x.evolve(out, code_rate=rate)}


@register_block("crc_decoder", category="coding")
class CRCDecoder(Block):
    """Check and strip the CRC; reports the frame error rate detected by the CRC."""

    Params = _CRCParams
    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port(Kind.BITS, doc="Payload bits."), "valid": Port(Kind.FLAGS, doc="CRC ok per frame.")}

    def on_reset(self) -> None:
        self._frames = 0
        self._failures = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        bits = hard_decision(x.data) if x.kind == Kind.LLR else x.data
        poly = CRC_POLYNOMIALS[self.params.polynomial]
        r = poly.bit_length() - 1
        payload, parity = bits[..., :-r], bits[..., -r:]
        valid = np.all(crc_parity(payload, poly) == parity, axis=-1)
        self._frames += valid.size
        self._failures += int(np.count_nonzero(~valid))
        return {"out": Signal(payload, Kind.BITS, meta=x.meta), "valid": Signal(valid, Kind.FLAGS)}

    def metrics(self) -> dict[str, float]:
        return {"frames": self._frames, "crc_failures": self._failures,
                "bler": self._failures / self._frames if self._frames else float("nan")}
