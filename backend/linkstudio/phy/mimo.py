"""Multi-antenna (MIMO) blocks for flat-fading single-carrier links.

Multi-stream signals have shape ``(batch, streams, n)``. The MIMO channel normalizes
its matrix by ``1/sqrt(Nt)`` so the total transmit power is independent of the number
of antennas and every receive antenna sees unit average power; the AWGN SNR is
therefore the SNR per receive antenna.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.channel.fading import complex_normal
from linkstudio.phy.dsp.constellations import Constellation


def _as_streams(x: np.ndarray) -> np.ndarray:
    return x[:, None, :] if x.ndim == 2 else x


def _exp_correlation(n: int, rho: float) -> np.ndarray:
    """Exponential correlation matrix R[i, j] = rho^|i-j| and its Hermitian square root."""
    idx = np.arange(n)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    w, v = np.linalg.eigh(R)
    return (v * np.sqrt(np.maximum(w, 0))) @ v.conj().T


@register_block("stream_demux", category="demultiplexer", label="Stream Demultiplexer")
class StreamDemux(Block):
    """Spatial multiplexing: distribute symbols round-robin over ``n_streams`` antennas."""

    class Params(BlockParams):
        n_streams: int = Field(2, ge=1, le=16, description="Number of transmit streams Nt.")

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS, doc="(batch, Nt, n / Nt)")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        nt = self.params.n_streams
        if x.length % nt:
            raise BlockError(f"Frame length {x.length} is not a multiple of {nt} streams.")
        out = x.data.reshape(x.data.shape[0], -1, nt).transpose(0, 2, 1)
        return {"out": x.evolve(np.ascontiguousarray(out), spatial_rate=nt, n_symbols=out.shape[-1])}


@register_block("stream_mux", category="multiplexer", label="Stream Multiplexer")
class StreamMux(Block):
    """Inverse of ``stream_demux``: interleave the streams back into one sequence."""

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        data = _as_streams(x.data)
        B = data.shape[0]
        out = data.transpose(0, 2, 1).reshape(B, -1)
        nv = x.meta.get("noise_var")
        if isinstance(nv, np.ndarray) and nv.shape == data.shape:
            nv = nv.transpose(0, 2, 1).reshape(B, -1)
        return {"out": x.evolve(out, noise_var=nv, n_symbols=out.shape[-1], spatial_rate=1)}


@register_block("mimo_channel", category="medium", subcategory="Wireless", label="MIMO Channel")
class MIMOChannel(Block):
    """Flat Rayleigh MIMO channel y = H x, H of size Nr x Nt, normalized by 1/sqrt(Nt).

    * ``block``: one matrix per frame; ``fast``: a new matrix every symbol.
    * Optional exponential antenna correlation (Kronecker model) at TX and RX.

    The true matrix is published on ``csi`` (``csi_type="mimo"``), shape
    ``(batch, Nr, Nt)`` for block fading or ``(batch, n, Nr, Nt)`` for fast fading.
    """

    class Params(BlockParams):
        n_rx: int = Field(2, ge=1, le=16, description="Receive antennas Nr.")
        variation: Literal["block", "fast"] = "block"
        tx_correlation: float = Field(0.0, ge=0, lt=1, description="Exponential correlation at TX.")
        rx_correlation: float = Field(0.0, ge=0, lt=1, description="Exponential correlation at RX.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in"), "csi": Port(Kind.CSI)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        data = _as_streams(x.data)
        B, nt, n = data.shape
        nr = self.params.n_rx
        shape = (B, nr, nt) if self.params.variation == "block" else (B, n, nr, nt)
        H = complex_normal(self.rng, shape)
        if self.params.tx_correlation or self.params.rx_correlation:
            H = _exp_correlation(nr, self.params.rx_correlation) @ H @ _exp_correlation(nt, self.params.tx_correlation)
        H = H / np.sqrt(nt)
        if self.params.variation == "block":
            y = H @ data
        else:
            y = np.einsum("bnrt,btn->brn", H, data)
        power = x.meta.get("power", 1.0)
        csi = Signal(H, Kind.CSI, fs=x.fs, meta={**x.meta, "csi_type": "mimo"})
        return {"out": x.evolve(y, power=power), "csi": csi}


def _per_symbol_h(H: np.ndarray, n: int) -> np.ndarray:
    """Return H with shape (batch, n, Nr, Nt)."""
    return np.broadcast_to(H[:, None], (H.shape[0], n, *H.shape[1:])) if H.ndim == 3 else H


@register_block("mimo_detector", category="detector", label="MIMO Detector")
class MIMODetector(Block):
    """Linear (ZF, MMSE) or maximum-likelihood MIMO detection with known channel.

    Linear detectors output unbiased per-stream symbols and the per-stream
    post-detection noise variance (``noise_var``) for soft demapping. ML performs an
    exhaustive search (M^Nt candidates) and outputs hard symbol decisions.
    With a single transmit stream ZF/MMSE reduce to maximum-ratio combining.
    """

    class Params(BlockParams):
        method: Literal["zf", "mmse", "ml"] = "mmse"

    inputs = {"in": Port(Kind.SYMBOLS), "csi": Port(Kind.CSI)}
    outputs = {"out": Port(Kind.SYMBOLS, doc="(batch, Nt, n)")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, csi = inputs["in"], inputs["csi"]
        if csi.meta.get("csi_type") != "mimo":
            raise BlockError("mimo_detector needs the csi of a mimo_channel.")
        y = _as_streams(x.data)  # (B, Nr, n)
        B, nr, n = y.shape
        H = _per_symbol_h(csi.data, n)  # (B, n, Nr, Nt)
        nt = H.shape[-1]
        n0 = float(np.mean(x.meta.get("noise_var", 0.0)))
        yv = y.transpose(0, 2, 1)[..., None]  # (B, n, Nr, 1)
        Hh = np.conj(np.swapaxes(H, -1, -2))
        if self.params.method == "ml":
            const: Constellation | None = x.meta.get("constellation")
            if const is None:
                raise BlockError("ML detection needs the constellation (mapper upstream).")
            if const.order ** nt > 4096:
                raise BlockError("ML search space too large (M^Nt > 4096).")
            cands = np.array(list(itertools.product(const.points, repeat=nt))).T  # (Nt, C)
            dist = np.sum(np.abs(yv - H @ cands) ** 2, axis=-2)  # (B, n, C)
            s = cands[:, np.argmin(dist, axis=-1)]  # (Nt, B, n)
            return {"out": x.evolve(np.moveaxis(s, 0, 1), noise_var=max(n0, 1e-12), spatial_rate=1)}
        gram = Hh @ H
        reg = n0 if self.params.method == "mmse" else 0.0
        inv = np.linalg.inv(gram + reg * np.eye(nt))
        s = (inv @ Hh @ yv)[..., 0]  # (B, n, Nt)
        if self.params.method == "zf":
            nv = n0 * np.real(np.diagonal(inv, axis1=-2, axis2=-1))
        else:
            bias = np.real(np.diagonal(inv @ gram, axis1=-2, axis2=-1))  # (W H)_kk
            s = s / bias
            nv = (1 - bias) / np.maximum(bias, 1e-12)
        return {"out": x.evolve(s.transpose(0, 2, 1), noise_var=nv.transpose(0, 2, 1))}


@register_block("alamouti_encoder", category="encoder", subcategory="Space-Time Codes", label="Alamouti Encoder")
class AlamoutiEncoder(Block):
    """Alamouti space-time block code: 2 transmit antennas, rate 1, full diversity."""

    inputs = {"in": Port(Kind.SYMBOLS)}
    outputs = {"out": Port(Kind.SYMBOLS, doc="(batch, 2, n)")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        if x.length % 2:
            raise BlockError("Alamouti needs an even number of symbols.")
        s1, s2 = x.data[:, 0::2], x.data[:, 1::2]
        out = np.empty((x.data.shape[0], 2, x.length), dtype=np.complex128)
        out[:, 0, 0::2], out[:, 0, 1::2] = s1, -np.conj(s2)
        out[:, 1, 0::2], out[:, 1, 1::2] = s2, np.conj(s1)
        return {"out": x.evolve(out, spatial_rate=1)}


@register_block("alamouti_decoder", category="decoder", subcategory="Space-Time Codes", label="Alamouti Decoder")
class AlamoutiDecoder(Block):
    """Linear Alamouti combining with known (block-fading) channel and any number of RX antennas."""

    inputs = {"in": Port(Kind.SYMBOLS), "csi": Port(Kind.CSI)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, csi = inputs["in"], inputs["csi"]
        H = csi.data
        if csi.meta.get("csi_type") != "mimo" or H.ndim != 3 or H.shape[-1] != 2:
            raise BlockError("alamouti_decoder needs a 2-TX block-fading mimo_channel.")
        y = _as_streams(x.data)
        r1, r2 = y[:, :, 0::2], y[:, :, 1::2]  # (B, Nr, n/2)
        h1, h2 = H[:, :, 0:1], H[:, :, 1:2]  # (B, Nr, 1)
        norm = np.sum(np.abs(h1) ** 2 + np.abs(h2) ** 2, axis=1)  # (B, 1)
        s1 = np.sum(np.conj(h1) * r1 + h2 * np.conj(r2), axis=1) / norm
        s2 = np.sum(np.conj(h2) * r1 - h1 * np.conj(r2), axis=1) / norm
        out = np.empty((y.shape[0], y.shape[-1]), dtype=np.complex128)
        out[:, 0::2], out[:, 1::2] = s1, s2
        nv = np.broadcast_to(float(np.mean(x.meta.get("noise_var", 0.0))) / norm, out.shape)
        return {"out": x.evolve(out, noise_var=nv)}
