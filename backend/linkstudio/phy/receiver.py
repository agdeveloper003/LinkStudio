"""Single-carrier equalizers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation


def _gains_at_symbols(csi: Signal, n_symbols: int) -> np.ndarray:
    """Channel gains aligned with symbol-rate samples.

    Block-fading gains (one per frame) broadcast; per-sample gains of an
    oversampled waveform are picked at the symbol instants (nominal delay + k*sps).
    """
    h = csi.data
    if h.shape[-1] == 1:
        return h
    sps = csi.meta.get("sps", 1)
    if sps > 1:
        idx = int(round(csi.meta.get("delay", 0.0))) + np.arange(n_symbols) * sps
        return h[..., np.clip(idx, 0, h.shape[-1] - 1)]
    if h.shape[-1] != n_symbols:
        raise BlockError(f"CSI length {h.shape[-1]} does not match {n_symbols} symbols.")
    return h


@register_block("one_tap_equalizer", category="equalizer", subcategory="Channel-Based", label="One-Tap Equalizer")
class OneTapEqualizer(Block):
    """Coherent equalization of flat fading with known gains (``csi`` input).

    * ``zf``: y / h; the noise variance becomes N0 / |h|^2.
    * ``phase``: y * exp(-j angle(h)); amplitude untouched (enough for PSK).
    """

    class Params(BlockParams):
        method: Literal["zf", "phase"] = "zf"

    inputs = {"in": Port(Kind.SYMBOLS), "csi": Port(Kind.CSI)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, csi = inputs["in"], inputs["csi"]
        if csi.meta.get("csi_type") != "gain":
            raise BlockError("one_tap_equalizer needs flat-fading gains; use linear_equalizer for multipath.")
        h = _gains_at_symbols(csi, x.length)
        nv = x.meta.get("noise_var")
        if self.params.method == "zf":
            y = x.data / h
            if nv is not None:
                nv = np.broadcast_to(nv / np.abs(h) ** 2, y.shape)
        else:
            y = x.data * np.exp(-1j * np.angle(h))
        return {"out": x.evolve(y, noise_var=nv)}


@register_block("linear_equalizer", category="equalizer", subcategory="Channel-Based", label="Linear Equalizer")
class LinearEqualizer(Block):
    """FIR equalizer designed from the known channel impulse response (``csi`` input).

    Taps solve ``min ||H w - e_d||^2`` (ZF) or its regularized MMSE version, where
    ``H`` is the convolution matrix of the CIR; the decision delay ``d`` is either
    given or chosen per frame to minimize the MSE. The output is unbiased and
    ``noise_var`` is updated with the noise enhancement ``||w||^2``.
    """

    class Params(BlockParams):
        n_taps: int = Field(15, ge=1, description="Equalizer length.")
        method: Literal["zf", "mmse"] = "mmse"
        delay: int | None = Field(None, ge=0, description="Decision delay (default: optimal).")

    inputs = {"in": Port(Kind.SYMBOLS), "csi": Port(Kind.CSI)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, csi = inputs["in"], inputs["csi"]
        if csi.meta.get("csi_type") != "cir":
            raise BlockError("linear_equalizer needs a channel impulse response (multipath block).")
        h = csi.data  # (B, L)
        B, L = h.shape
        n = self.params.n_taps
        Lc = n + L - 1
        H = np.zeros((B, Lc, n), dtype=np.complex128)
        for j in range(n):
            H[:, j:j + L, j] = h
        nv = x.meta.get("noise_var", 0.0)
        lam = np.mean(nv) if self.params.method == "mmse" else 1e-9
        A = np.conj(np.swapaxes(H, 1, 2)) @ H + lam * np.eye(n)
        W = np.linalg.solve(A, np.conj(np.swapaxes(H, 1, 2)))  # (B, n, Lc): column d = taps for delay d
        if self.params.delay is not None:
            if self.params.delay >= Lc:
                raise BlockError(f"Delay must be < n_taps + L - 1 = {Lc}.")
            d = np.full(B, self.params.delay)
        else:
            gain = np.real(np.einsum("bij,bji->bi", H, W))  # (H W)[d, d]
            d = np.argmax(gain, axis=1)
        rows = np.arange(B)
        w = W[rows, :, d]  # (B, n)
        bias = np.einsum("bij,bj->bi", H, w)[rows, d]
        w = w / bias[:, None]
        full = np.stack([np.convolve(x.data[b], w[b]) for b in range(B)])
        idx = d[:, None] + np.arange(x.length)
        y = np.take_along_axis(np.pad(full, ((0, 0), (0, Lc))), idx, axis=1)
        new_nv = np.asarray(nv) * np.sum(np.abs(w) ** 2, axis=1)[:, None]
        return {"out": x.evolve(y, noise_var=new_nv)}


@register_block("lms_equalizer", category="equalizer", subcategory="Adaptive", label="LMS Equalizer")
class LMSEqualizer(Block):
    """Adaptive (N)LMS FIR equalizer: trained on known symbols, then decision directed.

    The ``ref`` input supplies the transmitted symbols; only the first ``n_train``
    of every frame are used for training. Decisions use the upstream constellation.
    The residual MSE measured after training is written to ``noise_var``.
    """

    class Params(BlockParams):
        n_taps: int = Field(11, ge=1)
        step_size: float = Field(0.1, gt=0, description="Adaptation step mu.")
        normalized: bool = Field(True, description="Use NLMS (step normalized by input power).")
        n_train: int = Field(200, ge=0, description="Training symbols per frame.")
        decision_directed: bool = Field(True, description="Keep adapting on decisions after training.")

    inputs = {"in": Port(Kind.SYMBOLS), "ref": Port(Kind.SYMBOLS, required=False, doc="Training symbols.")}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        ref = inputs.get("ref")
        p = self.params
        const: Constellation | None = x.meta.get("constellation")
        if ref is None and p.n_train > 0:
            raise BlockError("Training needs the 'ref' input (transmitted symbols).")
        if const is None and p.decision_directed:
            raise BlockError("Decision-directed mode needs the constellation (mapper upstream).")
        B, N = x.data.shape
        n, D = p.n_taps, p.n_taps // 2
        w = np.zeros((B, n), dtype=np.complex128)
        w[:, D] = 1.0
        ypad = np.concatenate([np.zeros((B, n - 1)), x.data, np.zeros((B, D))], axis=1)
        out = np.empty((B, N), dtype=np.complex128)
        err_power, err_count = 0.0, 0
        for t in range(N + D):
            u = ypad[:, t:t + n][:, ::-1]  # [y[t], y[t-1], ...]
            z = np.sum(np.conj(w) * u, axis=1)
            k = t - D
            if k < 0:
                continue
            out[:, k] = z
            if k < p.n_train:
                target = ref.data[:, k]
            elif p.decision_directed:
                target = const.points[const.nearest(z)]
            else:
                continue
            e = target - z
            if k >= p.n_train:
                err_power += float(np.sum(np.abs(e) ** 2))
                err_count += B
            mu = p.step_size / (1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True)) if p.normalized \
                else p.step_size
            w += mu * u * np.conj(e)[:, None]
        nv = err_power / err_count if err_count else x.meta.get("noise_var")
        return {"out": x.evolve(out, noise_var=nv)}
