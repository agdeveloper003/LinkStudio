"""Wireline (copper) communications.

* Line codes: NRZ, Manchester, AMI (``line_encoder`` / ``line_decoder``) and the
  block codes 4B/5B and 64B/66B (``line_block_encoder`` / ``line_block_decoder``).
* Channels: frequency-dependent cable loss (``copper_line``), NEXT/FEXT crosstalk,
  impulse noise, hybrid echo.
* SerDes receiver: CTLE, adaptive FFE + DFE, echo canceller, jitter injection.
* DSL: DMT modulation with water-filling bit loading.

Wireline signals are real-valued baseband; blocks keep real arrays real.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation, pam, psk, qam
from linkstudio.phy.dsp.filters import apply_frequency_response, cubic_resample

# ------------------------------------------------------------------ line codes
_FOUR_B_FIVE_B = ["11110", "01001", "10100", "10101", "01010", "01011", "01110", "01111",
                  "10010", "10011", "10110", "10111", "11010", "11011", "11100", "11101"]
FOUR_B_FIVE_B = np.array([[int(c) for c in w] for w in _FOUR_B_FIVE_B], dtype=np.uint8)  # (16, 5)


@register_block("line_encoder", category="mapper", label="Line Code Mapper (NRZ, AMI, MLT-3)")
class LineEncoder(Block):
    """Baseband line code: bits -> real symbols.

    * ``nrz``: 1 -> +1, 0 -> -1 (2-PAM),
    * ``manchester`` (IEEE 802.3): 0 -> (+1, -1), 1 -> (-1, +1); two chips per bit, DC-free,
    * ``ami``: 0 -> 0, 1 -> alternately +1 / -1 (ternary, DC-free),
    * ``mlt3`` (100BASE-TX): every 1 steps the level through 0, +1, 0, -1 (low bandwidth).
    """

    class Params(BlockParams):
        code: Literal["nrz", "manchester", "ami", "mlt3"] = "nrz"
        symbol_rate: float = Field(1e9, gt=0, description="Output symbol (chip) rate in Hz.")

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.SYMBOLS, doc="Real symbols.")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        b = x.data.astype(np.float64)
        code = self.params.code
        if code == "nrz":
            s, k, power, const = 2 * b - 1, 1.0, 1.0, pam(2)
        elif code == "manchester":
            s = np.stack([1 - 2 * b, 2 * b - 1], axis=-1).reshape(*b.shape[:-1], -1)
            k, power, const = 0.5, 1.0, None
        elif code == "ami":
            sign = np.where(np.cumsum(b, axis=-1) % 2 == 1, 1.0, -1.0)
            s, k, power, const = b * sign, 1.0, 0.5, None
        else:
            cycle = np.array([0.0, 1.0, 0.0, -1.0])
            s, k, power, const = cycle[np.cumsum(b, axis=-1).astype(int) % 4], 1.0, 0.5, None
        meta = {**x.meta, "bits_per_symbol": k, "power": power, "n_symbols": s.shape[-1], "sps": 1,
                "line_code": code, "constellation": const}
        return {"out": Signal(s, Kind.SYMBOLS, fs=self.params.symbol_rate, meta=meta)}


@register_block("line_decoder", category="demapper", label="Line Code Demapper (NRZ, AMI, MLT-3)")
class LineDecoder(Block):
    """Hard decisions for NRZ / Manchester / AMI / MLT-3 symbols (the real part is used)."""

    class Params(BlockParams):
        code: Literal["nrz", "manchester", "ami", "mlt3"] = "nrz"
        threshold: float = Field(0.5, gt=0, description="AMI / MLT-3 decision threshold on |y|.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        y = np.real(x.data)
        if self.params.code == "nrz":
            bits = y > 0
        elif self.params.code == "manchester":
            pairs = y[..., : (y.shape[-1] // 2) * 2].reshape(*y.shape[:-1], -1, 2)
            bits = pairs[..., 0] < pairs[..., 1]
        elif self.params.code == "ami":
            bits = np.abs(y) > self.params.threshold
        else:  # MLT-3: a 1 is a change of level
            level = np.where(y > self.params.threshold, 1, np.where(y < -self.params.threshold, -1, 0))
            prev = np.concatenate([np.zeros_like(level[..., :1]), level[..., :-1]], axis=-1)
            bits = level != prev
        return {"out": Signal(bits.astype(np.uint8), Kind.BITS, meta=x.meta)}


def _self_sync_scramble(bits: np.ndarray) -> np.ndarray:
    """64B/66B scrambler s[n] = d[n] ^ s[n-39] ^ s[n-58] (zero initial state)."""
    n = bits.shape[-1]
    s = np.zeros((*bits.shape[:-1], 58 + n), dtype=np.uint8)
    pos = 58
    while pos < 58 + n:
        m = min(39, 58 + n - pos)
        s[..., pos:pos + m] = bits[..., pos - 58:pos - 58 + m] ^ s[..., pos - 39:pos - 39 + m] ^ s[..., pos - 58:pos - 58 + m]
        pos += m
    return s[..., 58:]


def _self_sync_descramble(s: np.ndarray) -> np.ndarray:
    padded = np.concatenate([np.zeros((*s.shape[:-1], 58), dtype=np.uint8), s], axis=-1)
    n = s.shape[-1]
    return padded[..., 58:] ^ padded[..., 19:19 + n] ^ padded[..., :n]


@register_block("line_block_encoder", category="encoder", subcategory="Line Block Codes", label="Line Block Encoder (4B/5B, 64B/66B)")
class LineBlockEncoder(Block):
    """4B/5B (FDDI, 100BASE-TX) or 64B/66B (10G/100G Ethernet) block line code.

    64B/66B prepends the data sync header ``01`` to every 64-bit block and scrambles
    the payload with the self-synchronous polynomial x^58 + x^39 + 1.
    """

    class Params(BlockParams):
        code: Literal["4b5b", "64b66b"] = "64b66b"

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        b = x.data
        if self.params.code == "4b5b":
            if x.length % 4:
                raise BlockError("4B/5B needs a multiple of 4 bits.")
            nib = b.reshape(*b.shape[:-1], -1, 4) @ np.array([8, 4, 2, 1])
            out = FOUR_B_FIVE_B[nib].reshape(*b.shape[:-1], -1)
        else:
            if x.length % 64:
                raise BlockError("64B/66B needs a multiple of 64 bits.")
            scrambled = _self_sync_scramble(b).reshape(*b.shape[:-1], -1, 64)
            header = np.broadcast_to(np.array([0, 1], dtype=np.uint8), (*scrambled.shape[:-1], 2))
            out = np.concatenate([header, scrambled], axis=-1).reshape(*b.shape[:-1], -1)
        return {"out": x.evolve(out, code_rate=x.meta.get("code_rate", 1.0) * x.length / out.shape[-1])}


@register_block("line_block_decoder", category="decoder", subcategory="Line Block Codes", label="Line Block Decoder (4B/5B, 64B/66B)")
class LineBlockDecoder(Block):
    """Inverse of ``line_block_encoder``; counts invalid code words / sync headers."""

    class Params(BlockParams):
        code: Literal["4b5b", "64b66b"] = "64b66b"

    inputs = {"in": Port(Kind.BITS | Kind.LLR)}
    outputs = {"out": Port(Kind.BITS)}

    def setup(self) -> None:
        lut = np.full(32, 0, dtype=np.int64)
        valid = np.zeros(32, dtype=bool)
        for value, row in enumerate(FOUR_B_FIVE_B):
            idx = int("".join(map(str, row)), 2)
            lut[idx], valid[idx] = value, True
        self._lut, self._valid = lut, valid

    def on_reset(self) -> None:
        self._words = 0
        self._invalid = 0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        b = (x.data < 0).astype(np.uint8) if x.kind == Kind.LLR else x.data
        if self.params.code == "4b5b":
            words = b[..., : (b.shape[-1] // 5) * 5].reshape(*b.shape[:-1], -1, 5) @ np.array([16, 8, 4, 2, 1])
            self._words += words.size
            self._invalid += int(np.count_nonzero(~self._valid[words]))
            nib = self._lut[words]
            out = ((nib[..., None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8).reshape(*b.shape[:-1], -1)
        else:
            blocks = b[..., : (b.shape[-1] // 66) * 66].reshape(*b.shape[:-1], -1, 66)
            header_ok = blocks[..., 0] != blocks[..., 1]
            self._words += header_ok.size
            self._invalid += int(np.count_nonzero(~header_ok))
            out = _self_sync_descramble(blocks[..., 2:].reshape(*b.shape[:-1], -1))
        return {"out": Signal(out, Kind.BITS, meta=x.meta)}

    def metrics(self) -> dict[str, float]:
        return {"invalid_words": self._invalid, "code_words": self._words}


# --------------------------------------------------------------------- cables
# Insertion loss in dB/km: a * sqrt(f_MHz) + b * f_MHz (+ c / sqrt(f_MHz) for the TIA categories).
CABLE_PRESETS: dict[str, tuple[float, float]] = {
    "cat5e": (19.67, 0.23),  # TIA-568 Cat 5e (per 100 m: 1.967 sqrt(f) + 0.023 f + 0.050 / sqrt(f))
    "cat6a": (18.2, 0.091),  # TIA-568 Cat 6A (per 100 m: 1.82 sqrt(f) + 0.0091 f + 0.25 / sqrt(f))
    "dsl_0.4mm": (22.5, 0.0),  # 0.4 mm (26 AWG) telephone twisted pair, sqrt(f) fit
    "dsl_0.5mm": (17.5, 0.0),  # 0.5 mm (24 AWG) telephone twisted pair, sqrt(f) fit
    "rg6_coax": (6.5, 0.003),  # RG-6 coaxial cable
    "backplane": (3.16, 1.0),  # FR-4 PCB trace (~0.1 dB/m/sqrt(GHz) + 1 dB/m/GHz)
}
LOW_FREQUENCY_TERM = {"cat5e": 0.50, "cat6a": 2.5}  # c in dB/km x sqrt(MHz), valid from 1 MHz
DIELECTRIC_CONSTANT = {"backplane": 4.0, "rg6_coax": 1.5}  # others: polyethylene, 2.3

# Primary constants of polyethylene-insulated cable at 70 F (ITU-T G.996.1, Annex A, Tables A.1/A.2):
# frequency (Hz), R (ohm/mile, loop), L (mH/mile), G (umho/mile); C = 0.083 uF/mile for both gauges.
_PIC_F = np.array([1, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1e3, 1.5e3, 2e3, 3e3, 5e3, 7e3,
                   1e4, 1.5e4, 2e4, 3e4, 5e4, 7e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6, 3e6, 5e6])
_PIC_G = np.array([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.011, 0.016, 0.022, 0.028, 0.040, 0.063, 0.084,
                   0.115, 0.164, 0.210, 0.299, 0.466, 0.625, 0.853, 1.213, 1.558, 2.217, 3.458, 4.634, 6.320,
                   8.993, 11.550, 16.436, 25.633, 34.351, 46.849, 66.665, 85.624, 121.841, 190.021])
PIC_CABLES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "pic_26awg": (
        np.array([440.75] * 4 + [440.76] * 8 + [440.77, 440.78, 440.79, 440.81, 440.83, 440.88, 441.01, 441.15,
                  441.39, 441.87, 442.83, 443.88, 447.81, 453.09, 463.39, 485.80, 513.04, 575.17, 699.61,
                  812.95, 956.65, 1154.38, 1321.07, 1600.68, 2044.07]),
        np.array([0.9861] * 13 + [0.9859, 0.9858, 0.9856, 0.9854, 0.9850, 0.9843, 0.9836, 0.9825, 0.9807,
                  0.9789, 0.9758, 0.9660, 0.9546, 0.9432, 0.9306, 0.9212, 0.9062, 0.8816, 0.8614, 0.8331,
                  0.8146, 0.8001, 0.7823, 0.7638])),
    "pic_24awg": (
        np.array([277.19] * 9 + [277.20] * 3 + [277.21, 277.22, 277.23, 277.25, 277.28, 277.34, 277.48, 277.66,
                  277.96, 278.58, 279.35, 281.30, 286.82, 294.29, 308.41, 337.22, 369.03, 431.55, 541.69,
                  632.08, 746.04, 902.84, 1035.03, 1256.77, 1608.38]),
        np.array([0.9861] * 9 + [0.9860] * 3 + [0.9859, 0.9858, 0.9857, 0.9854, 0.9852, 0.9848, 0.9839, 0.9829,
                  0.9816, 0.9793, 0.9770, 0.9723, 0.9577, 0.9464, 0.9347, 0.9204, 0.9087, 0.8885, 0.8570,
                  0.8350, 0.8146, 0.7947, 0.7825, 0.7676, 0.7523])),
}
MILE_M = 1609.344
CablePreset = Literal["cat5e", "cat6a", "dsl_0.4mm", "dsl_0.5mm", "rg6_coax", "backplane", "pic_26awg", "pic_24awg"]


def pic_primary_constants(f: np.ndarray, preset: str) -> tuple[np.ndarray, ...]:
    """R, L, G, C per metre of a G.996.1 PIC cable at |f| (log-frequency interpolation).

    Above the 5 MHz end of the table R grows as sqrt(f), G as f and L stays constant.
    """
    r_mi, l_mi = PIC_CABLES[preset]
    fa = np.maximum(np.abs(f), 1.0)
    lf, lt = np.log10(np.minimum(fa, _PIC_F[-1])), np.log10(_PIC_F)
    over = np.maximum(fa / _PIC_F[-1], 1.0)
    r = np.interp(lf, lt, r_mi) * np.sqrt(over) / MILE_M
    ind = np.interp(lf, lt, l_mi) * 1e-3 / MILE_M
    g = np.interp(lf, lt, _PIC_G) * over * 1e-6 / MILE_M
    return r, ind, g, np.full_like(r, 0.083e-6 / MILE_M)


def pic_response(f: np.ndarray, preset: str, length_m: float, termination_ohm: float = 100.0) -> np.ndarray:
    """Insertion transfer function of a PIC loop between equal resistive terminations (ABCD matrix).

    H = 2 Z / (A Z + B + C Z^2 + D Z) with A = D = cosh(gamma l), B = Z0 sinh(gamma l),
    C = sinh(gamma l) / Z0. The front delay l sqrt(L C) at the top of the table is removed.
    """
    r, ind, g, c = pic_primary_constants(f, preset)
    w = 2 * np.pi * np.maximum(np.abs(f), 1.0)  # the table starts at 1 Hz (DC behaves like 1 Hz)
    zs, ys = r + 1j * w * ind, g + 1j * w * c
    x = np.sqrt(zs * ys) * length_m
    z0 = np.sqrt(zs / ys)
    z = termination_ohm
    h = 2 * z / (2 * np.cosh(x) * z + z0 * np.sinh(x) + np.sinh(x) / z0 * z ** 2)
    _, l_hi, _, c_hi = pic_primary_constants(np.array([_PIC_F[-1]]), preset)
    h = h * np.exp(1j * w * length_m * np.sqrt(l_hi[0] * c_hi[0]))
    return np.where(f < 0, np.conj(h), h)


def dielectric_response(f: np.ndarray, loss_db_m_hz: float, length_m: float, eps_r: float = 4.0) -> np.ndarray:
    """Causal dielectric loss (Djordjevic-Sarkar wideband permittivity), loss proportional to f.

    eps(w) = eps_inf + d_eps / (m2 - m1) log10((w2 + j w) / (w1 + j w)) with w1 = 1e4 and
    w2 = 1e12 rad/s has an almost constant loss tangent in between and is causal. The loss
    tangent is set so that the attenuation is ``loss_db_m_hz`` x f, and eps' = eps_r at 1 GHz.
    The response is exp(-j w l (sqrt(eps) - sqrt(eps_inf)) / c): the front delay is removed.
    """
    if loss_db_m_hz == 0:
        return np.ones_like(f, dtype=np.complex128)
    m1, m2 = 4.0, 12.0
    c0 = 299_792_458.0
    tan_d = loss_db_m_hz * c0 / (20 / np.log(10) * np.pi * np.sqrt(eps_r))
    d_eps = tan_d * eps_r * (m2 - m1) * np.log(10) / (np.pi / 2)

    def log_term(w):
        return d_eps / (m2 - m1) * np.log10((10 ** m2 + 1j * w) / (10 ** m1 + 1j * w))

    eps_inf = eps_r - float(np.real(log_term(2 * np.pi * 1e9)))
    w = 2 * np.pi * np.asarray(f, dtype=float)
    n = np.sqrt(eps_inf + log_term(w))
    return np.exp(-1j * w * length_m * (n - np.sqrt(eps_inf)) / c0)


def cable_response(f: np.ndarray, preset: str, length_m: float, a: float | None = None,
                   b: float | None = None, termination_ohm: float = 100.0) -> np.ndarray:
    """Complex frequency response of a cable.

    * ``pic_*`` presets: RLCG transmission line with the ITU-T G.996.1 primary constants.
    * Other presets: skin effect a sqrt(f) with its exact minimum phase, sqrt(j w), times a
      causal dielectric loss b f (:func:`dielectric_response`). The TIA categories add the
      term c / sqrt(f) of their insertion-loss formula (as attenuation only, held constant
      below 1 MHz where the formula is not specified).
    """
    if preset in PIC_CABLES:
        return pic_response(f, preset, length_m, termination_ohm)
    pa, pb = CABLE_PRESETS[preset]
    a = pa if a is None else a
    b = pb if b is None else b
    f_mhz = np.abs(f) / 1e6
    km = length_m / 1000
    neper = np.log(10) / 20
    skin = a * np.sqrt(f_mhz) * km * neper
    low = LOW_FREQUENCY_TERM.get(preset, 0.0) / np.sqrt(np.maximum(f_mhz, 1.0)) * km * neper
    dielectric = dielectric_response(f, b * 1e-9, length_m, DIELECTRIC_CONSTANT.get(preset, 2.3))
    return np.exp(-skin * (1 + 1j * np.sign(f)) - low) * dielectric


def _require_fs(x: Signal, what: str) -> float:
    if not x.fs:
        raise BlockError(f"{what} needs the sample rate (fs) of the signal.")
    return x.fs


@register_block("copper_line", category="medium", subcategory="Copper", label="Copper Line")
class CopperLine(Block):
    """Cable channel with frequency-dependent loss (twisted pair, coax, PCB trace).

    The ``csi`` output carries the sampled impulse response (``csi_type="cir"``) so
    the ``linear_equalizer`` or ``dmt_demodulator`` can use it. Loss in dB/km is
    ``a * sqrt(f_MHz) + b * f_MHz`` from the preset unless ``a`` / ``b`` are given; the
    ``pic_26awg`` / ``pic_24awg`` presets are RLCG lines with the ITU-T G.996.1 constants.
    """

    class Params(BlockParams):
        preset: CablePreset = "cat5e"
        length_m: float = Field(100.0, gt=0, description="Cable length in metres.")
        termination_ohm: float = Field(100.0, gt=0, description="Source and load resistance (pic_* presets).")
        a: float | None = Field(None, ge=0, description="Skin-effect loss, dB/km/sqrt(MHz).")
        b: float | None = Field(None, ge=0, description="Dielectric loss, dB/km/MHz.")
        max_taps: int = Field(256, ge=4, description="Longest impulse response reported on csi.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in"), "csi": Port(Kind.CSI)}

    def response(self, f: np.ndarray) -> np.ndarray:
        p = self.params
        return cable_response(f, p.preset, p.length_m, p.a, p.b, p.termination_ohm)

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        return self.response(f)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "copper_line")
        n_h = 4096
        h = np.fft.ifft(self.response(np.fft.fftfreq(n_h, 1 / fs))).real
        energy = np.cumsum(h ** 2) / np.sum(h ** 2)
        L = int(min(np.searchsorted(energy, 0.9995) + 1, self.params.max_taps))
        y = apply_frequency_response(x.data, self.response, fs, pad=4 * L)
        cir = np.broadcast_to(h[:L], (x.data.shape[0], L)).astype(np.complex128)
        csi = Signal(cir, Kind.CSI, fs=fs, meta={**x.meta, "csi_type": "cir"})
        return {"out": x.evolve(y, power=None), "csi": csi}


def crosstalk_psd_gain(f: np.ndarray, model: str, n: int, length_m: float, line: np.ndarray | None) -> np.ndarray:
    """|H_xt(f)|^2 of the ANSI T1.417 99 % worst-case NEXT/FEXT models (f in Hz)."""
    fa = np.abs(f)
    if model == "next":
        return 8.536e-15 * n ** 0.6 * fa ** 1.5
    length_ft = length_m * 3.281
    line_gain = np.abs(line) ** 2 if line is not None else 1.0
    return 7.744e-21 * n ** 0.6 * length_ft * fa ** 2 * line_gain


@register_block("crosstalk", category="medium", subcategory="Copper", label="Crosstalk (NEXT, FEXT)")
class Crosstalk(Block):
    """Near-end (NEXT) or far-end (FEXT) crosstalk from ``n_disturbers`` similar pairs.

    The disturbers have the spectrum of the ``tx`` input (the transmitted signal of a
    neighbouring pair; the ``in`` signal itself when unconnected) with random phases,
    shaped by the ANSI coupling model. FEXT also includes the line loss.
    """

    class Params(BlockParams):
        model: Literal["next", "fext"] = "next"
        n_disturbers: int = Field(10, ge=1, le=100)
        preset: CablePreset = "dsl_0.4mm"
        length_m: float = Field(1000.0, gt=0, description="Coupling length (FEXT) in metres.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES), "tx": Port(Kind.SYMBOLS | Kind.SAMPLES, required=False)}
    outputs = {"out": Port.like("in")}

    def on_reset(self) -> None:
        self._ratio: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        ref = inputs.get("tx", x).data
        fs = _require_fs(x, "crosstalk")
        n = x.length
        f = np.fft.fftfreq(n, 1 / fs)
        p = self.params
        line = cable_response(f, p.preset, p.length_m) if p.model == "fext" else None
        gain = np.sqrt(crosstalk_psd_gain(f, p.model, p.n_disturbers, p.length_m, line))
        ref = ref[..., :n] if ref.shape[-1] >= n else np.pad(ref, [(0, 0)] * (ref.ndim - 1) + [(0, n - ref.shape[-1])])
        w = np.fft.fft(self.rng.standard_normal(ref.shape), axis=-1)
        spectrum = np.abs(np.fft.fft(ref, axis=-1)) * w / np.maximum(np.abs(w), 1e-12) * gain
        xt = np.fft.ifft(spectrum, axis=-1)
        xt = xt.real if not np.iscomplexobj(x.data) else xt
        self._ratio.append(float(np.mean(np.abs(xt) ** 2) / max(np.mean(np.abs(x.data) ** 2), 1e-30)))
        return {"out": x.evolve(x.data + xt)}

    def metrics(self) -> dict[str, float]:
        return {"crosstalk_to_signal_db": 10 * float(np.log10(np.mean(self._ratio)))} if self._ratio else {}


@register_block("impulse_noise", category="medium", subcategory="Copper", label="Impulse Noise")
class ImpulseNoise(Block):
    """Bursts of strong Gaussian noise (electrical switching, appliances)."""

    class Params(BlockParams):
        burst_probability: float = Field(1e-3, ge=0, le=1, description="Burst start probability per sample.")
        burst_samples: int = Field(32, ge=1, description="Burst duration.")
        power_db: float = Field(10.0, description="Burst noise power relative to the signal power.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        starts = (self.rng.random(x.data.shape) < p.burst_probability).astype(float)
        active = np.apply_along_axis(lambda r: np.convolve(r, np.ones(p.burst_samples))[: r.size], -1, starts) > 0
        sigma = np.sqrt(np.mean(np.abs(x.data) ** 2) * 10 ** (p.power_db / 10))
        noise = self.rng.standard_normal(x.data.shape)
        if np.iscomplexobj(x.data):
            noise = (noise + 1j * self.rng.standard_normal(x.data.shape)) / np.sqrt(2)
        return {"out": x.evolve(x.data + active * sigma * noise)}


# ------------------------------------------------------------- full duplex echo
@register_block("hybrid_echo", category="medium", subcategory="Copper", label="Hybrid Echo")
class HybridEcho(Block):
    """Leakage of the local transmitter into the receiver through the hybrid (full duplex)."""

    class Params(BlockParams):
        echo_db: float = Field(-10.0, description="Echo level relative to the local transmit signal.")
        taps: list[float] = Field([1.0, 0.6, 0.3, 0.1], description="Echo path impulse response shape.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES), "tx": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, tx = inputs["in"], inputs["tx"]
        taps = np.asarray(self.params.taps) / np.linalg.norm(self.params.taps) * 10 ** (self.params.echo_db / 20)
        n = x.length
        echo = np.apply_along_axis(lambda r: np.convolve(r, taps)[:n], -1, tx.data[..., :n])
        return {"out": x.evolve(x.data + echo)}


@register_block("echo_canceller", category="equalizer", subcategory="Echo Cancellation", label="Echo Canceller")
class EchoCanceller(Block):
    """NLMS adaptive filter estimating the echo from the local ``tx`` signal and subtracting it.

    A large step is used for the first ``train_samples`` (fast convergence, as during link
    start-up), then a small one keeps the misadjustment noise low.
    """

    class Params(BlockParams):
        n_taps: int = Field(16, ge=1)
        train_step_size: float = Field(0.05, gt=0, lt=2)
        step_size: float = Field(0.002, gt=0, lt=2, description="Tracking step after training.")
        train_samples: int = Field(1000, ge=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES), "tx": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def on_reset(self) -> None:
        self._in_power = self._out_power = 0.0

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x, tx = inputs["in"], inputs["tx"]
        y = x.data
        B, N = y.shape
        T = self.params.n_taps
        ref = np.concatenate([np.zeros((B, T - 1)), tx.data[:, :N]], axis=1)
        w = np.zeros((B, T), dtype=np.result_type(y, ref))
        out = np.empty_like(y)
        for n in range(N):
            u = ref[:, n:n + T][:, ::-1]
            e = y[:, n] - np.sum(w * u, axis=1)
            out[:, n] = e
            mu = self.params.train_step_size if n < self.params.train_samples else self.params.step_size
            w += mu * e[:, None] * np.conj(u) / (1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True))
        half = N // 2  # measure after convergence
        self._in_power += float(np.mean(np.abs(y[:, half:]) ** 2))
        self._out_power += float(np.mean(np.abs(out[:, half:]) ** 2))
        return {"out": x.evolve(out)}

    def metrics(self) -> dict[str, float]:
        if not self._in_power:
            return {}
        return {"power_reduction_db": 10 * float(np.log10(self._in_power / max(self._out_power, 1e-30)))}


# ------------------------------------------------------------ SerDes receiver
@register_block("ctle", category="equalizer", subcategory="Analog", label="CTLE")
class CTLE(Block):
    """Continuous-time linear equalizer: H(s) = A (1 + s/wz) / ((1 + s/wp1)(1 + s/wp2)).

    Boosts high frequencies by about 20 log10(f_p1 / f_z) dB to compensate cable loss.
    """

    class Params(BlockParams):
        zero_hz: float = Field(1e9, gt=0)
        pole1_hz: float = Field(6e9, gt=0)
        pole2_hz: float = Field(2.5e10, gt=0)
        dc_gain_db: float = 0.0

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def response(self, f: np.ndarray) -> np.ndarray:
        p = self.params
        s = 2j * np.pi * f
        return 10 ** (p.dc_gain_db / 20) * (1 + s / (2 * np.pi * p.zero_hz)) / (
            (1 + s / (2 * np.pi * p.pole1_hz)) * (1 + s / (2 * np.pi * p.pole2_hz)))

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        return self.response(f)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        y = apply_frequency_response(x.data, self.response, _require_fs(x, "ctle"), pad=256)
        return {"out": x.evolve(y, noise_var=None, power=None)}


@register_block("dfe_equalizer", category="equalizer", subcategory="Adaptive", label="Decision-Feedback Equalizer")
class DFEEqualizer(Block):
    """Adaptive feed-forward + decision-feedback equalizer (NLMS), symbol spaced.

    Trained with the first ``n_train`` symbols of ``ref``, then decision directed with
    the upstream constellation. The DFE cancels post-cursor ISI without noise
    enhancement, which is why SerDes receivers use it.
    """

    class Params(BlockParams):
        ffe_taps: int = Field(7, ge=1)
        dfe_taps: int = Field(4, ge=0)
        step_size: float = Field(0.05, gt=0, lt=2)
        n_train: int = Field(300, ge=0)

    inputs = {"in": Port(Kind.SYMBOLS), "ref": Port(Kind.SYMBOLS, required=False)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        ref = inputs.get("ref")
        const: Constellation | None = x.meta.get("constellation")
        p = self.params
        if const is None:
            raise BlockError("dfe_equalizer needs the constellation (mapper or nrz line_encoder upstream).")
        if ref is None and p.n_train:
            raise BlockError("Training needs the 'ref' input.")
        y = x.data.astype(np.complex128)
        B, N = y.shape
        F, K, D = p.ffe_taps, p.dfe_taps, p.ffe_taps // 2
        pad = np.concatenate([np.zeros((B, F - 1 - D)), y, np.zeros((B, D))], axis=1)
        w = np.zeros((B, F), dtype=np.complex128)
        w[:, F - 1 - D] = 1.0
        bfb = np.zeros((B, K), dtype=np.complex128)
        past = np.zeros((B, K), dtype=np.complex128)
        out = np.empty((B, N), dtype=np.complex128)
        err, count = 0.0, 0
        for n in range(N):
            u = pad[:, n:n + F][:, ::-1]
            z = np.sum(w * u, axis=1) - np.sum(bfb * past, axis=1)
            out[:, n] = z
            d = ref.data[:, n] if n < p.n_train else const.points[const.nearest(z)]
            e = d - z
            if n >= p.n_train:
                err += float(np.sum(np.abs(e) ** 2))
                count += B
            norm = 1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True) + np.sum(np.abs(past) ** 2, axis=1, keepdims=True)
            w += p.step_size * e[:, None] * np.conj(u) / norm
            if K:
                bfb -= p.step_size * e[:, None] * np.conj(past) / norm
                past = np.concatenate([d[:, None], past[:, :-1]], axis=1)
        if not np.iscomplexobj(x.data):
            out = out.real
        return {"out": x.evolve(out, noise_var=err / count if count else x.meta.get("noise_var"))}


@register_block("jitter", category="impairment", subcategory="Timing", label="Clock Jitter")
class Jitter(Block):
    """Sampling-clock jitter: random (RJ, per symbol) and sinusoidal (SJ), in unit intervals."""

    class Params(BlockParams):
        rj_rms_ui: float = Field(0.02, ge=0, description="Random jitter RMS (fraction of a symbol).")
        sj_amplitude_ui: float = Field(0.0, ge=0, description="Sinusoidal jitter amplitude (UI).")
        sj_frequency_hz: float = Field(1e6, gt=0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        sps = x.meta.get("sps", 1)
        fs = _require_fs(x, "jitter")
        B, N = x.data.shape
        n = np.arange(N)
        n_sym = N // sps + 2
        rj = self.rng.standard_normal((B, n_sym)) * self.params.rj_rms_ui
        rj_t = np.stack([np.interp(n / sps, np.arange(n_sym), r) for r in rj])
        sj = self.params.sj_amplitude_ui * np.sin(2 * np.pi * self.params.sj_frequency_hz * n / fs)
        t = n[None, :] + sps * (rj_t + sj[None, :])
        return {"out": x.evolve(cubic_resample(x.data, t))}


# ------------------------------------------------------------------------ DMT
def waterfill_bits(snr: np.ndarray, gap_db: float, margin_db: float, max_bits: int) -> np.ndarray:
    """Bits per tone from the gap approximation b = log2(1 + SNR / (gap * margin)).

    Rounded down to the sizes the mapper supports (0, 1 = BPSK, or even = square QAM).
    """
    gamma = 10 ** ((gap_db + margin_db) / 10)
    b = np.floor(np.log2(1 + snr / gamma)).astype(int)
    b = np.clip(b, 0, max_bits)
    return np.where(b >= 2, b - (b % 2), b)


def _tone_constellation(bits: int) -> Constellation:
    return psk(2) if bits == 1 else qam(1 << bits)


@register_block("dmt_modulator", category="modulator", subcategory="Multicarrier", label="DMT Modulator")
class DMTModulator(Block):
    """Discrete multitone (ADSL/VDSL): real IFFT with Hermitian symmetry, per-tone bit loading.

    ``waterfill`` loading computes the expected SNR of every tone from the cable model
    (``line_preset``, ``line_length_m``) and the AWGN level ``snr_db`` (same definition
    as the ``awgn`` block in ``snr`` mode), then applies the gap approximation.
    The total bits per DMT symbol and the data rate are reported as metrics.
    """

    class Params(BlockParams):
        n_fft: int = Field(512, ge=16, description="Real IFFT size (tones = n_fft / 2).")
        cp_len: int = Field(40, ge=0)
        first_tone: int = Field(33, ge=1, description="First used tone (ADSL downstream starts at 33).")
        last_tone: int = Field(255, ge=1)
        sample_rate: float = Field(2.208e6, gt=0, description="Hz (ADSL: 4.3125 kHz x 512).")
        loading: Literal["uniform", "waterfill"] = "waterfill"
        uniform_bits: int = Field(4, ge=1, le=14)
        max_bits: int = Field(14, ge=1, le=14)
        gap_db: float = Field(9.8, description="SNR gap (9.8 dB: uncoded QAM at BER 1e-7).")
        margin_db: float = Field(3.0)
        snr_db: float = Field(60.0, description="Expected AWGN SNR (awgn 'snr' mode).")
        line_preset: CablePreset = "dsl_0.4mm"
        line_length_m: float = Field(2000.0, gt=0)

    inputs = {"in": Port(Kind.BITS)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def setup(self) -> None:
        p = self.params
        if p.last_tone >= p.n_fft // 2 or p.first_tone > p.last_tone:
            raise BlockError("Need 1 <= first_tone <= last_tone < n_fft / 2.")
        tones = np.arange(p.first_tone, p.last_tone + 1)
        if p.loading == "uniform":
            bits = np.full(tones.size, p.uniform_bits - (p.uniform_bits % 2 if p.uniform_bits > 1 else 0))
        else:
            f = tones * p.sample_rate / p.n_fft
            H = cable_response(f, p.line_preset, p.line_length_m)
            snr = 10 ** (p.snr_db / 10) * np.abs(H) ** 2 * p.n_fft / (2 * tones.size)
            bits = waterfill_bits(snr, p.gap_db, p.margin_db, p.max_bits)
        if bits.sum() == 0:
            raise BlockError("No tone can carry data: raise snr_db or shorten the line.")
        self.tones, self.bits = tones, bits
        self.offsets = np.concatenate([[0], np.cumsum(bits)[:-1]])
        self.groups = {int(b): np.nonzero(bits == b)[0] for b in np.unique(bits) if b > 0}
        self.consts = {b: _tone_constellation(b) for b in self.groups}

    @property
    def bits_per_symbol(self) -> int:
        return int(self.bits.sum())

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        B, n = x.data.shape[0], x.length
        per = self.bits_per_symbol
        n_sym = -(-n // per)
        bits = np.zeros((B, n_sym * per), dtype=np.uint8)
        bits[:, :n] = x.data
        bits = bits.reshape(B, n_sym, per)
        X = np.zeros((B, n_sym, p.n_fft), dtype=np.complex128)
        for b, idx in self.groups.items():
            sel = self.offsets[idx][:, None] + np.arange(b)  # (n_tones_b, b)
            X[..., self.tones[idx]] = self.consts[b].map(bits[..., sel].reshape(B, n_sym, -1))
        X[..., p.n_fft - self.tones] = np.conj(X[..., self.tones])
        t = np.fft.ifft(X, axis=-1, norm="ortho").real
        out = np.concatenate([t[..., p.n_fft - p.cp_len:], t], axis=-1).reshape(B, -1)
        loaded = int(np.count_nonzero(self.bits))
        meta = {**x.meta, "dmt": self, "n_bits": n, "power": 2 * loaded / p.n_fft, "sps": 1, "delay": 0.0}
        return {"out": Signal(out, Kind.SAMPLES, fs=p.sample_rate, meta=meta)}

    def metrics(self) -> dict[str, float]:
        p = self.params
        return {"bits_per_dmt_symbol": self.bits_per_symbol,
                "data_rate_mbps": self.bits_per_symbol * p.sample_rate / (p.n_fft + p.cp_len) / 1e6}


@register_block("dmt_demodulator", category="demodulator", subcategory="Multicarrier", label="DMT Demodulator")
class DMTDemodulator(Block):
    """Remove the cyclic prefix, FFT, one-tap frequency-domain equalization and per-tone demapping.

    The FEQ uses the channel impulse response on ``csi`` (e.g. from ``copper_line``);
    without it the channel is assumed flat.
    """

    inputs = {"in": Port(Kind.SAMPLES), "csi": Port(Kind.CSI, required=False)}
    outputs = {"out": Port(Kind.BITS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        mod: DMTModulator | None = x.meta.get("dmt")
        if mod is None:
            raise BlockError("dmt_demodulator needs a dmt_modulator upstream.")
        p = mod.params
        L = p.n_fft + p.cp_len
        B = x.data.shape[0]
        n_sym = x.length // L
        frames = np.real(x.data[:, :n_sym * L]).reshape(B, n_sym, L)[..., p.cp_len:]
        Y = np.fft.fft(frames, axis=-1, norm="ortho")[..., mod.tones]
        if "csi" in inputs:
            H = np.fft.fft(inputs["csi"].data, n=p.n_fft, axis=-1)[:, None, mod.tones]
            Y = Y / H
        bits = np.zeros((B, n_sym, mod.bits_per_symbol), dtype=np.uint8)
        for b, idx in mod.groups.items():
            dec = mod.consts[b].demap_hard(Y[..., idx])  # (B, n_sym, n_tones_b * b)
            sel = mod.offsets[idx][:, None] + np.arange(b)
            bits[..., sel.reshape(-1)] = dec
        out = bits.reshape(B, -1)[:, : x.meta.get("n_bits", bits.size // B)]
        return {"out": Signal(out, Kind.BITS, meta=x.meta)}
