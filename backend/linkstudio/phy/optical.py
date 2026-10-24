"""Optical communications: transmitters, fiber, amplifiers, receivers and coherent DSP.

Optical signals are complex field envelopes in sqrt(W) (|E|^2 = optical power in W),
shape ``(batch, n)`` or ``(batch, 2, n)`` for dual polarization. Metadata:
``optical`` (True), ``wavelength_nm``, ``power`` (W), ``cd_ps_nm`` (accumulated
chromatic dispersion) and ``ase_psd`` (ASE power spectral density per polarization,
W/Hz, used for OSNR).

* IM/DD: ``intensity_modulator`` -> ``fiber`` / ``optical_amplifier`` -> ``photodiode`` -> ``agc``
* Coherent: ``iq_modulator`` -> ``optical_link`` -> ``coherent_receiver`` -> ``cd_compensation``
  -> ``fo_compensation`` -> ``adaptive_mimo_equalizer`` -> ``phase_recovery``
* Free space: ``fso_turbulence``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation

C_LIGHT = 299_792_458.0
H_PLANCK = 6.62607015e-34
Q_E = 1.602176634e-19
REF_BW = 12.5e9  # OSNR reference bandwidth (0.1 nm at 1550 nm)


def dbm_to_w(p_dbm: float) -> float:
    return 1e-3 * 10 ** (p_dbm / 10)


def w_to_dbm(p_w: float) -> float:
    return 10 * np.log10(max(p_w, 1e-300) / 1e-3)


def _require_fs(x: Signal, what: str) -> float:
    if not x.fs:
        raise BlockError(f"{what} needs the sample rate (fs) of the signal.")
    return x.fs


def _wiener_phase(rng: np.random.Generator, batch: int, n: int, linewidth: float, fs: float) -> np.ndarray:
    if linewidth <= 0:
        return np.zeros((batch, n))
    return np.cumsum(rng.standard_normal((batch, n)) * np.sqrt(2 * np.pi * linewidth / fs), axis=-1)


def _pol_axes(e: np.ndarray) -> tuple[int, ...]:
    return (-1,) if e.ndim == 2 else (-2, -1)


def osnr_db(meta: dict) -> float | None:
    """OSNR in 0.1 nm from the signal power and the accumulated ASE density."""
    ase = meta.get("ase_psd")
    power = meta.get("power")
    if not ase or power is None:
        return None
    return 10 * float(np.log10(power / (2 * ase * REF_BW)))


# ---------------------------------------------------------------- transmitters
@register_block("intensity_modulator", category="optical")
class IntensityModulator(Block):
    """Laser with intensity modulation (directly modulated laser / EML) for IM/DD links.

    The real drive (NRZ, PAM-4, DMT...) is normalized to [-1, 1] and mapped to the optical
    power P = P_avg (1 + m x) with modulation depth m = (ER - 1) / (ER + 1) for extinction
    ratio ER. Optional relative intensity noise, chirp (phase = alpha/2 ln P) and laser
    phase noise.
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Average optical power.")
        extinction_ratio_db: float = Field(10.0, gt=0)
        rin_db_hz: float | None = Field(None, description="Relative intensity noise (e.g. -150 dB/Hz).")
        chirp_alpha: float = Field(0.0, description="Henry factor of a directly modulated laser.")
        linewidth_hz: float = Field(0.0, ge=0)
        wavelength_nm: float = Field(1550.0, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "intensity_modulator")
        drive = np.real(x.data)
        peak = np.max(np.abs(drive)) or 1.0
        er = 10 ** (p.extinction_ratio_db / 10)
        m = (er - 1) / (er + 1)
        p_avg = dbm_to_w(p.power_dbm)
        power = p_avg * (1 + m * np.clip(drive / peak, -1, 1))
        if p.rin_db_hz is not None:
            power = power * (1 + np.sqrt(10 ** (p.rin_db_hz / 10) * fs / 2) * self.rng.standard_normal(power.shape))
        power = np.maximum(power, 0.0)
        phase = _wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
        if p.chirp_alpha:
            phase = phase + p.chirp_alpha / 2 * np.log(np.maximum(power, 1e-12 * p_avg) / p_avg)
        field = np.sqrt(power) * np.exp(1j * phase)
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg,
                "cd_ps_nm": 0.0, "ase_psd": 0.0}
        return {"out": Signal(field, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("iq_modulator", category="optical")
class IQModulator(Block):
    """Laser + IQ Mach-Zehnder modulator for coherent transmission (single or dual polarization).

    ``linear`` maps the complex drive directly onto the field; ``mzm`` applies the
    sinusoidal MZM transfer function to I and Q with peak drive ``drive_swing`` x V_pi.
    The laser power is split equally between the polarizations.
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Launch power (all polarizations).")
        model: Literal["linear", "mzm"] = "linear"
        drive_swing: float = Field(0.5, gt=0, le=1, description="Peak drive relative to V_pi (mzm).")
        linewidth_hz: float = Field(100e3, ge=0, description="Laser linewidth.")
        wavelength_nm: float = Field(1550.0, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "iq_modulator")
        d = x.data.astype(np.complex128)
        npol = d.shape[1] if d.ndim == 3 else 1
        if p.model == "mzm":
            peak = max(np.max(np.abs(d.real)), np.max(np.abs(d.imag))) or 1.0
            arg = np.pi / 2 * p.drive_swing
            d = (np.sin(arg * d.real / peak) + 1j * np.sin(arg * d.imag / peak))
        rms = np.sqrt(np.mean(np.abs(d) ** 2, axis=_pol_axes(d), keepdims=True))
        p_tot = dbm_to_w(p.power_dbm)
        field = d / np.maximum(rms, 1e-30) * np.sqrt(p_tot / npol)
        phase = _wiener_phase(self.rng, d.shape[0], d.shape[-1], p.linewidth_hz, fs)
        field = field * np.exp(1j * (phase[:, None, :] if d.ndim == 3 else phase))
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_tot,
                "cd_ps_nm": 0.0, "ase_psd": 0.0}
        return {"out": x.evolve(field, kind=Kind.SAMPLES, **meta)}


# ------------------------------------------------------------------- fiber
def beta2(dispersion_ps_nm_km: float, wavelength_nm: float) -> float:
    """Group-velocity dispersion beta2 in s^2/m."""
    D = dispersion_ps_nm_km * 1e-6  # s/m^2
    lam = wavelength_nm * 1e-9
    return -D * lam ** 2 / (2 * np.pi * C_LIGHT)


def dispersion_operator(n: int, fs: float, cd_ps_nm: float, wavelength_nm: float) -> np.ndarray:
    """exp(j beta2 L / 2 w^2) for an accumulated dispersion D*L in ps/nm."""
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
    b2l = beta2(cd_ps_nm, wavelength_nm) * 1e3  # D*L in ps/nm == (ps/nm/km) * km
    return np.exp(1j * b2l / 2 * w ** 2)


def _random_unitary(rng: np.random.Generator, batch: int) -> np.ndarray:
    a = rng.standard_normal((batch, 2, 2)) + 1j * rng.standard_normal((batch, 2, 2))
    q, r = np.linalg.qr(a)
    return q * (np.diagonal(r, axis1=1, axis2=2) / np.abs(np.diagonal(r, axis1=1, axis2=2)))[:, None, :]


def propagate(e: np.ndarray, fs: float, length_km: float, att_db_km: float, d_ps_nm_km: float,
              wavelength_nm: float, nonlinear: bool, gamma: float, steps_per_km: float) -> np.ndarray:
    """Split-step Fourier solution of the NLSE (Manakov equation for dual polarization)."""
    n = e.shape[-1]
    alpha = att_db_km / (10 * np.log10(np.e)) / 1e3  # 1/m
    if not nonlinear or gamma == 0:
        lin = dispersion_operator(n, fs, d_ps_nm_km * length_km, wavelength_nm) * np.exp(-alpha * length_km * 1e3 / 2)
        return np.fft.ifft(np.fft.fft(e, axis=-1) * lin, axis=-1)
    steps = max(1, int(np.ceil(length_km * steps_per_km)))
    dz = length_km * 1e3 / steps
    half = dispersion_operator(n, fs, d_ps_nm_km * length_km / steps / 2, wavelength_nm) * np.exp(-alpha * dz / 4)
    # Effective length around the step midpoint (where the nonlinear phase is applied).
    leff = 2 * np.sinh(alpha * dz / 2) / alpha if alpha > 0 else dz
    g = gamma * 1e-3 * (8 / 9 if e.ndim == 3 else 1.0)  # 1/(W km) -> 1/(W m)
    for _ in range(steps):
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
        p = np.sum(np.abs(e) ** 2, axis=1, keepdims=True) if e.ndim == 3 else np.abs(e) ** 2
        e = e * np.exp(1j * g * p * leff)
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
    return e


def apply_pmd(rng: np.random.Generator, e: np.ndarray, fs: float, dgd_s: float) -> np.ndarray:
    """Random polarization rotation plus first-order PMD (differential group delay)."""
    B, _, n = e.shape
    u1, u2 = _random_unitary(rng, B), _random_unitary(rng, B)
    e = np.einsum("bij,bjn->bin", u1, e)
    if dgd_s > 0:
        w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
        E = np.fft.fft(e, axis=-1)
        E[:, 0] *= np.exp(1j * w * dgd_s / 2)
        E[:, 1] *= np.exp(-1j * w * dgd_s / 2)
        e = np.fft.ifft(E, axis=-1)
    return np.einsum("bij,bjn->bin", u2, e)


class _FiberParams(BlockParams):
    attenuation_db_km: float = Field(0.2, ge=0)
    dispersion_ps_nm_km: float = Field(17.0, description="Chromatic dispersion parameter D.")
    nonlinear: bool = Field(False, description="Kerr nonlinearity (split-step Fourier).")
    gamma_w_km: float = Field(1.3, ge=0, description="Nonlinear coefficient (1/W/km).")
    steps_per_km: float = Field(0.5, gt=0, description="Split-step resolution.")
    pmd_ps_sqrt_km: float = Field(0.0, ge=0, description="PMD coefficient (dual polarization).")
    polarization_rotation: bool = Field(False, description="Random polarization state per frame.")


@register_block("fiber", category="optical")
class Fiber(Block):
    """Single-mode fiber span: attenuation, chromatic dispersion, optional Kerr nonlinearity and PMD."""

    class Params(_FiberParams):
        length_km: float = Field(80.0, ge=0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "fiber")
        lam = x.meta.get("wavelength_nm", 1550.0)
        e = propagate(x.data, fs, p.length_km, p.attenuation_db_km, p.dispersion_ps_nm_km, lam,
                      p.nonlinear, p.gamma_w_km, p.steps_per_km)
        if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
            e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.length_km))
        loss = 10 ** (-p.attenuation_db_km * p.length_km / 10)
        return {"out": x.evolve(e, power=(x.meta.get("power") or 1e-3) * loss,
                                ase_psd=x.meta.get("ase_psd", 0.0) * loss,
                                cd_ps_nm=x.meta.get("cd_ps_nm", 0.0) + p.dispersion_ps_nm_km * p.length_km)}


def ase_psd(gain_lin: float, nf_db: float, wavelength_nm: float) -> float:
    """ASE power spectral density per polarization (W/Hz) of an amplifier."""
    if gain_lin <= 1:
        return 0.0
    nu = C_LIGHT / (wavelength_nm * 1e-9)
    nsp = max(10 ** (nf_db / 10) * gain_lin / (2 * (gain_lin - 1)), 0.5)
    return nsp * H_PLANCK * nu * (gain_lin - 1)


def _add_ase(rng: np.random.Generator, e: np.ndarray, psd: float, fs: float) -> np.ndarray:
    sigma = np.sqrt(psd * fs / 2)
    return e + sigma * (rng.standard_normal(e.shape) + 1j * rng.standard_normal(e.shape))


@register_block("optical_amplifier", category="optical")
class OpticalAmplifier(Block):
    """Erbium-doped fiber amplifier with amplified spontaneous emission (ASE) noise.

    ``gain`` mode applies ``gain_db``; ``power`` mode sets the output power to
    ``target_power_dbm``. The OSNR (0.1 nm) after the amplifier is reported as a metric.
    """

    class Params(BlockParams):
        mode: Literal["gain", "power"] = "power"
        gain_db: float = Field(20.0, ge=0)
        target_power_dbm: float = Field(0.0)
        noise_figure_db: float = Field(5.0, ge=3.0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def on_reset(self) -> None:
        self._osnr: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "optical_amplifier")
        p_in = x.meta.get("power") or float(np.mean(np.sum(np.abs(x.data) ** 2, axis=1) if x.data.ndim == 3
                                                    else np.abs(x.data) ** 2))
        g = 10 ** (p.gain_db / 10) if p.mode == "gain" else dbm_to_w(p.target_power_dbm) / p_in
        s = ase_psd(g, p.noise_figure_db, x.meta.get("wavelength_nm", 1550.0))
        e = _add_ase(self.rng, x.data * np.sqrt(g), s, fs)
        meta = {"power": p_in * g, "ase_psd": x.meta.get("ase_psd", 0.0) * g + s}
        o = osnr_db({**x.meta, **meta})
        if o is not None:
            self._osnr.append(o)
        return {"out": x.evolve(e, **meta)}

    def metrics(self) -> dict[str, float]:
        return {"osnr_db": float(np.mean(self._osnr))} if self._osnr else {}


@register_block("optical_link", category="optical")
class OpticalLink(Block):
    """Multi-span amplified link: ``n_spans`` x (fiber span + EDFA compensating the span loss)."""

    class Params(_FiberParams):
        n_spans: int = Field(10, ge=1, le=100)
        span_km: float = Field(80.0, gt=0)
        noise_figure_db: float = Field(5.0, ge=3.0)
        launch_power_dbm: float | None = Field(None, description="Rescale the input to this power.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def on_reset(self) -> None:
        self._osnr: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "optical_link")
        lam = x.meta.get("wavelength_nm", 1550.0)
        e = x.data
        power = (x.meta.get("power") or 1e-3)
        if p.launch_power_dbm is not None:
            e = e * np.sqrt(dbm_to_w(p.launch_power_dbm) / power)
            power = dbm_to_w(p.launch_power_dbm)
        ase = x.meta.get("ase_psd", 0.0)
        g = 10 ** (p.attenuation_db_km * p.span_km / 10)
        s = ase_psd(g, p.noise_figure_db, lam)
        for _ in range(p.n_spans):
            e = propagate(e, fs, p.span_km, p.attenuation_db_km, p.dispersion_ps_nm_km, lam,
                          p.nonlinear, p.gamma_w_km, p.steps_per_km)
            if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
                e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.span_km))
            e = _add_ase(self.rng, e * np.sqrt(g), s, fs)
            ase = ase + s  # the span loss and the amplifier gain cancel
        meta = {"power": power, "ase_psd": ase,
                "cd_ps_nm": x.meta.get("cd_ps_nm", 0.0) + p.dispersion_ps_nm_km * p.span_km * p.n_spans}
        o = osnr_db({**x.meta, **meta})
        if o is not None:
            self._osnr.append(o)
        return {"out": x.evolve(e, **meta)}

    def metrics(self) -> dict[str, float]:
        return {"osnr_db": float(np.mean(self._osnr))} if self._osnr else {}


@register_block("wdm_multiplexer", category="optical")
class WDMMultiplexer(Block):
    """Wavelength-division multiplexing: add neighbouring channels around the channel under test.

    The interferers carry independent random symbols with the same constellation, pulse
    shape and power as the input channel, at multiples of ``spacing_ghz``. Propagating the
    combined field through a nonlinear ``fiber`` / ``optical_link`` produces cross-phase
    modulation and four-wave mixing. The sample rate must cover all channels.
    """

    class Params(BlockParams):
        n_channels: int = Field(5, ge=1, le=21, description="Total number of channels (odd).")
        spacing_ghz: float = Field(50.0, gt=0)
        interferer_power_db: float = Field(0.0, description="Interferer power relative to the channel under test.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "wdm_multiplexer")
        if p.n_channels % 2 == 0:
            raise BlockError("Use an odd number of WDM channels (the channel under test is the centre one).")
        span = p.n_channels * p.spacing_ghz * 1e9
        if span > fs:
            raise BlockError(f"Sample rate {fs / 1e9:.0f} GHz cannot hold {p.n_channels} x {p.spacing_ghz} GHz: "
                             "raise the pulse shaper sps.")
        taps, sps = x.meta.get("pulse_taps"), x.meta.get("sps")
        const = x.meta.get("constellation")
        if taps is None or not sps or const is None:
            raise BlockError("wdm_multiplexer needs mapper + pulse_shaper metadata upstream.")
        e = x.data
        n = e.shape[-1]
        lead = e.shape[:-1]
        n_sym = int(x.meta.get("n_symbols", n // sps))
        power = np.mean(np.abs(e) ** 2, axis=-1, keepdims=True) * 10 ** (p.interferer_power_db / 10)
        t = np.arange(n) / fs
        out = e.copy()
        half = p.n_channels // 2
        for k in [k for k in range(-half, half + 1) if k]:
            sym = const.points[self.rng.integers(0, const.order, (*lead, n_sym))]
            up = np.zeros((*lead, n_sym * sps), dtype=np.complex128)
            up[..., ::sps] = sym
            wave = np.apply_along_axis(lambda r: np.convolve(r, taps), -1, up)[..., :n]
            if wave.shape[-1] < n:
                wave = np.concatenate([wave, np.zeros((*lead, n - wave.shape[-1]))], axis=-1)
            wave *= np.sqrt(power / np.maximum(np.mean(np.abs(wave) ** 2, axis=-1, keepdims=True), 1e-300))
            out += wave * np.exp(2j * np.pi * k * p.spacing_ghz * 1e9 * t)
        return {"out": x.evolve(out, wdm={"n_channels": p.n_channels, "spacing_hz": p.spacing_ghz * 1e9})}


@register_block("wdm_demultiplexer", category="optical")
class WDMDemultiplexer(Block):
    """Select one WDM channel with an optical band-pass filter (rectangular or 2nd-order super-Gaussian)."""

    class Params(BlockParams):
        channel: int = Field(0, description="Channel index relative to the centre (0 = channel under test).")
        bandwidth_ghz: float | None = Field(None, gt=0, description="Filter bandwidth (default: channel spacing).")
        shape: Literal["rect", "supergaussian"] = "supergaussian"

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "wdm_demultiplexer")
        wdm = x.meta.get("wdm") or {}
        spacing = wdm.get("spacing_hz", 50e9)
        bw = (self.params.bandwidth_ghz * 1e9) if self.params.bandwidth_ghz else spacing
        n = x.length
        shift = np.exp(-2j * np.pi * self.params.channel * spacing * np.arange(n) / fs)
        f = np.fft.fftfreq(n, 1 / fs)
        if self.params.shape == "rect":
            H = (np.abs(f) <= bw / 2).astype(float)
        else:
            H = np.exp(-np.log(2) / 2 * (2 * f / bw) ** 4)  # -3 dB at +-bw/2
        y = np.fft.ifft(np.fft.fft(x.data * shift, axis=-1) * H, axis=-1)
        meta = {k: v for k, v in x.meta.items() if k != "wdm"}
        return {"out": Signal(y, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("fso_turbulence", category="optical")
class FSOTurbulence(Block):
    """Free-space optical scintillation: log-normal (weak) or gamma-gamma (moderate/strong).

    The intensity gain h (E[h] = 1) is constant over a frame; its value is on ``csi``.
    """

    class Params(BlockParams):
        model: Literal["lognormal", "gamma_gamma"] = "lognormal"
        log_intensity_variance: float = Field(0.1, gt=0, description="Var(ln I) for the log-normal model.")
        alpha: float = Field(4.0, gt=0, description="Large-scale eddies (gamma-gamma).")
        beta: float = Field(2.0, gt=0, description="Small-scale eddies (gamma-gamma).")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES), "csi": Port(Kind.CSI)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        B = x.data.shape[0]
        if p.model == "lognormal":
            s2 = p.log_intensity_variance
            h = np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), B))
        else:
            h = self.rng.gamma(p.alpha, 1 / p.alpha, B) * self.rng.gamma(p.beta, 1 / p.beta, B)
        shape = (B,) + (1,) * (x.data.ndim - 1)
        gain = np.sqrt(h).reshape(shape)
        csi = Signal(np.sqrt(h)[:, None].astype(np.complex128), Kind.CSI, meta={**x.meta, "csi_type": "gain"})
        return {"out": x.evolve(x.data * gain), "csi": csi}


# ------------------------------------------------------------------ receivers
class _DetectorParams(BlockParams):
    responsivity: float = Field(0.8, gt=0, description="A/W.")
    thermal_noise_pa_rthz: float = Field(20.0, ge=0, description="Input-referred current noise density.")
    shot_noise: bool = True


@register_block("photodiode", category="optical")
class Photodiode(Block):
    """Direct detection (PIN or APD): square-law current with shot and thermal noise.

    Noise bandwidth is fs/2 (the bandwidth of the sampled signal). With ``apd_gain`` > 1
    the excess-noise factor F = k M + (1 - k)(2 - 1/M) applies to the shot noise.
    """

    class Params(_DetectorParams):
        dark_current_na: float = Field(5.0, ge=0)
        apd_gain: float = Field(1.0, ge=1.0)
        apd_k: float = Field(0.2, ge=0, le=1, description="APD ionization ratio.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Photocurrent (A), real.")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        bw = _require_fs(x, "photodiode") / 2
        power = np.sum(np.abs(x.data) ** 2, axis=1) if x.data.ndim == 3 else np.abs(x.data) ** 2
        M = p.apd_gain
        F = p.apd_k * M + (1 - p.apd_k) * (2 - 1 / M)
        i_primary = p.responsivity * power + p.dark_current_na * 1e-9
        var = (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw * np.ones_like(power)
        if p.shot_noise:
            var = var + 2 * Q_E * M ** 2 * F * i_primary * bw
        current = M * i_primary + np.sqrt(var) * self.rng.standard_normal(power.shape)
        meta = {**x.meta, "optical": False, "noise_var": float(np.mean(var)),
                "power": float(np.mean((M * i_primary) ** 2))}
        return {"out": Signal(current, Kind.SAMPLES, fs=x.fs, meta=meta)}


@register_block("coherent_receiver", category="optical")
class CoherentReceiver(Block):
    """Coherent front-end: local oscillator, 90-degree hybrid and balanced photodiodes.

    Output is the complex photocurrent proportional to E * conj(E_LO) per polarization,
    including LO frequency offset and phase noise, LO shot noise and thermal noise.

    Frames are processed with circular FFTs (fiber, dispersion compensation), so the LO
    phase is made periodic over the frame: the offset is rounded to the frame frequency
    grid (fs / n) and the phase-noise random walk is pinned at both ends (Brownian bridge).
    """

    class Params(_DetectorParams):
        lo_power_dbm: float = Field(13.0)
        lo_linewidth_hz: float = Field(100e3, ge=0)
        frequency_offset_hz: float = Field(0.0, description="Transmitter laser minus LO frequency.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "coherent_receiver")
        e = x.data
        B, n = e.shape[0], e.shape[-1]
        walk = _wiener_phase(self.rng, B, n + 1, p.lo_linewidth_hz, fs)
        bridge = walk[:, :n] - np.arange(n) / n * walk[:, n:n + 1]
        fo = np.round(p.frequency_offset_hz * n / fs) * fs / n
        phase = bridge + 2 * np.pi * fo * np.arange(n) / fs
        lo = np.exp(-1j * (phase[:, None, :] if e.ndim == 3 else phase))
        p_lo = dbm_to_w(p.lo_power_dbm)
        scale = p.responsivity * np.sqrt(p_lo)
        bw = fs / 2
        var = 2 * (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw
        if p.shot_noise:
            var += 2 * Q_E * p.responsivity * p_lo * bw
        noise = np.sqrt(var / 2) * (self.rng.standard_normal(e.shape) + 1j * self.rng.standard_normal(e.shape))
        out = scale * e * lo + noise
        npol = e.shape[1] if e.ndim == 3 else 1
        sig_power = scale ** 2 * (x.meta.get("power") or 1e-3) / npol
        meta = {**x.meta, "optical": False, "noise_var": var, "power": sig_power}
        return {"out": Signal(out, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("agc", category="receiver")
class AGC(Block):
    """Automatic gain control: remove the mean (optional) and scale every frame and stream
    to the nominal symbol power (``power`` metadata of the upstream mapper / line code).

    The known noise variance is subtracted when estimating the signal power, so decision
    thresholds stay correct at low SNR.
    """

    class Params(BlockParams):
        remove_dc: bool = True
        target_power: float | None = Field(None, gt=0, description="Override the target power.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        y = x.data
        if self.params.remove_dc:
            y = y - np.mean(y, axis=-1, keepdims=True)
        target = self.params.target_power or 1.0
        const = x.meta.get("constellation")
        if self.params.target_power is None and x.meta.get("line_code") is not None:
            target = {"nrz": 1.0, "manchester": 1.0, "ami": 0.5, "mlt3": 0.5}[x.meta["line_code"]]
        elif self.params.target_power is None and const is not None:
            target = float(np.mean(np.abs(const.points) ** 2))
        var = np.mean(np.abs(y) ** 2, axis=-1, keepdims=True)
        nv = x.meta.get("noise_var")
        sig = np.maximum(var - (float(np.mean(nv)) if nv is not None else 0.0), 0.05 * var)
        g = np.sqrt(target / np.maximum(sig, 1e-300))
        meta = {"power": target}
        if nv is not None:
            meta["noise_var"] = np.mean(nv) * np.mean(g ** 2)
        return {"out": x.evolve(y * g, **meta)}


# --------------------------------------------------------------- coherent DSP
@register_block("cd_compensation", category="optical")
class CDCompensation(Block):
    """Frequency-domain chromatic dispersion compensation (inverse all-pass filter).

    The accumulated dispersion is read from the ``cd_ps_nm`` metadata unless given.
    """

    class Params(BlockParams):
        dispersion_ps_nm: float | None = Field(None, description="Total D*L to remove (ps/nm).")
        wavelength_nm: float | None = Field(None, gt=0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "cd_compensation")
        cd = self.params.dispersion_ps_nm if self.params.dispersion_ps_nm is not None else x.meta.get("cd_ps_nm", 0.0)
        lam = self.params.wavelength_nm or x.meta.get("wavelength_nm", 1550.0)
        H = np.conj(dispersion_operator(x.length, fs, cd, lam))
        y = np.fft.ifft(np.fft.fft(x.data, axis=-1) * H, axis=-1)
        return {"out": x.evolve(y, cd_ps_nm=x.meta.get("cd_ps_nm", 0.0) - cd,
                                cd_compensated_ps_nm=x.meta.get("cd_compensated_ps_nm", 0.0) + cd)}


@register_block("fo_compensation", category="sync")
class FOCompensation(Block):
    """Blind frequency-offset estimation from the spectral peak of the M-th power
    (M = 4 for QPSK / square QAM), joint over polarizations, and correction.

    The estimate uses a low-pass filtered copy (|f| < ``estimation_bandwidth`` x symbol
    rate) to reject out-of-band noise. If chromatic dispersion was compensated before
    this block, the bulk delay that the offset causes (beta2 L delta_omega) is removed too.
    """

    class Params(BlockParams):
        power: int = Field(4, ge=1, le=8)
        estimation_bandwidth: float = Field(0.75, gt=0, description="Low-pass for the estimate (x symbol rate).")

    inputs = {"in": Port(Kind.SAMPLES | Kind.SYMBOLS)}
    outputs = {"out": Port.like("in")}

    def on_reset(self) -> None:
        self._estimates: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "fo_compensation")
        n = x.length
        m = self.params.power
        rs = fs / x.meta.get("sps", 1)
        f_n = np.fft.fftfreq(n, 1 / fs)
        lp = np.fft.ifft(np.fft.fft(x.data, axis=-1) * (np.abs(f_n) < self.params.estimation_bandwidth * rs), axis=-1)
        if lp.ndim == 3 and lp.shape[1] == 2:
            # A polarization mix a*s1 + b*s2 has an M-th power line ~ a^M + b^M that can
            # vanish; extra linear combinations guarantee a usable line.
            x0, x1 = lp[:, 0], lp[:, 1]
            lp = np.stack([x0, x1, (x0 + x1) / np.sqrt(2), (x0 + 1j * x1) / np.sqrt(2)], axis=1)
        nfft = int(2 ** np.ceil(np.log2(4 * n)))
        spec = np.abs(np.fft.fft(lp ** m, n=nfft, axis=-1)) ** 2
        if spec.ndim == 3:
            spec = spec.sum(axis=1)
        f = np.fft.fftfreq(nfft, 1 / fs)
        df = f[np.argmax(spec, axis=-1)] / m  # (B,)
        self._estimates.extend(df.tolist())
        rot = np.exp(-2j * np.pi * df[:, None] * np.arange(n) / fs)
        y = x.data * (rot[:, None, :] if x.data.ndim == 3 else rot)
        cd = x.meta.get("cd_compensated_ps_nm", 0.0)
        if cd:  # residual of dispersion compensation applied to a frequency-shifted spectrum
            b2l = beta2(cd, x.meta.get("wavelength_nm", 1550.0)) * 1e3
            w = 2 * np.pi * f_n
            dw = 2 * np.pi * df[:, None]
            fix = np.exp(1j * b2l / 2 * (2 * w[None, :] * dw + dw ** 2))
            y = np.fft.ifft(np.fft.fft(y, axis=-1) * (fix[:, None, :] if y.ndim == 3 else fix), axis=-1)
        return {"out": x.evolve(y)}

    def metrics(self) -> dict[str, float]:
        return {"fo_estimate_hz": float(np.mean(self._estimates))} if self._estimates else {}


@register_block("adaptive_mimo_equalizer", category="receiver")
class AdaptiveMIMOEqualizer(Block):
    """Fractionally spaced 2x2 (NxN) butterfly FIR equalizer for coherent receivers.

    Undoes polarization rotation, PMD and residual dispersion. Adaptation: ``cma``
    (blind constant modulus) or ``lms`` (trained with ``ref`` for ``n_train`` symbols,
    then decision directed). Input: ``sps`` samples per symbol starting at the
    ``delay`` metadata (e.g. after ``matched_filter`` with ``decimate: false``).
    """

    class Params(BlockParams):
        n_taps: int = Field(15, ge=1, description="Taps per filter (odd).")
        step_size: float = Field(1e-3, gt=0, description="Step size in tracking (DD / CMA) mode.")
        train_step_size: float = Field(1e-2, gt=0, description="Larger step during LMS training (fast acquisition).")
        algorithm: Literal["cma", "lms"] = "lms"
        n_train: int = Field(512, ge=0)
        sps: int = Field(2, ge=1)

    inputs = {"in": Port(Kind.SAMPLES), "ref": Port(Kind.SYMBOLS, required=False)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        ref = inputs.get("ref")
        p = self.params
        const: Constellation | None = x.meta.get("constellation")
        data = x.data if x.data.ndim == 3 else x.data[:, None, :]
        B, P, N = data.shape
        sps = x.meta.get("sps", p.sps)
        if sps != p.sps:
            raise BlockError(f"Input has {sps} samples per symbol, expected {p.sps}.")
        start = int(round(x.meta.get("delay", 0.0)))
        n_sym = int(x.meta.get("n_symbols", (N - start) // sps))
        n_sym = min(n_sym, (N - start - 1) // sps + 1)
        T = p.n_taps | 1
        D = T // 2
        pad = np.concatenate([np.zeros((B, P, D)), data, np.zeros((B, P, D + sps))], axis=-1)
        W = np.zeros((B, P, P, T), dtype=np.complex128)
        for q in range(P):
            W[:, q, q, D] = 1.0
        refd = None
        if ref is not None:
            refd = ref.data if ref.data.ndim == 3 else ref.data[:, None, :]
        use_lms = p.algorithm == "lms" and refd is not None
        if p.algorithm == "lms" and refd is None and p.n_train:
            raise BlockError("LMS training needs the 'ref' input (transmitted symbols).")
        radius = float(np.mean(np.abs(const.points) ** 4) / np.mean(np.abs(const.points) ** 2)) if const else 1.0
        out = np.empty((B, P, n_sym), dtype=np.complex128)
        err, count = 0.0, 0
        for k in range(n_sym):
            c = start + k * sps + D  # centre index in the padded signal
            u = pad[:, :, c - D:c + D + 1][:, :, ::-1]  # (B, P, T)
            y = np.einsum("bpqt,bqt->bp", W, u)
            out[:, :, k] = y
            if not use_lms:  # blind: phase-insensitive, leaves the carrier phase to phase_recovery
                e = y * (radius - np.abs(y) ** 2)
            elif k < p.n_train:
                e = refd[:, :, k] - y
            else:
                e = const.points[const.nearest(y)] - y if const is not None else refd[:, :, k] - y
                err += float(np.sum(np.abs(e) ** 2))
                count += e.size
            mu = p.train_step_size if use_lms and k < p.n_train else p.step_size
            W += mu * e[:, :, None, None] * np.conj(u)[:, None, :, :]
        out = out if x.data.ndim == 3 else out[:, 0]
        meta = {"sps": 1, "delay": 0.0, "n_symbols": n_sym, "noise_var": err / count if count else None}
        return {"out": Signal(out, Kind.SYMBOLS, fs=x.fs / sps if x.fs else None, meta={**x.meta, **meta})}


@register_block("phase_recovery", category="sync")
class PhaseRecovery(Block):
    """Feed-forward carrier phase recovery for large laser linewidths.

    * ``bps``: blind phase search (any QAM) over ``n_test_phases`` in [0, pi/2),
    * ``viterbi_viterbi``: 4th-power estimator (QPSK).

    Estimates are averaged over 2 * ``half_window`` + 1 symbols and unwrapped. The
    quadrant ambiguity is resolved with ``n_ref`` known symbols of ``ref`` starting at
    ``ref_start`` (after the equalizer has converged), like a pilot block.
    """

    class Params(BlockParams):
        method: Literal["bps", "viterbi_viterbi"] = "bps"
        n_test_phases: int = Field(32, ge=4)
        half_window: int = Field(16, ge=1)
        n_ref: int = Field(64, ge=1)
        ref_start: int = Field(600, ge=0, description="First reference symbol used for the ambiguity.")

    inputs = {"in": Port(Kind.SYMBOLS), "ref": Port(Kind.SYMBOLS, required=False)}
    outputs = {"out": Port(Kind.SYMBOLS)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        const: Constellation | None = x.meta.get("constellation")
        p = self.params
        y = x.data
        kernel = np.ones(2 * p.half_window + 1)

        def smooth(a: np.ndarray) -> np.ndarray:
            flat = a.reshape(-1, a.shape[-1])
            sm = np.stack([np.convolve(r, kernel, mode="same") for r in flat])
            return sm.reshape(a.shape)

        if p.method == "viterbi_viterbi":
            s = smooth(y ** 4)
            phi = (np.angle(s) - np.pi) / 4
        else:
            if const is None:
                raise BlockError("BPS needs the constellation (mapper upstream).")
            phases = np.arange(p.n_test_phases) / p.n_test_phases * np.pi / 2 - np.pi / 4
            dist = np.empty((*y.shape, phases.size))
            for i, ph in enumerate(phases):
                r = y * np.exp(-1j * ph)
                dist[..., i] = np.abs(r - const.points[const.nearest(r)]) ** 2
            dist = np.moveaxis(smooth(np.moveaxis(dist, -1, 0)), 0, -1)
            phi = phases[np.argmin(dist, axis=-1)]
        phi = np.unwrap(4 * phi, axis=-1) / 4
        out = y * np.exp(-1j * phi)
        ref = inputs.get("ref")
        if ref is not None:
            a = min(p.ref_start, max(out.shape[-1] - p.n_ref, 0))
            win = slice(a, min(a + p.n_ref, out.shape[-1], ref.data.shape[-1]))
            rots = np.array([np.sum(np.abs(out[..., win] * 1j ** k - ref.data[..., win]) ** 2, axis=-1)
                             for k in range(4)])
            best = np.argmin(rots, axis=0)
            out = out * (1j ** best)[..., None]
        return {"out": x.evolve(out)}


@register_block("optical_power_meter", category="metrics")
class OpticalPowerMeter(Block):
    """Average optical power (dBm) and OSNR (0.1 nm) of an optical signal."""

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {}

    def on_reset(self) -> None:
        self._p: list[float] = []
        self._osnr: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        e = x.data
        power = np.sum(np.abs(e) ** 2, axis=1) if e.ndim == 3 else np.abs(e) ** 2
        self._p.append(float(np.mean(power)))
        o = osnr_db(x.meta)
        if o is not None:
            self._osnr.append(o)
        return {}

    def metrics(self) -> dict[str, float]:
        if not self._p:
            return {}
        out = {"power_dbm": w_to_dbm(float(np.mean(self._p)))}
        if self._osnr:
            out["osnr_db"] = float(np.mean(self._osnr))
        return out
