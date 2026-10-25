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
from scipy.special import erf

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation
from linkstudio.phy.dsp.filters import apply_frequency_response, bessel_thomson, dc_group_delay

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


def intensity_noise(rng: np.random.Generator, power: np.ndarray, p_avg: float, rin_db_hz: float | None,
                    fs: float) -> np.ndarray:
    """Add relative intensity noise: Gaussian with variance RIN x P_avg^2 x fs/2 (noise bandwidth fs/2)."""
    if rin_db_hz is None:
        return power
    return power + p_avg * np.sqrt(10 ** (rin_db_hz / 10) * fs / 2) * rng.standard_normal(power.shape)


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
def laser_response(f: np.ndarray, resonance_hz: float, damping_per_s: float, parasitic_hz: float | None) -> np.ndarray:
    """Small-signal modulation response of a semiconductor laser with a parasitic pole.

    H(f) = f_r^2 / (f_r^2 - f^2 + j f gamma / (2 pi)) x 1 / (1 + j f / f_p), from the
    linearized rate equations (resonance f_r, damping rate gamma) and the RC parasitics
    of the chip (Coldren, Corzine & Mashanovitch, ch. 5).
    """
    h = resonance_hz ** 2 / (resonance_hz ** 2 - f ** 2 + 1j * f * damping_per_s / (2 * np.pi))
    if parasitic_hz:
        h = h / (1 + 1j * f / parasitic_hz)
    return h


@register_block("intensity_modulator", category="optical")
class IntensityModulator(Block):
    """Laser with intensity modulation (directly modulated laser / EML) for IM/DD links.

    The real drive (NRZ, PAM-4, DMT...) is normalized to [-1, 1] and mapped to the optical
    power P = P_avg (1 + m x) with modulation depth m = (ER - 1) / (ER + 1) for extinction
    ratio ER. Optional: laser modulation bandwidth (``resonance_ghz``, small-signal
    rate-equation response), relative intensity noise, laser phase noise and the chirp of a
    directly modulated laser (Koch & Bowers): instantaneous frequency
    alpha/(4 pi) d(ln P)/dt (transient) + kappa_ad (P - P_avg) (adiabatic).
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Average optical power.")
        extinction_ratio_db: float = Field(10.0, gt=0)
        rin_db_hz: float | None = Field(None, description="Relative intensity noise (e.g. -150 dB/Hz).")
        chirp_alpha: float = Field(0.0, description="Henry factor of a directly modulated laser.")
        adiabatic_chirp_ghz_mw: float = Field(0.0, description="Adiabatic chirp: frequency shift per mW above P_avg.")
        resonance_ghz: float | None = Field(None, gt=0, description="Relaxation-oscillation frequency (None = ideal).")
        damping_per_ns: float = Field(50.0, gt=0, description="Damping rate gamma (1/ns).")
        parasitic_ghz: float | None = Field(None, gt=0, description="Parasitic RC pole.")
        linewidth_hz: float = Field(0.0, ge=0)
        wavelength_nm: float = Field(1550.0, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Modulation response (drive to optical power); flat without laser dynamics."""
        p = self.params
        if p.resonance_ghz is None:
            return np.ones(np.shape(f), dtype=complex)
        return laser_response(f, p.resonance_ghz * 1e9, p.damping_per_ns * 1e9,
                              p.parasitic_ghz * 1e9 if p.parasitic_ghz else None)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "intensity_modulator")
        drive = np.real(x.data)
        peak = np.max(np.abs(drive)) or 1.0
        drive = np.clip(drive / peak, -1, 1)
        delay = x.meta.get("delay", 0.0)
        if p.resonance_ghz is not None:
            def h(f: np.ndarray) -> np.ndarray:
                return laser_response(f, p.resonance_ghz * 1e9, p.damping_per_ns * 1e9,
                                      p.parasitic_ghz * 1e9 if p.parasitic_ghz else None)
            drive = apply_frequency_response(drive, h, fs, pad=256)
            delay += dc_group_delay(h, fs) * fs
        er = 10 ** (p.extinction_ratio_db / 10)
        m = (er - 1) / (er + 1)
        p_avg = dbm_to_w(p.power_dbm)
        power = p_avg * (1 + m * drive)
        power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, fs), 0.0)
        phase = _wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
        if p.chirp_alpha:
            phase = phase + p.chirp_alpha / 2 * np.log(np.maximum(power, 1e-12 * p_avg) / p_avg)
        if p.adiabatic_chirp_ghz_mw:
            kappa = p.adiabatic_chirp_ghz_mw * 1e9 / 1e-3  # Hz per W
            phase = phase + 2 * np.pi * kappa * np.cumsum(power - p_avg, axis=-1) / fs
        field = np.sqrt(power) * np.exp(1j * phase)
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg,
                "cd_ps_nm": 0.0, "ase_psd": 0.0, "delay": delay, "unit": "√W"}
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
                "cd_ps_nm": 0.0, "ase_psd": 0.0, "unit": "√W"}
        return {"out": x.evolve(field, kind=Kind.SAMPLES, **meta)}


# ------------------------------------------------------------------- fiber
def beta2(dispersion_ps_nm_km: float, wavelength_nm: float) -> float:
    """Group-velocity dispersion beta2 in s^2/m."""
    D = dispersion_ps_nm_km * 1e-6  # s/m^2
    lam = wavelength_nm * 1e-9
    return -D * lam ** 2 / (2 * np.pi * C_LIGHT)


def beta3(dispersion_ps_nm_km: float, slope_ps_nm2_km: float, wavelength_nm: float) -> float:
    """Third-order dispersion beta3 in s^3/m from D and the dispersion slope S = dD/dlambda."""
    D = dispersion_ps_nm_km * 1e-6  # s/m^2
    S = slope_ps_nm2_km * 1e3  # ps/(nm^2 km) -> s/m^3
    lam = wavelength_nm * 1e-9
    return (lam / (2 * np.pi * C_LIGHT)) ** 2 * (lam ** 2 * S + 2 * lam * D)


def dispersion_operator(n: int, fs: float, cd_ps_nm: float, wavelength_nm: float,
                        slope_ps_nm2: float = 0.0) -> np.ndarray:
    """All-pass dispersion filter for an accumulated D*L (ps/nm) and slope S*L (ps/nm^2).

    exp(j (beta2 L w^2 / 2 - beta3 L w^3 / 6)) on the NumPy frequency grid. The envelope
    follows Agrawal's convention (carrier exp(-j w0 t)), so a positive NumPy frequency is a
    lower optical frequency; hence the minus sign of the odd beta3 term.
    """
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
    b2l = beta2(cd_ps_nm, wavelength_nm) * 1e3  # D*L in ps/nm == (ps/nm/km) * km
    b3l = beta3(cd_ps_nm, slope_ps_nm2, wavelength_nm) * 1e3
    return np.exp(1j * (b2l / 2 * w ** 2 - b3l / 6 * w ** 3))


def _random_unitary(rng: np.random.Generator, batch: int) -> np.ndarray:
    a = rng.standard_normal((batch, 2, 2)) + 1j * rng.standard_normal((batch, 2, 2))
    q, r = np.linalg.qr(a)
    return q * (np.diagonal(r, axis1=1, axis2=2) / np.abs(np.diagonal(r, axis1=1, axis2=2)))[:, None, :]


def propagate(e: np.ndarray, fs: float, length_km: float, att_db_km: float, d_ps_nm_km: float,
              wavelength_nm: float, nonlinear: bool, gamma: float, steps_per_km: float,
              slope_ps_nm2_km: float = 0.0, nonlinear_phase_rad: float | None = None) -> np.ndarray:
    """Split-step Fourier solution of the NLSE (Manakov equation for dual polarization).

    The step is at most 1 / ``steps_per_km`` km. With ``nonlinear_phase_rad`` it is also
    limited so that the peak nonlinear phase per step, gamma P_peak L_eff, stays below that
    value (nonlinear-phase-rotation criterion of Sinkin et al. 2003).
    """
    n = e.shape[-1]
    alpha = att_db_km / (10 * np.log10(np.e)) / 1e3  # 1/m
    if not nonlinear or gamma == 0:
        lin = dispersion_operator(n, fs, d_ps_nm_km * length_km, wavelength_nm, slope_ps_nm2_km * length_km)
        return np.fft.ifft(np.fft.fft(e, axis=-1) * lin * np.exp(-alpha * length_km * 1e3 / 2), axis=-1)
    total = length_km * 1e3
    max_dz = total / max(1, int(np.ceil(length_km * steps_per_km)))
    g = gamma * 1e-3 * (8 / 9 if e.ndim == 3 else 1.0)  # 1/(W km) -> 1/(W m)
    z = 0.0
    while z < total * (1 - 1e-12):
        dz = min(max_dz, total - z)
        if nonlinear_phase_rad:
            p_peak = float(np.max(np.sum(np.abs(e) ** 2, axis=1) if e.ndim == 3 else np.abs(e) ** 2))
            dz = min(dz, max(nonlinear_phase_rad / max(g * p_peak, 1e-30), 1.0))
        half = dispersion_operator(n, fs, d_ps_nm_km * dz / 2e3, wavelength_nm, slope_ps_nm2_km * dz / 2e3) \
            * np.exp(-alpha * dz / 4)
        # Effective length around the step midpoint (where the nonlinear phase is applied).
        leff = 2 * np.sinh(alpha * dz / 2) / alpha if alpha > 0 else dz
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
        p = np.sum(np.abs(e) ** 2, axis=1, keepdims=True) if e.ndim == 3 else np.abs(e) ** 2
        e = e * np.exp(1j * g * p * leff)
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
        z += dz
    return e


def pmd_transfer(rng: np.random.Generator, batch: int, w: np.ndarray, mean_dgd_s: float,
                 n_sections: int) -> np.ndarray:
    """Jones matrices T(w), shape (batch, n, 2, 2), of a fiber with random birefringence.

    Coarse-step (waveplate) model: ``n_sections`` birefringent sections with random
    orientation, each with DGD mean_dgd sqrt(3 pi / (8 N)), so that the total DGD has the
    Maxwellian statistics of real fiber (mean ``mean_dgd_s``) and higher-order PMD.
    """
    t = np.broadcast_to(_random_unitary(rng, batch)[:, None], (batch, w.size, 2, 2)).copy()
    if mean_dgd_s <= 0 or n_sections == 0:
        return t
    dtau = mean_dgd_s * np.sqrt(3 * np.pi / (8 * n_sections))
    d = np.zeros((w.size, 2, 2), dtype=np.complex128)
    d[:, 0, 0] = np.exp(1j * w * dtau / 2)
    d[:, 1, 1] = np.exp(-1j * w * dtau / 2)
    for _ in range(n_sections):
        u = _random_unitary(rng, batch)
        t = np.einsum("nij,bjk,bnkl->bnil", d, u, t)
    return np.einsum("bij,bnjk->bnik", _random_unitary(rng, batch), t)


def apply_pmd(rng: np.random.Generator, e: np.ndarray, fs: float, mean_dgd_s: float,
              n_sections: int = 10) -> np.ndarray:
    """Random polarization evolution and PMD (waveplate model of :func:`pmd_transfer`)."""
    B, _, n = e.shape
    t = pmd_transfer(rng, B, 2 * np.pi * np.fft.fftfreq(n, 1 / fs), mean_dgd_s, n_sections)
    return np.fft.ifft(np.einsum("bnij,bjn->bin", t, np.fft.fft(e, axis=-1)), axis=-1)


class _FiberParams(BlockParams):
    attenuation_db_km: float = Field(0.2, ge=0)
    dispersion_ps_nm_km: float = Field(17.0, description="Chromatic dispersion parameter D.")
    nonlinear: bool = Field(False, description="Kerr nonlinearity (split-step Fourier).")
    dispersion_slope_ps_nm2_km: float = Field(0.0, description="Dispersion slope S (G.652 at 1550 nm: ~0.057).")
    gamma_w_km: float = Field(1.3, ge=0, description="Nonlinear coefficient (1/W/km).")
    steps_per_km: float = Field(0.5, gt=0, description="Minimum split-step resolution (longest step 1/steps_per_km).")
    nonlinear_phase_rad: float | None = Field(0.05, gt=0, description="Largest peak nonlinear phase per step.")
    pmd_ps_sqrt_km: float = Field(0.0, ge=0, description="PMD coefficient (dual polarization).")
    pmd_sections: int = Field(10, ge=1, le=200, description="Birefringent sections per span (waveplate model).")
    polarization_rotation: bool = Field(False, description="Random polarization state per frame.")


@register_block("fiber", category="optical")
class Fiber(Block):
    """Single-mode fiber span: attenuation, chromatic dispersion, optional Kerr nonlinearity and PMD."""

    class Params(_FiberParams):
        length_km: float = Field(80.0, ge=0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}
    two_sided_response = True

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Linear field transfer function: attenuation and dispersion (beta2, beta3)."""
        p = self.params
        lam = x.meta.get("wavelength_nm", 1550.0)
        cd = p.dispersion_ps_nm_km * p.length_km
        b2l = beta2(cd, lam) * 1e3
        b3l = beta3(cd, p.dispersion_slope_ps_nm2_km * p.length_km, lam) * 1e3
        w = 2 * np.pi * np.asarray(f, dtype=float)
        loss = 10 ** (-p.attenuation_db_km * p.length_km / 20)
        return loss * np.exp(1j * (b2l / 2 * w ** 2 - b3l / 6 * w ** 3))

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "fiber")
        lam = x.meta.get("wavelength_nm", 1550.0)
        e = propagate(x.data, fs, p.length_km, p.attenuation_db_km, p.dispersion_ps_nm_km, lam,
                      p.nonlinear, p.gamma_w_km, p.steps_per_km, p.dispersion_slope_ps_nm2_km, p.nonlinear_phase_rad)
        if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
            e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.length_km), p.pmd_sections)
        loss = 10 ** (-p.attenuation_db_km * p.length_km / 10)
        return {"out": x.evolve(e, power=(x.meta.get("power") or 1e-3) * loss,
                                ase_psd=x.meta.get("ase_psd", 0.0) * loss,
                                cd_ps_nm=x.meta.get("cd_ps_nm", 0.0) + p.dispersion_ps_nm_km * p.length_km,
                                cd_slope_ps_nm2=x.meta.get("cd_slope_ps_nm2", 0.0)
                                + p.dispersion_slope_ps_nm2_km * p.length_km)}


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
                          p.nonlinear, p.gamma_w_km, p.steps_per_km, p.dispersion_slope_ps_nm2_km,
                          p.nonlinear_phase_rad)
            if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
                e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.span_km), p.pmd_sections)
            e = _add_ase(self.rng, e * np.sqrt(g), s, fs)
            ase = ase + s  # the span loss and the amplifier gain cancel
        meta = {"power": power, "ase_psd": ase,
                "cd_ps_nm": x.meta.get("cd_ps_nm", 0.0) + p.dispersion_ps_nm_km * p.span_km * p.n_spans,
                "cd_slope_ps_nm2": x.meta.get("cd_slope_ps_nm2", 0.0)
                + p.dispersion_slope_ps_nm2_km * p.span_km * p.n_spans}
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


def rytov_variance(cn2: float, wavelength_nm: float, distance_m: float) -> float:
    """Plane-wave Rytov variance sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)."""
    k = 2 * np.pi / (wavelength_nm * 1e-9)
    return 1.23 * cn2 * k ** (7 / 6) * distance_m ** (11 / 6)


def gamma_gamma_parameters(sigma_r2: float) -> tuple[float, float]:
    """Large- and small-scale parameters (alpha, beta) of the gamma-gamma model, plane wave
    (Al-Habash, Andrews & Phillips 2001)."""
    s125 = sigma_r2 ** 1.2  # sigma_R^(12/5)
    alpha = 1 / np.expm1(0.49 * sigma_r2 / (1 + 1.11 * s125) ** (7 / 6))
    beta = 1 / np.expm1(0.51 * sigma_r2 / (1 + 0.69 * s125) ** (5 / 6))
    return float(alpha), float(beta)


def kim_attenuation_db_km(visibility_km: float, wavelength_nm: float) -> float:
    """Specific attenuation of haze and fog from the visibility (ITU-R P.1814-1, eqs. 8-9).

    gamma = 17 / V (0.55 um / lambda)^q dB/km with Kim's size-distribution exponent q;
    V is the visibility for the 2 % contrast threshold (1.31 x the WMO 5 % visibility).
    """
    v = visibility_km
    q = 1.6 if v > 50 else 1.3 if v > 6 else 0.16 * v + 0.34 if v >= 1 else v - 0.5 if v >= 0.5 else 0.0
    return 17 / v * (550 / wavelength_nm) ** q


def geometric_loss_db(distance_m: float, divergence_mrad: float, rx_aperture_m: float) -> float:
    """Beam-spreading loss 10 log10(S_d / S_capture), S_d = pi/4 (d theta)^2 (ITU-R P.1814-1, eq. 2)."""
    beam = distance_m * divergence_mrad * 1e-3
    return max(0.0, 20 * np.log10(beam / rx_aperture_m))


@register_block("fso_turbulence", category="optical")
class FSOTurbulence(Block):
    """Free-space optical scintillation: log-normal (weak) or gamma-gamma (moderate/strong).

    The intensity gain h (E[h] = 1) is constant over a frame; its value is on ``csi``. With
    ``rytov_variance`` set, the model parameters follow from the turbulence strength: alpha
    and beta of Al-Habash et al., and Var(ln I) = ln(1 + sigma_I^2) with the same
    scintillation index sigma_I^2 = 1/alpha + 1/beta + 1/(alpha beta).
    """

    class Params(BlockParams):
        model: Literal["lognormal", "gamma_gamma"] = "lognormal"
        log_intensity_variance: float = Field(0.1, gt=0, description="Var(ln I) for the log-normal model.")
        alpha: float = Field(4.0, gt=0, description="Large-scale eddies (gamma-gamma).")
        beta: float = Field(2.0, gt=0, description="Small-scale eddies (gamma-gamma).")
        rytov_variance: float | None = Field(None, gt=0, description="Derive the parameters from sigma_R^2.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES), "csi": Port(Kind.CSI)}

    def model_parameters(self) -> tuple[float, float, float]:
        """(Var(ln I), alpha, beta) in use."""
        p = self.params
        if p.rytov_variance is None:
            return p.log_intensity_variance, p.alpha, p.beta
        a, b = gamma_gamma_parameters(p.rytov_variance)
        return float(np.log1p(1 / a + 1 / b + 1 / (a * b))), a, b

    def draw(self, n: int) -> np.ndarray:
        s2, a, b = self.model_parameters()
        if self.params.model == "lognormal":
            return np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), n))
        return self.rng.gamma(a, 1 / a, n) * self.rng.gamma(b, 1 / b, n)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        B = x.data.shape[0]
        h = self.draw(B)
        shape = (B,) + (1,) * (x.data.ndim - 1)
        gain = np.sqrt(h).reshape(shape)
        csi = Signal(np.sqrt(h)[:, None].astype(np.complex128), Kind.CSI, meta={**x.meta, "csi_type": "gain"})
        return {"out": x.evolve(x.data * gain), "csi": csi}


@register_block("fso_channel", category="optical")
class FSOChannel(Block):
    """Terrestrial free-space optical link from geometry and weather.

    * Atmospheric attenuation from the visibility (Kim model, ITU-R P.1814-1).
    * Geometric (beam-spreading) loss for a full divergence angle and receiver aperture.
    * Scintillation from Cn^2 through the Rytov variance (gamma-gamma or log-normal).
    * Pointing errors (Farid & Hranilovic 2007): radial jitter r ~ Rayleigh(sigma_s) gives
      h_p = exp(-2 r^2 / w_zeq^2), with the equivalent beam width of a Gaussian beam of
      radius w_z = d theta / 2 on an aperture of radius a.

    The fading terms are constant over a frame (quasi-static) and appear on ``csi``.
    """

    class Params(BlockParams):
        distance_m: float = Field(1000.0, gt=0)
        visibility_km: float | None = Field(None, gt=0, description="Visibility (2 % threshold); None = clear air.")
        divergence_mrad: float = Field(2.0, gt=0, description="Full beam divergence angle.")
        rx_aperture_m: float = Field(0.08, gt=0, description="Receiver aperture diameter.")
        cn2: float = Field(1e-14, ge=0, description="Refractive-index structure parameter (m^-2/3).")
        turbulence: Literal["gamma_gamma", "lognormal", "none"] = "gamma_gamma"
        pointing_jitter_m: float = Field(0.0, ge=0, description="RMS jitter per axis at the receiver.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES), "csi": Port(Kind.CSI)}

    def on_reset(self) -> None:
        self._hp: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        lam = x.meta.get("wavelength_nm", 1550.0)
        B = x.data.shape[0]
        atm = kim_attenuation_db_km(p.visibility_km, lam) * p.distance_m / 1e3 if p.visibility_km else 0.0
        loss = 10 ** (-(atm + geometric_loss_db(p.distance_m, p.divergence_mrad, p.rx_aperture_m)) / 10)
        h = np.ones(B)
        s2r = rytov_variance(p.cn2, lam, p.distance_m)
        if p.turbulence != "none" and s2r > 0:
            a, b = gamma_gamma_parameters(s2r)
            if p.turbulence == "gamma_gamma":
                h = self.rng.gamma(a, 1 / a, B) * self.rng.gamma(b, 1 / b, B)
            else:
                s2 = np.log1p(1 / a + 1 / b + 1 / (a * b))
                h = np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), B))
        if p.pointing_jitter_m > 0:
            wz = p.distance_m * p.divergence_mrad * 1e-3 / 2
            v = np.sqrt(np.pi / 2) * (p.rx_aperture_m / 2) / wz
            wzeq2 = wz ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2))
            r = p.pointing_jitter_m * np.sqrt(-2 * np.log(1 - self.rng.random(B)))
            hp = np.exp(-2 * r ** 2 / wzeq2)
            self._hp.extend(hp.tolist())
            h = h * hp
        shape = (B,) + (1,) * (x.data.ndim - 1)
        out = x.data * np.sqrt(loss) * np.sqrt(h).reshape(shape)
        csi = Signal(np.sqrt(h)[:, None].astype(np.complex128), Kind.CSI, meta={**x.meta, "csi_type": "gain"})
        self._last = {"atmospheric_loss_db": atm, "geometric_loss_db": -10 * np.log10(loss) - atm,
                      "rytov_variance": s2r}
        return {"out": x.evolve(out, power=(x.meta.get("power") or 1e-3) * loss), "csi": csi}

    def metrics(self) -> dict[str, float]:
        if not hasattr(self, "_last"):
            return {}
        m = dict(self._last)
        if self._hp:
            m["pointing_loss_db"] = -10 * float(np.log10(np.mean(self._hp)))
        return m


# ------------------------------------------------------------------ receivers
class _DetectorParams(BlockParams):
    responsivity: float = Field(0.8, gt=0, description="A/W.")
    thermal_noise_pa_rthz: float = Field(20.0, ge=0, description="Input-referred current noise density.")
    shot_noise: bool = True


@register_block("photodiode", category="optical")
class Photodiode(Block):
    """Direct detection (PIN or APD): square-law current with shot and thermal noise.

    Noise is generated over the simulation bandwidth fs/2. With ``apd_gain`` > 1 the
    excess-noise factor F = k M + (1 - k)(2 - 1/M) applies to the shot noise. An optional
    ``bandwidth_ghz`` applies a 4th-order Bessel-Thomson receiver response (signal and
    noise), the reference receiver of optical Ethernet specifications.
    """

    class Params(_DetectorParams):
        dark_current_na: float = Field(5.0, ge=0)
        apd_gain: float = Field(1.0, ge=1.0)
        apd_k: float = Field(0.2, ge=0, le=1, description="APD ionization ratio.")
        bandwidth_ghz: float | None = Field(None, gt=0, description="Receiver 3-dB bandwidth (None = fs/2).")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Photocurrent (A), real.")}

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Receiver bandwidth (Bessel-Thomson); flat when no bandwidth is set."""
        bw = self.params.bandwidth_ghz
        return bessel_thomson(f, bw * 1e9) if bw is not None else np.ones(np.shape(f), dtype=complex)

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
        noise_var = float(np.mean(var))
        delay = x.meta.get("delay", 0.0)
        if p.bandwidth_ghz is not None:
            def h(f: np.ndarray) -> np.ndarray:
                return bessel_thomson(f, p.bandwidth_ghz * 1e9)
            current = apply_frequency_response(current, h, x.fs, pad=256)
            noise_var *= float(np.mean(np.abs(h(np.fft.fftfreq(4096, 1 / x.fs))) ** 2))  # white noise through |H|^2
            delay += dc_group_delay(h, x.fs) * x.fs
        meta = {**x.meta, "optical": False, "noise_var": noise_var, "delay": delay, "unit": "A",
                "power": float(np.mean((M * i_primary) ** 2))}
        return {"out": Signal(current, Kind.SAMPLES, fs=x.fs, meta=meta)}


@register_block("coherent_receiver", category="optical")
class CoherentReceiver(Block):
    """Coherent front-end: local oscillator, 90-degree hybrid and balanced photodiodes.

    Output is the complex photocurrent proportional to E * conj(E_LO) per polarization,
    including LO frequency offset and phase noise, LO shot noise and thermal noise. With
    dual polarization the LO power is split between the two hybrids.

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
        npol = e.shape[1] if e.ndim == 3 else 1
        p_lo = dbm_to_w(p.lo_power_dbm) / npol  # a polarization splitter feeds one hybrid per polarization
        scale = p.responsivity * np.sqrt(p_lo)
        bw = fs / 2
        var = 2 * (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw
        if p.shot_noise:
            var += 2 * Q_E * p.responsivity * p_lo * bw
        noise = np.sqrt(var / 2) * (self.rng.standard_normal(e.shape) + 1j * self.rng.standard_normal(e.shape))
        out = scale * e * lo + noise
        sig_power = scale ** 2 * (x.meta.get("power") or 1e-3) / npol
        meta = {**x.meta, "optical": False, "noise_var": var, "power": sig_power, "unit": "A"}
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
        meta = {"power": target, "unit": None}
        if nv is not None:
            meta["noise_var"] = np.mean(nv) * np.mean(g ** 2)
        return {"out": x.evolve(y * g, **meta)}


# --------------------------------------------------------------- coherent DSP
@register_block("cd_compensation", category="optical")
class CDCompensation(Block):
    """Frequency-domain chromatic dispersion compensation (inverse all-pass filter).

    The accumulated dispersion and slope are read from the ``cd_ps_nm`` and
    ``cd_slope_ps_nm2`` metadata unless given.
    """

    class Params(BlockParams):
        dispersion_ps_nm: float | None = Field(None, description="Total D*L to remove (ps/nm).")
        slope_ps_nm2: float | None = Field(None, description="Total S*L to remove (ps/nm^2).")
        wavelength_nm: float | None = Field(None, gt=0)

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        fs = _require_fs(x, "cd_compensation")
        cd = self.params.dispersion_ps_nm if self.params.dispersion_ps_nm is not None else x.meta.get("cd_ps_nm", 0.0)
        slope = self.params.slope_ps_nm2
        if slope is None:
            slope = x.meta.get("cd_slope_ps_nm2", 0.0)
        lam = self.params.wavelength_nm or x.meta.get("wavelength_nm", 1550.0)
        H = np.conj(dispersion_operator(x.length, fs, cd, lam, slope))
        y = np.fft.ifft(np.fft.fft(x.data, axis=-1) * H, axis=-1)
        return {"out": x.evolve(y, cd_ps_nm=x.meta.get("cd_ps_nm", 0.0) - cd,
                                cd_slope_ps_nm2=x.meta.get("cd_slope_ps_nm2", 0.0) - slope,
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
        meta = {"sps": 1, "delay": 0.0, "n_symbols": n_sym, "noise_var": err / count if count else None,
                "unit": None}
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
