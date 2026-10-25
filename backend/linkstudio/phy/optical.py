"""Optical communications: transmitters, fiber, amplifiers, receivers and coherent DSP.

Optical signals are complex field envelopes in sqrt(W) (|E|^2 = optical power in W),
shape ``(batch, n)`` or ``(batch, 2, n)`` for dual polarization. Metadata:
``optical`` (True), ``wavelength_nm``, ``power`` (W), ``cd_ps_nm`` (accumulated
chromatic dispersion) and ``ase_psd`` (ASE power spectral density per polarization,
W/Hz, used for OSNR).

* IM/DD: ``intensity_modulator`` / ``mach_zehnder_modulator`` / ``dml`` -> ``fiber`` ->
  ``photodiode`` -> ``agc``
* Coherent: ``iq_modulator`` -> ``optical_link`` -> ``coherent_receiver`` -> ``cd_compensation``
  -> ``fo_compensation`` -> ``adaptive_mimo_equalizer`` -> ``phase_recovery``
* Free space: ``fso_channel`` / ``fso_turbulence``.

The rate-equation lasers (``dml``, ``vcsel``) live in :mod:`linkstudio.phy.lasers`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy.special import erf

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.constellations import Constellation
from linkstudio.phy.dsp.filters import apply_frequency_response, bessel_thomson, dc_group_delay
from linkstudio.phy.theory import C_LIGHT, H_PLANCK, Q_E, gn_eta, responsivity

REF_BW = 12.5e9  # OSNR reference bandwidth (0.1 nm at 1550 nm)
K_BOLTZMANN = 1.380649e-23


def dbm_to_w(p_dbm: float) -> float:
    return 1e-3 * 10 ** (p_dbm / 10)


def w_to_dbm(p_w: float) -> float:
    return 10 * np.log10(max(p_w, 1e-300) / 1e-3)


def _require_fs(x: Signal, what: str) -> float:
    if not x.fs:
        raise BlockError(f"{what} needs the sample rate (fs) of the signal.")
    return x.fs


def wiener_phase(rng: np.random.Generator, batch: int, n: int, linewidth: float, fs: float) -> np.ndarray:
    """Laser phase noise: a random walk with increments of variance 2 pi linewidth / fs (Lorentzian line)."""
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


def osnr_db(meta: dict, bandwidth: float = REF_BW) -> float | None:
    """OSNR = P / P_ASE with P_ASE = 2 S_ASE B (both polarizations) in the bandwidth B (0.1 nm default)."""
    ase = meta.get("ase_psd")
    power = meta.get("power")
    if not ase or power is None:
        return None
    return 10 * float(np.log10(power / (2 * ase * bandwidth)))


def symbol_rate(x: Signal) -> float | None:
    """Symbol rate fs / sps of a pulse-shaped signal, if known."""
    sps = x.meta.get("sps")
    return x.fs / sps if x.fs and sps else None


def osnr_metrics(x: Signal, meta: dict) -> dict[str, float]:
    """OSNR in 0.1 nm and, for pulse-shaped signals, in B = R_s (equal to E_s / N_0 per polarization pair)."""
    out = {}
    o = osnr_db({**x.meta, **meta})
    if o is not None:
        out["osnr_db"] = o
        rs = symbol_rate(x)
        if rs:
            out["osnr_rs_db"] = osnr_db({**x.meta, **meta}, rs)
    return out


# ---------------------------------------------------------------- transmitters
def first_order_lowpass(bandwidth_ghz: float | None):
    """Equivalent electrical response H_mod(f) = 1 / (1 + j f / B) of a modulator or driver (None: flat)."""
    def h(f: np.ndarray) -> np.ndarray:
        if bandwidth_ghz is None:
            return np.ones(np.shape(f), dtype=complex)
        return 1 / (1 + 1j * np.asarray(f) / (bandwidth_ghz * 1e9))
    return h


def normalized_drive(x: Signal) -> np.ndarray:
    """Real drive scaled to a peak of 1 per frame."""
    drive = np.real(x.data)
    return np.clip(drive / (np.max(np.abs(drive)) or 1.0), -1, 1)


def cw_laser(rng: np.random.Generator, shape: tuple[int, int], power_w: float, linewidth_hz: float,
             rin_db_hz: float | None, fs: float) -> np.ndarray:
    """Continuous-wave laser field sqrt(P + n_RIN) exp(j phi): RIN on the power, Wiener phase noise."""
    power = np.maximum(intensity_noise(rng, np.full(shape, power_w), power_w, rin_db_hz, fs), 0.0)
    return np.sqrt(power) * np.exp(1j * wiener_phase(rng, *shape, linewidth_hz, fs))


def mzm_field(v: np.ndarray, v_pi: float, extinction_ratio_db: float | None = None,
              push_pull: bool = True) -> np.ndarray:
    """Field transfer E_out / E_in of a Mach-Zehnder modulator biased for the sine law.

    The two arms carry the phases k V1 and k V2 (k = pi / V_pi) and recombine as
    (E_in / 2)(exp(j k V1) + exp(j k V2)); a static pi phase between the arms puts the null at
    V = 0. Push-pull (V1 = -V2 = V/2) gives the chirp-free E_out = E_in sin(pi V / (2 V_pi)).
    Single-arm drive (V1 = V, V2 = 0) adds the phase pi V / (2 V_pi). A finite extinction
    ratio comes from unequal arm amplitudes (1 +- delta)/2 with delta = 10^(-ER/20).
    """
    d = 0.0 if extinction_ratio_db is None else 10 ** (-extinction_ratio_db / 20)
    theta = np.pi * v / (2 * v_pi)
    if push_pull:
        return ((1 + d) / 2 * np.exp(1j * theta) - (1 - d) / 2 * np.exp(-1j * theta)) / 1j
    return ((1 + d) / 2 * np.exp(2j * theta) - (1 - d) / 2) / 1j


@register_block("intensity_modulator", category="modulator", subcategory="Optical", label="Intensity Modulator")
class IntensityModulator(Block):
    """Ideal intensity modulation for IM/DD links (PAM-M power levels set by the extinction ratio).

    The real drive (NRZ, PAM-4, DMT...) is normalized to [-1, 1] and mapped to the optical
    power P = P_avg (1 + m x) with modulation depth m = (ER - 1) / (ER + 1): equally spaced
    PAM levels from P_1 to P_M = ER P_1. A chirp-free model of an external modulator or of
    the PAM-M transmitter of the IM/DD performance formulas; optional modulator bandwidth,
    relative intensity noise and laser phase noise. Use ``dml`` for a directly modulated
    laser and ``mach_zehnder_modulator`` for the sinusoidal MZM characteristic.
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Average optical power.")
        extinction_ratio_db: float = Field(10.0, gt=0)
        bandwidth_ghz: float | None = Field(None, gt=0, description="Modulator bandwidth H_mod (None = ideal).")
        rin_db_hz: float | None = Field(None, description="Relative intensity noise (e.g. -150 dB/Hz).")
        linewidth_hz: float = Field(0.0, ge=0)
        wavelength_nm: float = Field(1550.0, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Modulator response (drive to optical power)."""
        return first_order_lowpass(self.params.bandwidth_ghz)(f)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "intensity_modulator")
        drive = normalized_drive(x)
        delay = x.meta.get("delay", 0.0)
        if p.bandwidth_ghz is not None:
            h = first_order_lowpass(p.bandwidth_ghz)
            drive = apply_frequency_response(drive, h, fs, pad=256)
            delay += dc_group_delay(h, fs) * fs
        er = 10 ** (p.extinction_ratio_db / 10)
        p_avg = dbm_to_w(p.power_dbm)
        power = p_avg * (1 + (er - 1) / (er + 1) * drive)
        power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, fs), 0.0)
        field = np.sqrt(power) * np.exp(1j * wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs))
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg,
                "cd_ps_nm": 0.0, "ase_psd": 0.0, "delay": delay, "unit": "√W"}
        return {"out": Signal(field, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("mach_zehnder_modulator", category="modulator", subcategory="Optical", label="Mach-Zehnder Modulator")
class MachZehnderModulator(Block):
    """CW laser and external Mach-Zehnder modulator for IM/DD (lithium-niobate MZM).

    The drive x in [-1, 1] (after the equivalent electrical response H_mod) sets
    V(t) = V_bias + V_swing x(t) on the sine-law characteristic
    E_out = sqrt(P_CW L) sin(pi V / (2 V_pi)), P_out = P_CW L sin^2(pi V / (2 V_pi)), with
    insertion loss L. The default quadrature bias V_pi / 2 and swing V_pi / 2 go from the
    null to the peak, as for OOK and PAM-M. Push-pull drive is chirp-free; single-arm drive
    and a finite extinction ratio (arm imbalance) add chirp. Metrics: average output power
    and modulated extinction ratio.
    """

    class Params(BlockParams):
        laser_power_dbm: float = Field(13.0, description="CW laser power.")
        linewidth_hz: float = Field(1e6, ge=0, description="Laser linewidth.")
        rin_db_hz: float | None = Field(None, description="Laser relative intensity noise.")
        v_pi: float = Field(4.0, gt=0, description="Half-wave (switching) voltage.")
        bias_v: float | None = Field(None, description="Bias voltage (None = quadrature, V_pi / 2).")
        swing_v: float | None = Field(None, ge=0, description="Peak drive voltage (None = V_pi / 2).")
        drive: Literal["push_pull", "single_arm"] = "push_pull"
        extinction_ratio_db: float | None = Field(25.0, gt=0, description="Static extinction ratio (None = ideal).")
        insertion_loss_db: float = Field(4.0, ge=0)
        bandwidth_ghz: float | None = Field(None, gt=0, description="Electro-optic bandwidth H_mod (None = ideal).")
        wavelength_nm: float = Field(1550.0, gt=0)

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Equivalent electrical response of the electrodes and driver."""
        return first_order_lowpass(self.params.bandwidth_ghz)(f)

    def voltages(self) -> tuple[float, float]:
        p = self.params
        return (p.v_pi / 2 if p.bias_v is None else p.bias_v), (p.v_pi / 2 if p.swing_v is None else p.swing_v)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "mach_zehnder_modulator")
        drive = normalized_drive(x)
        delay = x.meta.get("delay", 0.0)
        if p.bandwidth_ghz is not None:
            h = first_order_lowpass(p.bandwidth_ghz)
            drive = apply_frequency_response(drive, h, fs, pad=256)
            delay += dc_group_delay(h, fs) * fs
        bias, swing = self.voltages()
        p_cw = dbm_to_w(p.laser_power_dbm) * 10 ** (-p.insertion_loss_db / 10)
        e_in = cw_laser(self.rng, drive.shape, p_cw, p.linewidth_hz, p.rin_db_hz, fs)
        push = p.drive == "push_pull"
        field = e_in * mzm_field(bias + swing * drive, p.v_pi, p.extinction_ratio_db, push)
        levels = np.abs(mzm_field(np.array([bias - swing, bias + swing]), p.v_pi, p.extinction_ratio_db, push)) ** 2
        p_avg = float(np.mean(np.abs(field) ** 2))
        self._last = {"power_dbm": w_to_dbm(p_avg),
                      "extinction_ratio_db": 10 * np.log10(max(levels.max(), 1e-300) / max(levels.min(), 1e-300))}
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg,
                "cd_ps_nm": 0.0, "ase_psd": 0.0, "delay": delay, "unit": "√W"}
        return {"out": Signal(field, Kind.SAMPLES, fs=fs, meta=meta)}

    def metrics(self) -> dict[str, float]:
        return dict(self._last) if hasattr(self, "_last") else {}


@register_block("iq_modulator", category="modulator", subcategory="Optical", label="Optical IQ Modulator")
class IQModulator(Block):
    """Laser + nested IQ Mach-Zehnder modulator for coherent transmission (single or dual polarization).

    Each polarization has two MZMs biased at null for I and Q, recombined with a 90 degree
    phase shift: E = E_I + j E_Q. ``linear`` maps the complex drive directly onto the field;
    ``mzm`` applies the sine law sin(pi V / (2 V_pi)) with peak drive ``drive_swing`` x V_pi,
    a finite child-MZM extinction ratio and an error of the 90 degree phase. Dual
    polarization uses two IQ modulators and a polarization beam combiner, with the laser
    power split equally. The output is scaled to the launch power ``power_dbm`` (as set by
    a booster amplifier).
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Launch power (all polarizations).")
        model: Literal["linear", "mzm"] = "linear"
        drive_swing: float = Field(0.5, gt=0, le=1, description="Peak drive relative to V_pi (mzm).")
        extinction_ratio_db: float | None = Field(None, gt=0, description="Child-MZM extinction ratio (mzm).")
        quadrature_error_deg: float = Field(0.0, description="Error of the 90 degree I/Q phase.")
        bandwidth_ghz: float | None = Field(None, gt=0, description="Electro-optic bandwidth H_mod (None = ideal).")
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
        if p.bandwidth_ghz is not None:
            d = apply_frequency_response(d, first_order_lowpass(p.bandwidth_ghz), fs, pad=256)
        if p.model == "mzm":
            peak = max(np.max(np.abs(d.real)), np.max(np.abs(d.imag))) or 1.0
            ei = mzm_field(p.drive_swing * d.real / peak, 1.0, p.extinction_ratio_db)
            eq = mzm_field(p.drive_swing * d.imag / peak, 1.0, p.extinction_ratio_db)
            d = ei + np.exp(1j * np.pi / 2 * (1 + p.quadrature_error_deg / 90)) * eq
        elif p.quadrature_error_deg:
            d = d.real + np.exp(1j * np.pi / 2 * (1 + p.quadrature_error_deg / 90)) * d.imag
        rms = np.sqrt(np.mean(np.abs(d) ** 2, axis=_pol_axes(d), keepdims=True))
        p_tot = dbm_to_w(p.power_dbm)
        field = d / np.maximum(rms, 1e-30) * np.sqrt(p_tot / npol)
        phase = wiener_phase(self.rng, d.shape[0], d.shape[-1], p.linewidth_hz, fs)
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
              slope_ps_nm2_km: float = 0.0, nonlinear_phase_rad: float | None = None,
              gain: RamanGain | None = None) -> np.ndarray:
    """Split-step Fourier solution of the NLSE (Manakov equation for dual polarization).

    The step is at most 1 / ``steps_per_km`` km. With ``nonlinear_phase_rad`` it is also
    limited so that the peak nonlinear phase per step, gamma P_peak L_eff, stays below that
    value (nonlinear-phase-rotation criterion of Sinkin et al. 2003). ``gain`` adds a
    distributed (Raman) power gain g(z) to the attenuation.
    """
    n = e.shape[-1]
    alpha = att_db_km / (10 * np.log10(np.e)) / 1e3  # 1/m
    total = length_km * 1e3

    def amp(z1: float, z2: float) -> float:
        """Field factor exp(int (g - alpha) dz / 2) over [z1, z2]."""
        return float(np.exp(((gain.integral(z1, z2) if gain else 0.0) - alpha * (z2 - z1)) / 2))

    if not nonlinear or gamma == 0:
        lin = dispersion_operator(n, fs, d_ps_nm_km * length_km, wavelength_nm, slope_ps_nm2_km * length_km)
        return np.fft.ifft(np.fft.fft(e, axis=-1) * lin * amp(0.0, total), axis=-1)
    max_dz = total / max(1, int(np.ceil(length_km * steps_per_km)))
    g = gamma * 1e-3 * (8 / 9 if e.ndim == 3 else 1.0)  # 1/(W km) -> 1/(W m)
    z = 0.0
    while z < total * (1 - 1e-12):
        dz = min(max_dz, total - z)
        if nonlinear_phase_rad:
            p_peak = float(np.max(np.sum(np.abs(e) ** 2, axis=1) if e.ndim == 3 else np.abs(e) ** 2))
            dz = min(dz, max(nonlinear_phase_rad / max(g * p_peak, 1e-30), 1.0))
        half = dispersion_operator(n, fs, d_ps_nm_km * dz / 2e3, wavelength_nm, slope_ps_nm2_km * dz / 2e3)
        # Effective length around the step midpoint (where the nonlinear phase is applied),
        # for the local net loss alpha - g.
        a = alpha - (gain.coefficient(z + dz / 2) if gain else 0.0)
        leff = 2 * np.sinh(a * dz / 2) / a if abs(a) > 1e-12 else dz
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half * amp(z, z + dz / 2), axis=-1)
        p = np.sum(np.abs(e) ** 2, axis=1, keepdims=True) if e.ndim == 3 else np.abs(e) ** 2
        e = e * np.exp(1j * g * p * leff)
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half * amp(z + dz / 2, z + dz), axis=-1)
        z += dz
    return e


# Spontaneous-emission factor of Raman scattering at the 13 THz gain peak (room temperature):
# n_sp = 1 / (1 - exp(-h dnu / (k T))).
RAMAN_NSP = 1 / (1 - np.exp(-H_PLANCK * 13.2e12 / (K_BOLTZMANN * 300.0)))


class RamanGain:
    """Distributed gain of a counter-propagating Raman pump along a span of length L.

    The pump enters at the span end and decays as P_p exp(-alpha_p (L - z)); the signal power
    gain coefficient is g(z) = C_R P_p exp(-alpha_p (L - z)). The on/off gain is
    G_R = exp(C_R P_p L_eff,p) with L_eff,p = (1 - exp(-alpha_p L)) / alpha_p.
    """

    def __init__(self, pump_w: float, cr_w_km: float, pump_att_db_km: float, length_m: float) -> None:
        self.cp = cr_w_km * 1e-3 * pump_w  # 1/m at the pump input
        self.ap = pump_att_db_km / (10 * np.log10(np.e)) / 1e3
        self.length = length_m

    def coefficient(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.cp * np.exp(-self.ap * (self.length - np.asarray(z)))

    def integral(self, z1: float, z2: float) -> float:
        """int_z1^z2 g dz (nepers of power gain)."""
        if self.ap == 0:
            return self.cp * (z2 - z1)
        return float(self.cp / self.ap * (np.exp(-self.ap * (self.length - z2)) - np.exp(-self.ap * (self.length - z1))))

    def on_off_gain(self) -> float:
        return float(np.exp(self.integral(0.0, self.length)))

    def ase_psd(self, att_db_km: float, wavelength_nm: float, n_points: int = 4000) -> float:
        """ASE density per polarization at the span end: n_sp h nu int_0^L g(z) G(z, L) dz.

        G(z, L) = exp(int_z^L (g - alpha) dz') is the net gain from z to the span end.
        """
        alpha = att_db_km / (10 * np.log10(np.e)) / 1e3
        z = np.linspace(0.0, self.length, n_points)
        net = np.array([self.integral(zi, self.length) for zi in z]) - alpha * (self.length - z)
        nu = C_LIGHT / (wavelength_nm * 1e-9)
        return float(RAMAN_NSP * H_PLANCK * nu * np.trapezoid(self.coefficient(z) * np.exp(net), z))


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


# Typical fiber parameters at 1550 nm: attenuation (dB/km), D (ps/nm/km), gamma (1/W/km).
FIBER_TYPES: dict[str, tuple[float, float, float]] = {
    "smf": (0.20, 16.0, 1.3),  # standard single-mode fiber, ITU-T G.652
    "nzdsf": (0.21, 4.0, 2.0),  # non-zero dispersion-shifted fiber, G.655
    "pscf": (0.17, 21.0, 0.8),  # pure-silica-core fiber
    "dcf": (0.45, -100.0, 4.0),  # dispersion-compensating fiber
}


def fiber_constants(p: BlockParams) -> tuple[float, float, float]:
    """(attenuation dB/km, D ps/nm/km, gamma 1/W/km) of the selected fiber type or the custom fields."""
    if p.fiber_type != "custom":
        return FIBER_TYPES[p.fiber_type]
    return p.attenuation_db_km, p.dispersion_ps_nm_km, p.gamma_w_km


class _FiberParams(BlockParams):
    fiber_type: Literal["custom", "smf", "nzdsf", "pscf", "dcf"] = Field(
        "custom", description="Preset attenuation, D and gamma; custom uses the three fields below.")
    attenuation_db_km: float = Field(0.2, ge=0)
    dispersion_ps_nm_km: float = Field(17.0, description="Chromatic dispersion parameter D.")
    nonlinear: bool = Field(False, description="Kerr nonlinearity (split-step Fourier).")
    dispersion_slope_ps_nm2_km: float = Field(0.0, description="Dispersion slope S (G.652 at 1550 nm: ~0.057).")
    gamma_w_km: float = Field(1.3, ge=0, description="Nonlinear coefficient (1/W/km).")
    connector_loss_db: float = Field(0.0, ge=0, description="Lumped loss before the fiber (connectors, splices).")
    steps_per_km: float = Field(0.5, gt=0, description="Minimum split-step resolution (longest step 1/steps_per_km).")
    nonlinear_phase_rad: float | None = Field(0.05, gt=0, description="Largest peak nonlinear phase per step.")
    pmd_ps_sqrt_km: float = Field(0.0, ge=0, description="PMD coefficient (dual polarization).")
    pmd_sections: int = Field(10, ge=1, le=200, description="Birefringent sections per span (waveplate model).")
    polarization_rotation: bool = Field(False, description="Random polarization state per frame.")


@register_block("fiber", category="medium", subcategory="Fiber", label="Single-Mode Fiber")
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
        att, d, _ = fiber_constants(p)
        lam = x.meta.get("wavelength_nm", 1550.0)
        cd = d * p.length_km
        b2l = beta2(cd, lam) * 1e3
        b3l = beta3(cd, p.dispersion_slope_ps_nm2_km * p.length_km, lam) * 1e3
        w = 2 * np.pi * np.asarray(f, dtype=float)
        loss = 10 ** (-(att * p.length_km + p.connector_loss_db) / 20)
        return loss * np.exp(1j * (b2l / 2 * w ** 2 - b3l / 6 * w ** 3))

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "fiber")
        att, d, gamma = fiber_constants(p)
        lam = x.meta.get("wavelength_nm", 1550.0)
        e = x.data * 10 ** (-p.connector_loss_db / 20)
        e = propagate(e, fs, p.length_km, att, d, lam, p.nonlinear, gamma, p.steps_per_km,
                      p.dispersion_slope_ps_nm2_km, p.nonlinear_phase_rad)
        if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
            e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.length_km), p.pmd_sections)
        loss = 10 ** (-(att * p.length_km + p.connector_loss_db) / 10)
        return {"out": x.evolve(e, power=(x.meta.get("power") or 1e-3) * loss,
                                ase_psd=x.meta.get("ase_psd", 0.0) * loss,
                                cd_ps_nm=x.meta.get("cd_ps_nm", 0.0) + d * p.length_km,
                                cd_slope_ps_nm2=x.meta.get("cd_slope_ps_nm2", 0.0)
                                + p.dispersion_slope_ps_nm2_km * p.length_km)}


def ase_psd(gain_lin: float, nf_db: float, wavelength_nm: float) -> float:
    """ASE power spectral density per polarization (W/Hz): S_ASE = N_0 = n_sp h f (G - 1), n_sp = F / 2.

    Each of the four field quadratures carries white noise of density N_0 / 2, so the ASE
    power in a bandwidth B is P_ASE = 2 N_0 B = h f (G - 1) F B.
    """
    if gain_lin <= 1:
        return 0.0
    nu = C_LIGHT / (wavelength_nm * 1e-9)
    return 10 ** (nf_db / 10) / 2 * H_PLANCK * nu * (gain_lin - 1)


def _add_ase(rng: np.random.Generator, e: np.ndarray, psd: float, fs: float) -> np.ndarray:
    sigma = np.sqrt(psd * fs / 2)
    return e + sigma * (rng.standard_normal(e.shape) + 1j * rng.standard_normal(e.shape))


def saturated_gain(g0: float, p_in_w: float, p_sat_w: float) -> float:
    """Gain of a homogeneously broadened amplifier: G = G0 exp(-(G - 1) P_in / P_sat)."""
    from scipy.optimize import brentq
    if p_in_w <= 0:
        return g0
    return float(brentq(lambda g: g - g0 * np.exp(-(g - 1) * p_in_w / p_sat_w), 1.0, g0))


@register_block("optical_amplifier", category="amplifier", label="Optical Amplifier (EDFA, SOA)")
class OpticalAmplifier(Block):
    """Optical amplifier (EDFA, SOA) with amplified spontaneous emission noise.

    E_out = sqrt(G) E_in + n(t): the gain is linear on the field and the ASE is white
    Gaussian noise on the four quadratures, with density N_0 / 2 = n_sp h f (G - 1) / 2 each and
    n_sp = F / 2 (noise figure F >= 2). ``gain`` mode applies ``gain_db``, optionally
    compressed by gain saturation; ``power`` mode sets the output power to
    ``target_power_dbm``. Metrics: OSNR in 0.1 nm and in B = R_s, gain.
    """

    class Params(BlockParams):
        mode: Literal["gain", "power"] = "power"
        gain_db: float = Field(20.0, ge=0)
        target_power_dbm: float = Field(0.0)
        noise_figure_db: float = Field(5.0, ge=3.0)
        saturation_power_dbm: float | None = Field(None, description="Saturation power P_sat (gain mode).")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def on_reset(self) -> None:
        self._osnr: list[float] = []
        self._osnr_rs: list[float] = []
        self._gain: list[float] = []

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "optical_amplifier")
        p_in = x.meta.get("power") or float(np.mean(np.sum(np.abs(x.data) ** 2, axis=1) if x.data.ndim == 3
                                                    else np.abs(x.data) ** 2))
        if p.mode == "gain":
            g = 10 ** (p.gain_db / 10)
            if p.saturation_power_dbm is not None:
                g = saturated_gain(g, p_in, dbm_to_w(p.saturation_power_dbm))
        else:
            g = dbm_to_w(p.target_power_dbm) / p_in
        s = ase_psd(g, p.noise_figure_db, x.meta.get("wavelength_nm", 1550.0))
        e = _add_ase(self.rng, x.data * np.sqrt(g), s, fs)
        meta = {"power": p_in * g, "ase_psd": x.meta.get("ase_psd", 0.0) * g + s}
        m = osnr_metrics(x, meta)
        if "osnr_db" in m:
            self._osnr.append(m["osnr_db"])
        if "osnr_rs_db" in m:
            self._osnr_rs.append(m["osnr_rs_db"])
        self._gain.append(10 * np.log10(g))
        return {"out": x.evolve(e, **meta)}

    def metrics(self) -> dict[str, float]:
        out = {}
        for key, values in (("osnr_db", self._osnr), ("osnr_rs_db", self._osnr_rs), ("gain_db", self._gain)):
            if values:
                out[key] = float(np.mean(values))
        return out


@register_block("optical_attenuator", category="amplifier", label="Optical Attenuator")
class OpticalAttenuator(Block):
    """Passive loss: variable optical attenuator, 1xN power splitter, connectors.

    Total loss = ``loss_db`` + 10 log10(N) + excess loss for a splitter with ``split_ways`` = N
    outputs (P_out = P_in / (N eps)). Signal and ASE are attenuated alike, so the OSNR is
    unchanged.
    """

    class Params(BlockParams):
        loss_db: float = Field(0.0, ge=0)
        split_ways: int = Field(1, ge=1, description="Outputs of an ideal splitter (one output is kept).")
        excess_loss_db: float = Field(0.0, ge=0, description="Excess loss of the splitter.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def total_loss_db(self) -> float:
        p = self.params
        return p.loss_db + 10 * np.log10(p.split_ways) + p.excess_loss_db

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Flat loss."""
        return np.full(np.shape(f), 10 ** (-self.total_loss_db() / 20), dtype=complex)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        a = 10 ** (-self.total_loss_db() / 10)
        meta = {"power": (x.meta.get("power") or 1e-3) * a, "ase_psd": x.meta.get("ase_psd", 0.0) * a}
        return {"out": x.evolve(x.data * np.sqrt(a), **meta)}


@register_block("optical_link", category="medium", subcategory="Fiber", label="Amplified Optical Link")
class OpticalLink(Block):
    """Multi-span amplified link: ``n_spans`` x (connectors + fiber span + optional Raman + EDFA).

    By default every EDFA compensates the loss of its span (transparent spans, G A = 1);
    ``amplifier_gain_db`` fixes the gain instead, so the power drifts by (G A)^M. A
    counter-propagating Raman pump (``raman_pump_mw``) adds distributed gain in the fiber
    and its own ASE. Metrics: OSNR in 0.1 nm and in B = R_s, and for EDFA-only transparent
    links the GN-model estimate of the nonlinear interference (NLI power, GSNR and optimum
    launch power for the WDM comb in the ``wdm`` metadata); with Raman, the on/off gain and
    the equivalent noise figure of the span.
    """

    class Params(_FiberParams):
        n_spans: int = Field(10, ge=1, le=100)
        span_km: float = Field(80.0, gt=0)
        noise_figure_db: float = Field(5.0, ge=3.0)
        amplifier_gain_db: float | None = Field(None, ge=0, description="EDFA gain (None = span loss, G A = 1).")
        launch_power_dbm: float | None = Field(None, description="Rescale the input to this power.")
        raman_pump_mw: float = Field(0.0, ge=0, description="Counter-propagating Raman pump power per span.")
        raman_gain_w_km: float = Field(0.4, ge=0, description="Raman gain efficiency C_R at the signal (1/W/km).")
        raman_pump_attenuation_db_km: float = Field(0.25, ge=0, description="Fiber loss at the pump wavelength.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def raman(self) -> RamanGain | None:
        p = self.params
        if p.raman_pump_mw <= 0:
            return None
        return RamanGain(p.raman_pump_mw * 1e-3, p.raman_gain_w_km, p.raman_pump_attenuation_db_km, p.span_km * 1e3)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "optical_link")
        lam = x.meta.get("wavelength_nm", 1550.0)
        att, d, gamma = fiber_constants(p)
        e = x.data
        power = (x.meta.get("power") or 1e-3)
        if p.launch_power_dbm is not None:
            e = e * np.sqrt(dbm_to_w(p.launch_power_dbm) / power)
            power = dbm_to_w(p.launch_power_dbm)
        launch = power
        raman = self.raman()
        a_pre = 10 ** (-p.connector_loss_db / 10)
        a_f = 10 ** (-att * p.span_km / 10)
        g_r = raman.on_off_gain() if raman else 1.0
        s_r = raman.ase_psd(att, lam) if raman else 0.0
        g = 1 / (a_pre * a_f * g_r) if p.amplifier_gain_db is None else 10 ** (p.amplifier_gain_db / 10)
        s = ase_psd(g, p.noise_figure_db, lam)
        ase = x.meta.get("ase_psd", 0.0)
        for _ in range(p.n_spans):
            e = propagate(e * np.sqrt(a_pre), fs, p.span_km, att, d, lam, p.nonlinear, gamma, p.steps_per_km,
                          p.dispersion_slope_ps_nm2_km, p.nonlinear_phase_rad, gain=raman)
            if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
                e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.span_km), p.pmd_sections)
            if s_r:
                e = _add_ase(self.rng, e, s_r, fs)
            e = _add_ase(self.rng, e * np.sqrt(g), s, fs)
            ase = (ase * a_pre * a_f * g_r + s_r) * g + s
            power = power * a_pre * a_f * g_r * g
        meta = {"power": power, "ase_psd": ase,
                "cd_ps_nm": x.meta.get("cd_ps_nm", 0.0) + d * p.span_km * p.n_spans,
                "cd_slope_ps_nm2": x.meta.get("cd_slope_ps_nm2", 0.0)
                + p.dispersion_slope_ps_nm2_km * p.span_km * p.n_spans}
        self._last = osnr_metrics(x, meta)
        rs = symbol_rate(x)
        if raman:
            nu = C_LIGHT / (lam * 1e-9)
            g_passive = 1 / (a_pre * a_f)
            self._last["raman_on_off_gain_db"] = 10 * np.log10(g_r)
            self._last["equivalent_nf_db"] = 10 * np.log10(2 * (s_r * g + s) / (H_PLANCK * nu * (g_passive - 1)))
        elif rs and p.amplifier_gain_db is None:
            wdm = x.meta.get("wdm") or {}
            eta = gn_eta(gamma, att, beta2(d, lam) * 1e27, rs, p.span_km, wdm.get("n_channels", 1),
                         wdm.get("spacing_hz")) * a_pre ** 2  # launch power enters the fiber after A_pre
            p_ase = 2 * s * rs  # ASE of one span in B = R_s, both polarizations
            p_nli = p.n_spans * eta * launch ** 3
            self._last.update({"gn_nli_dbm": w_to_dbm(p_nli),
                               "gsnr_gn_db": 10 * np.log10(launch / (p.n_spans * p_ase + p_nli)),
                               "gn_optimum_launch_dbm": w_to_dbm((p_ase / (2 * eta)) ** (1 / 3))})
        return {"out": x.evolve(e, **meta)}

    def metrics(self) -> dict[str, float]:
        return dict(self._last) if hasattr(self, "_last") else {}


@register_block("wdm_multiplexer", category="multiplexer", label="WDM Multiplexer")
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


@register_block("wdm_demultiplexer", category="demultiplexer", label="WDM Demultiplexer")
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


@register_block("fso_turbulence", category="medium", subcategory="Free-Space Optics", label="FSO Turbulence")
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


@register_block("fso_channel", category="medium", subcategory="Free-Space Optics", label="FSO Channel")
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


@register_block("photodiode", category="demodulator", subcategory="Optical", label="Photodiode Receiver (PIN, APD)")
class Photodiode(Block):
    """Direct-detection receiver: PIN or APD photodiode and transimpedance amplifier (TIA).

    i = M R P: responsivity R = q eta / (h f) (from ``quantum_efficiency`` when set) and APD
    multiplication M. Noise is generated over the simulation bandwidth fs / 2: TIA noise of
    input-referred density IRND (``thermal_noise_pa_rthz``), shot noise 2 q M^2 F R P and dark
    current. The APD excess-noise factor is McIntyre's F = k M + (1 - k)(2 - 1/M) or the
    power law F = M^x of APD datasheets. An optional ``bandwidth_ghz`` applies a 4th-order
    Bessel-Thomson response (signal and noise), the reference receiver of optical
    Ethernet. With ``transimpedance_ohm`` the output is the TIA voltage G_TIA i.
    """

    class Params(_DetectorParams):
        quantum_efficiency: float | None = Field(None, gt=0, le=1,
                                                 description="Set R = q eta / (h f) instead of the responsivity.")
        dark_current_na: float = Field(5.0, ge=0)
        apd_gain: float = Field(1.0, ge=1.0)
        apd_excess_noise: Literal["mcintyre", "power_law"] = "mcintyre"
        apd_k: float = Field(0.2, ge=0, le=1, description="APD ionization ratio (McIntyre).")
        apd_x: float = Field(0.7, ge=0, le=1, description="Excess-noise exponent x of F = M^x (power law).")
        bandwidth_ghz: float | None = Field(None, gt=0, description="Receiver 3-dB bandwidth (None = fs/2).")
        transimpedance_ohm: float | None = Field(None, gt=0, description="TIA gain: output in volts.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Photocurrent (A) or TIA voltage (V), real.")}

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Receiver bandwidth (Bessel-Thomson); flat when no bandwidth is set."""
        bw = self.params.bandwidth_ghz
        return bessel_thomson(f, bw * 1e9) if bw is not None else np.ones(np.shape(f), dtype=complex)

    def excess_noise(self) -> float:
        p = self.params
        M = p.apd_gain
        if p.apd_excess_noise == "power_law":
            return M ** p.apd_x
        return p.apd_k * M + (1 - p.apd_k) * (2 - 1 / M)

    def responsivity_aw(self, wavelength_nm: float) -> float:
        p = self.params
        return responsivity(p.quantum_efficiency, wavelength_nm) if p.quantum_efficiency else p.responsivity

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        bw = _require_fs(x, "photodiode") / 2
        power = np.sum(np.abs(x.data) ** 2, axis=1) if x.data.ndim == 3 else np.abs(x.data) ** 2
        M, F = p.apd_gain, self.excess_noise()
        i_primary = self.responsivity_aw(x.meta.get("wavelength_nm", 1550.0)) * power + p.dark_current_na * 1e-9
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
        gain, unit = (p.transimpedance_ohm, "V") if p.transimpedance_ohm else (1.0, "A")
        meta = {**x.meta, "optical": False, "noise_var": noise_var * gain ** 2, "delay": delay, "unit": unit,
                "power": float(np.mean((M * i_primary * gain) ** 2))}
        return {"out": Signal(current * gain, Kind.SAMPLES, fs=x.fs, meta=meta)}


@register_block("coherent_receiver", category="demodulator", subcategory="Optical", label="Coherent Optical Receiver")
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
        hybrid_phase_error_deg: float = Field(0.0, description="Deviation of the hybrid from 90 degrees (I/Q skew).")
        internal_snr_db: float | None = Field(None, description="Transceiver ISNR: noise proportional to the signal.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        fs = _require_fs(x, "coherent_receiver")
        e = x.data
        B, n = e.shape[0], e.shape[-1]
        walk = wiener_phase(self.rng, B, n + 1, p.lo_linewidth_hz, fs)
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
        beat = scale * e * lo
        if p.hybrid_phase_error_deg:  # the Q output sees the LO at 90 + error degrees
            err = np.deg2rad(p.hybrid_phase_error_deg)
            beat = beat.real + 1j * (beat.imag * np.cos(err) - beat.real * np.sin(err))
        sig_power = scale ** 2 * (x.meta.get("power") or 1e-3) / npol
        if p.internal_snr_db is not None:  # P_N,int = chi P_sig with chi = 1 / ISNR
            var += sig_power * 10 ** (-p.internal_snr_db / 10)
        noise = np.sqrt(var / 2) * (self.rng.standard_normal(e.shape) + 1j * self.rng.standard_normal(e.shape))
        meta = {**x.meta, "optical": False, "noise_var": var, "power": sig_power, "unit": "A"}
        return {"out": Signal(beat + noise, Kind.SAMPLES, fs=fs, meta=meta)}


@register_block("agc", category="amplifier", label="Automatic Gain Control")
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
@register_block("cd_compensation", category="equalizer", subcategory="Dispersion Compensation", label="Chromatic Dispersion Compensation")
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


@register_block("fo_compensation", category="synchronization", subcategory="Carrier", label="Frequency Offset Compensation")
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


@register_block("adaptive_mimo_equalizer", category="equalizer", subcategory="Adaptive", label="Adaptive MIMO Equalizer")
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


@register_block("phase_recovery", category="synchronization", subcategory="Carrier", label="Carrier Phase Recovery")
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


@register_block("optical_power_meter", category="measurement", label="Optical Power Meter")
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
