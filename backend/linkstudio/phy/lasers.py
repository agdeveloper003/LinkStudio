"""Semiconductor lasers from their rate equations: directly modulated lasers and VCSELs.

A laser driven by a current I(t) is described by the carrier number N and the photon
number S of the lasing mode (Coldren, Corzine & Mashanovitch, ch. 2 and 5):

    dN/dt = (I - I_l)/q - N/tau_n - G(S) (N - N_T) S
    dS/dt = beta N/tau_n + G(S) (N - N_T) S - S/tau_p,       G(S) = g0 / (1 + eps S)

with the output power P = eta_opt h nu S / tau_p and the optical phase
dphi/dt = (alpha/2) g0 (N - N_th) (linewidth-enhancement factor alpha). Above threshold the
power grows linearly with the current (the P-I curve), modulation is limited by the
relaxation resonance, and alpha turns power changes into frequency changes (chirp): the
transient chirp follows dP/dt, the adiabatic chirp follows P through gain compression.

The VCSEL adds the transverse carrier profile and the device temperature (Shi, Qi, Wang &
Hu 2009, after Mena et al. 1999). The carrier density is expanded in Bessel functions,
N(r) = N0 - N1 J0(sigma1 r / R), and integrating the spatially dependent rate equations
over the active region with a Gaussian mode profile leaves three ordinary equations for
N0 (mean carrier number), N1 (spatial hole burning) and S0:

    dN0/dt = (I - I_l)/q - N0/tau_n - G(S0) [(N0 - N_T) h0 - N1 h1] S0
    dN1/dt = G(S0) [(N0 - N_T) h2 - N1 h3] S0 - N1 (1 + h)/tau_n
    dS0/dt = beta N0/tau_n + G(S0) [(N0 - N_T) h0 - N1 h1] S0 - S0/tau_p

with overlap integrals h0..h3 and the diffusion term h = (sigma1 L_D / R)^2. Tucker's
thermal model gives the leakage current I_l(T) = sum a_i T^(i-1), the junction temperature
T = T0 + R_th (V I - P) (thermal time constant ~1 us) and the diode characteristic
V = n V_T ln(I / I_ss + 1) + I R_s(T).

The single-mode laser is the special case h0 = 1, h1 = h2 = h3 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import Field
from scipy.special import j0

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.filters import apply_frequency_response, dc_group_delay
from linkstudio.phy.optical import C_LIGHT, H_PLANCK, Q_E, intensity_noise, wiener_phase

try:  # optional: compile the integrator
    from numba import njit as _njit
except ImportError:  # pragma: no cover - exercised only without numba
    _njit = None

K_BOLTZMANN = 1.380649e-23
SIGMA1 = 3.8317059702075125  # first root of J1 (first non-trivial Bessel mode of the active disc)


@dataclass(frozen=True)
class RateEquations:
    """Rate-equation constants in carrier and photon *numbers* (SI units)."""

    tau_n: float  # carrier lifetime (s)
    tau_p: float  # photon lifetime (s)
    g0: float  # gain coefficient (1/s per carrier above transparency)
    n_tr: float  # transparency carrier number N_T
    epsilon: float  # gain compression per photon
    beta: float  # spontaneous-emission coupling factor
    alpha: float  # linewidth-enhancement (Henry) factor
    eta_opt: float  # fraction of the photon loss rate S/tau_p emitted as useful output
    wavelength_nm: float
    h0: float = 1.0  # overlap of the mode with the mean carrier number
    h1: float = 0.0  # overlap of the mode with the hole-burning term
    h2: float = 0.0  # projection of the stimulated rate on the first Bessel mode
    h3: float = 0.0
    diffusion: float = 0.0  # h = (sigma1 L_D / R)^2

    @property
    def photon_power(self) -> float:
        """Output power per photon in the cavity (W): eta_opt h nu / tau_p."""
        return self.eta_opt * H_PLANCK * C_LIGHT / (self.wavelength_nm * 1e-9) / self.tau_p

    @property
    def threshold_carriers(self) -> float:
        """Carrier number at threshold without spontaneous emission: N_T + 1 / (g0 h0 tau_p)."""
        return self.n_tr + 1.0 / (self.g0 * self.h0 * self.tau_p)


def overlap_coefficients(active_radius_um: float, mode_radius_um: float) -> tuple[float, float, float, float]:
    """h0..h3 of Shi et al. for the Gaussian mode psi(r) = 2 (R/R_m)^2 exp(-2 r^2 / R_m^2).

    h0 = 2/R^2 int psi r dr, h1 = 2/R^2 int psi J0(s1 r/R) r dr, h2 = h1 / J0(s1)^2 and
    h3 = 2/(R^2 J0(s1)^2) int psi J0(s1 r/R)^2 r dr over the active disc of radius R.
    """
    R, Rm = active_radius_um, mode_radius_um
    r = np.linspace(0.0, R, 4001)
    psi = 2 * (R / Rm) ** 2 * np.exp(-2 * r ** 2 / Rm ** 2)
    b = j0(SIGMA1 * r / R)
    norm = 2 / R ** 2
    jn2 = j0(SIGMA1) ** 2
    h0 = norm * np.trapezoid(psi * r, r)
    h1 = norm * np.trapezoid(psi * b * r, r)
    h3 = norm / jn2 * np.trapezoid(psi * b ** 2 * r, r)
    return float(h0), float(h1), float(h1 / jn2), float(h3)


# ------------------------------------------------------------------ steady state
def _carriers(re: RateEquations, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N0, N1 and the stimulated factor (N0 - N_T) h0 - N1 h1 that hold S constant (dN1 = dS = 0)."""
    g = re.g0 / (1 + re.epsilon * s)
    k = g * s * re.h2 / (g * s * re.h3 + (1 + re.diffusion) / re.tau_n)  # N1 = k (N0 - N_T)
    c = re.h0 - re.h1 * k
    n0 = (s / re.tau_p + g * s * c * re.n_tr) / (re.beta / re.tau_n + g * s * c)
    return n0, k * (n0 - re.n_tr), c * (n0 - re.n_tr)


def steady_state(re: RateEquations, current: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static solution (N0, N1, S) for the net injected current I - I_l (A), by bisection on ln S."""
    i = np.atleast_1d(np.asarray(current, dtype=float))
    lo = np.full(i.shape, np.log(1e-8))
    hi = np.full(i.shape, np.log(1e10))
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        s = np.exp(mid)
        n0, _, r0 = _carriers(re, s)
        need = Q_E * (n0 / re.tau_n + re.g0 / (1 + re.epsilon * s) * r0 * s)
        below = need < i
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    s = np.exp(0.5 * (lo + hi))
    n0, n1, _ = _carriers(re, s)
    return n0, n1, s


def static_power(re: RateEquations, current: np.ndarray | float) -> np.ndarray:
    """Output power (W) of the static P-I characteristic for the net current I - I_l."""
    return steady_state(re, np.maximum(current, 0.0))[2] * re.photon_power


def threshold_current(re: RateEquations) -> float:
    """Net threshold current q N_th / tau_n (A)."""
    return Q_E * re.threshold_carriers / re.tau_n


# --------------------------------------------------------------------- dynamics
def _rhs(i, n0, n1, y, tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias):
    s = np.exp(y)
    g = g0 / (1.0 + eps * s)
    r0 = (n0 - n_tr) * h0 - n1 * h1
    dn0 = i / Q_E - n0 / tau_n - g * r0 * s
    dn1 = g * ((n0 - n_tr) * h2 - n1 * h3) * s - n1 * (1.0 + hd) / tau_n
    dy = beta * n0 / (tau_n * s) + g * r0 - 1.0 / tau_p
    dphi = 0.5 * alpha * g0 * (r0 - r_bias)
    return dn0, dn1, dy, dphi


def _integrate(current, dt, m, n0, n1, y, tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias):
    """RK4 over the samples of ``current`` (B, n) with ``m`` sub-steps per sample.

    Returns ln S and the phase at every sample and the final state (N0, N1, ln S). The
    current is linearly interpolated between samples.
    """
    B, n = current.shape
    ys = np.empty((B, n))
    ph = np.empty((B, n))
    phi = np.zeros(B)
    h = dt / m
    for k in range(n):
        ys[:, k] = y
        ph[:, k] = phi
        a = current[:, k]
        b = current[:, k + 1] if k + 1 < n else a
        for j in range(m):
            i1 = a + (b - a) * (j / m)
            i2 = a + (b - a) * ((j + 0.5) / m)
            i3 = a + (b - a) * ((j + 1.0) / m)
            k1 = _rhs(i1, n0, n1, y, tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias)
            k2 = _rhs(i2, n0 + 0.5 * h * k1[0], n1 + 0.5 * h * k1[1], y + 0.5 * h * k1[2],
                      tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias)
            k3 = _rhs(i2, n0 + 0.5 * h * k2[0], n1 + 0.5 * h * k2[1], y + 0.5 * h * k2[2],
                      tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias)
            k4 = _rhs(i3, n0 + h * k3[0], n1 + h * k3[1], y + h * k3[2],
                      tau_n, tau_p, g0, n_tr, eps, beta, alpha, h0, h1, h2, h3, hd, r_bias)
            n0 = n0 + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            n1 = n1 + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            y = y + h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            phi = phi + h / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    return ys, ph, n0, n1, y


if _njit is not None:
    _rhs = _njit(cache=True)(_rhs)
    _integrate = _njit(cache=True)(_integrate)


def simulate(re: RateEquations, current: np.ndarray, fs: float, preroll: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Large-signal response to the net current waveform ``current`` (B, n) in A.

    Returns the output power (W) and the optical phase (rad, carrier defined at the mean
    current). Each row starts from the static state at its first sample; with ``preroll``
    the last samples of the row are run first, so the frame starts in the state it ends
    with (as needed by the circular FFTs downstream).
    """
    current = np.ascontiguousarray(np.atleast_2d(current), dtype=float)
    B, n = current.shape
    preroll = min(preroll, n)
    start = current[:, n - preroll] if preroll else current[:, 0]
    n0, n1, s = steady_state(re, np.maximum(start, 0.0))
    _, _, r_bias = _carriers(re, steady_state(re, max(float(np.mean(current)), 0.0))[2])
    dt = 1.0 / fs
    m = max(1, int(np.ceil(dt / re.tau_p)))  # RK4 stays accurate with steps up to ~tau_p
    args = (re.tau_n, re.tau_p, re.g0, re.n_tr, re.epsilon, re.beta, re.alpha, re.h0, re.h1, re.h2, re.h3,
            re.diffusion, float(r_bias[0]))
    n0, n1, y = n0.copy(), n1.copy(), np.log(s)
    if preroll:
        _, _, n0, n1, y = _integrate(np.ascontiguousarray(current[:, n - preroll:]), dt, m, n0, n1, y, *args)
    ys, ph, _, _, _ = _integrate(current, dt, m, n0, n1, y, *args)
    return np.exp(ys) * re.photon_power, ph


# ------------------------------------------------------------- small signal
def jacobian(re: RateEquations, n0: float, n1: float, s: float) -> np.ndarray:
    """Jacobian of (dN0, dN1, dS) with respect to (N0, N1, S) at a state."""
    g = re.g0 / (1 + re.epsilon * s)
    dg = -re.g0 * re.epsilon / (1 + re.epsilon * s) ** 2
    r0 = (n0 - re.n_tr) * re.h0 - n1 * re.h1
    r2 = (n0 - re.n_tr) * re.h2 - n1 * re.h3
    return np.array([
        [-1 / re.tau_n - g * re.h0 * s, g * re.h1 * s, -(g + dg * s) * r0],
        [g * re.h2 * s, -g * re.h3 * s - (1 + re.diffusion) / re.tau_n, (g + dg * s) * r2],
        [re.beta / re.tau_n + g * re.h0 * s, -g * re.h1 * s, (g + dg * s) * r0 - 1 / re.tau_p],
    ])


def small_signal_response(re: RateEquations, current: float, f: np.ndarray) -> np.ndarray:
    """Modulation response P(f)/P(0) linearized around the static state at ``current`` (A).

    H(j w) = e_S^T (j w I - A)^-1 b / (e_S^T (-A)^-1 b), with A the Jacobian of the rate
    equations and b = (1/q, 0, 0) the current input.
    """
    n0, n1, s = (v[0] for v in steady_state(re, current))
    A = jacobian(re, n0, n1, s)
    b = np.array([1 / Q_E, 0.0, 0.0])
    w = 2j * np.pi * np.asarray(f, dtype=float)
    M = w[:, None, None] * np.eye(3) - A
    x = np.linalg.solve(M, np.broadcast_to(b, (w.size, 3))[..., None])[..., 0]
    dc = np.linalg.solve(-A, b)
    return x[:, 2] / dc[2]


# ----------------------------------------------------------------- thermal model
@dataclass(frozen=True)
class ThermalModel:
    """Tucker's thermal model of a VCSEL (temperatures in degrees Celsius)."""

    ambient_c: float
    r_th: float  # thermal impedance (K/W)
    leakage: tuple[float, ...]  # a_i of I_l(T) = sum a_i T^(i-1) (A, T in C)
    ideality: float  # diode ideality factor n
    iss0: float  # reverse saturation current at 0 C (A)
    b1: float  # I_ss(T) = b1 I_ss0 (T - b2)
    b2: float
    c1: float  # R_s(T) = c1 / (T - c2)
    c2: float

    def leakage_current(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.maximum(sum(a * t ** k for k, a in enumerate(self.leakage)), 0.0)

    def voltage(self, i: np.ndarray | float, t: np.ndarray | float) -> np.ndarray:
        """V = n V_T ln(I / I_ss + 1) + I R_s(T)."""
        i, t = np.asarray(i, dtype=float), np.asarray(t, dtype=float)
        vt = K_BOLTZMANN * (t + 273.15) / Q_E
        iss = self.b1 * self.iss0 * (t - self.b2)
        return self.ideality * vt * np.log(np.maximum(i, 0.0) / iss + 1) + i * self.c1 / (t - self.c2)


def thermal_operating_point(re: RateEquations, th: ThermalModel, current: np.ndarray | float,
                            iterations: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Junction temperature, leakage current and output power at a DC current (A).

    Solves T = T0 + R_th (V(I, T) I - P(I - I_l(T))) by damped fixed-point iteration (the
    static, slow limit of Tucker's thermal rate equation).
    """
    i = np.atleast_1d(np.asarray(current, dtype=float))
    t = np.full(i.shape, th.ambient_c)
    for _ in range(iterations):
        il = th.leakage_current(t)
        p = static_power(re, i - il)
        target = th.ambient_c + th.r_th * (th.voltage(i, t) * i - p)
        t_new = 0.5 * t + 0.5 * target
        if np.max(np.abs(t_new - t)) < 1e-6:
            t = t_new
            break
        t = t_new
    il = th.leakage_current(t)
    return t, il, static_power(re, i - il)


# ======================================================================= blocks
class _LaserBlock(Block):
    """Common processing of the rate-equation lasers (drive current -> optical field)."""

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES, doc="Drive, normalized to [-1, 1].")}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def rate_equations(self) -> RateEquations:
        raise NotImplementedError

    def operating_point(self) -> tuple[float, float | None]:
        """Leakage current (A) and junction temperature (C, None without thermal model)."""
        return 0.0, None

    def parasitic(self, f: np.ndarray) -> np.ndarray:
        """First-order RC pole of the chip and package on the drive current."""
        fp = self.params.parasitic_ghz
        return 1 / (1 + 1j * np.asarray(f) / (fp * 1e9)) if fp else np.ones(np.shape(f), dtype=complex)

    def setup(self) -> None:
        self.re = self.rate_equations()

    def bias_response(self, f: np.ndarray) -> np.ndarray:
        leak, _ = self.operating_point()
        i0 = max(self.params.bias_ma * 1e-3 - leak, 1e-6)
        return small_signal_response(self.re, i0, f) * self.parasitic(f)

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Small-signal modulation response (drive current to optical power) at the bias point."""
        return self.bias_response(f)

    def extra_meta(self) -> dict:
        return {}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        if not x.fs:
            raise BlockError(f"{self.type_name} needs the sample rate (fs) of the signal.")
        fs = x.fs
        drive = np.real(x.data)
        drive = np.clip(drive / (np.max(np.abs(drive)) or 1.0), -1, 1)
        current = p.bias_ma * 1e-3 + p.modulation_ma * 1e-3 * drive
        if p.parasitic_ghz:
            current = apply_frequency_response(current, self.parasitic, fs, pad=256)
        leak, temp = self.operating_point()
        preroll = int(round(3 * self.re.tau_n * fs))
        power, phase = simulate(self.re, np.maximum(current - leak, 0.0), fs, preroll=preroll)
        p_avg = float(np.mean(power))
        power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, fs), 0.0)
        phase = phase + wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
        delay = x.meta.get("delay", 0.0) + dc_group_delay(self.bias_response, fs) * fs
        lo, hi = (max(p.bias_ma - s * p.modulation_ma, 0.0) * 1e-3 - leak for s in (1, -1))
        p_lo, p_hi = static_power(self.re, np.array([lo, hi]))
        self._last = {"threshold_ma": (threshold_current(self.re) + leak) * 1e3,
                      "power_dbm": 10 * np.log10(p_avg / 1e-3),
                      "extinction_ratio_db": 10 * np.log10(p_hi / max(p_lo, 1e-12))}
        if temp is not None:
            self._last["temperature_c"] = temp
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg, "cd_ps_nm": 0.0,
                "ase_psd": 0.0, "unit": "√W", "delay": delay, **self.extra_meta()}
        return {"out": Signal(np.sqrt(power) * np.exp(1j * phase), Kind.SAMPLES, fs=fs, meta=meta)}

    def metrics(self) -> dict[str, float]:
        return dict(self._last) if hasattr(self, "_last") else {}


class _RateParams(BlockParams):
    carrier_lifetime_ns: float = Field(1.0, gt=0, description="tau_n.")
    photon_lifetime_ps: float = Field(2.5, gt=0, description="tau_p.")
    gain_coefficient: float = Field(8.0e4, gt=0, description="g0 (1/s per carrier above transparency).")
    transparency_carriers: float = Field(5.74e7, gt=0, description="N_T (carrier number).")
    gain_compression: float = Field(1.6e-7, ge=0, description="epsilon (per photon).")
    spontaneous_coupling: float = Field(1e-5, ge=0, le=1, description="beta.")
    linewidth_enhancement: float = Field(4.0, description="Henry factor alpha (chirp).")
    optical_efficiency: float = Field(0.3, gt=0, le=1, description="eta_opt: emitted share of the photon loss.")
    rin_db_hz: float | None = Field(None, description="Relative intensity noise added to the output.")
    linewidth_hz: float = Field(0.0, ge=0, description="Lorentzian linewidth (Wiener phase noise).")


@register_block("dml", category="optical")
class DirectlyModulatedLaser(_LaserBlock):
    """Directly modulated single-mode laser (DFB) solved from its rate equations.

    The normalized drive x sets the current I = I_bias + I_mod x, filtered by the RC
    parasitics of the chip. Carrier and photon numbers follow the large-signal rate
    equations, so the output shows the P-I threshold, the relaxation-oscillation ringing
    and the transient and adiabatic chirp of the linewidth-enhancement factor. Metrics:
    threshold current, average power and static extinction ratio.
    """

    class Params(_RateParams):
        bias_ma: float = Field(60.0, gt=0, description="Bias current.")
        modulation_ma: float = Field(36.0, ge=0, description="Peak current swing (drive +-1).")
        wavelength_nm: float = Field(1310.0, gt=0, description="O-band by default, where SMF dispersion is low.")
        parasitic_ghz: float | None = Field(35.0, gt=0, description="Parasitic RC pole (None = ideal chip).")

    def rate_equations(self) -> RateEquations:
        p = self.params
        return RateEquations(p.carrier_lifetime_ns * 1e-9, p.photon_lifetime_ps * 1e-12, p.gain_coefficient,
                             p.transparency_carriers, p.gain_compression, p.spontaneous_coupling,
                             p.linewidth_enhancement, p.optical_efficiency, p.wavelength_nm)


@register_block("vcsel", category="optical")
class VCSEL(_LaserBlock):
    """Directly modulated VCSEL with spatial hole burning and self-heating (Shi et al. 2009).

    Rate equations for the mean carrier number N0, the hole-burning term N1 (first Bessel
    mode of the carrier profile, with lateral diffusion) and the photon number S0, with
    overlap integrals of a Gaussian mode on the active disc. Tucker's thermal model sets the
    junction temperature from the dissipated power V I - P (diode V-I characteristic with
    temperature-dependent series resistance and saturation current) and the thermal
    leakage current I_l(T) = sum a_i T^(i-1), which raises the threshold and rolls the
    power over at high current. The thermal time constant (~1 us) is far longer than a frame,
    so the temperature is the static operating point at the bias current. The RMS spectral
    width is published as ``spectral_width_nm`` for the chromatic dispersion of a
    ``multimode_fiber``.
    """

    class Params(_RateParams):
        bias_ma: float = Field(7.0, gt=0, description="Bias current.")
        modulation_ma: float = Field(3.0, ge=0, description="Peak current swing (drive +-1).")
        wavelength_nm: float = Field(850.0, gt=0)
        spectral_width_nm: float = Field(0.4, ge=0, description="RMS spectral width (multimode VCSEL).")
        parasitic_ghz: float | None = Field(25.0, gt=0, description="Parasitic RC pole (None = ideal chip).")
        photon_lifetime_ps: float = Field(1.8, gt=0, description="tau_p.")
        gain_coefficient: float = Field(3.6e5, gt=0, description="g0 (1/s per carrier above transparency).")
        transparency_carriers: float = Field(1.85e6, gt=0, description="N_T (carrier number).")
        gain_compression: float = Field(1.5e-6, ge=0, description="epsilon (per photon).")
        spontaneous_coupling: float = Field(1e-4, ge=0, le=1, description="beta.")
        linewidth_enhancement: float = Field(3.0, description="Henry factor alpha (chirp).")
        optical_efficiency: float = Field(0.38, gt=0, le=1, description="eta_opt: emitted share of the photon loss.")
        rin_db_hz: float | None = Field(-140.0, description="Relative intensity noise added to the output.")
        spatial_hole_burning: bool = Field(True, description="Carrier profile (N1) and lateral diffusion.")
        active_radius_um: float = Field(3.5, gt=0, description="Radius R of the active region (oxide aperture).")
        mode_radius_um: float = Field(2.5, gt=0, description="Beam waist R_m of the Gaussian mode.")
        diffusion_length_um: float = Field(1.5, ge=0, description="Carrier diffusion length L_D.")
        ambient_c: float = Field(25.0, description="Ambient temperature T0.")
        thermal: bool = Field(True, description="Self-heating and thermal leakage (Tucker's model).")
        thermal_resistance_k_mw: float = Field(3.0, ge=0, description="R_th.")
        leakage_coefficients: list[float] = Field([0.0, 0.0, 0.0, 2e-9],
                                                  description="a_i of I_l = sum a_i T^(i-1) (A, T in C).")
        diode_ideality: float = Field(2.0, gt=0, description="n of the diode V-I characteristic.")
        saturation_current_a: float = Field(1e-15, gt=0, description="I_ss0, reverse saturation current at 0 C.")
        saturation_b1: float = Field(0.02, description="b1 of I_ss(T) = b1 I_ss0 (T - b2) (1/C).")
        saturation_b2: float = Field(-50.0, description="b2 of I_ss(T) (C).")
        series_c1: float = Field(6250.0, gt=0, description="c1 of R_s(T) = c1 / (T - c2) (ohm C).")
        series_c2: float = Field(-100.0, description="c2 of R_s(T) (C).")

    def rate_equations(self) -> RateEquations:
        p = self.params
        h = (1.0, 0.0, 0.0, 0.0)
        hd = 0.0
        if p.spatial_hole_burning:
            h = overlap_coefficients(p.active_radius_um, p.mode_radius_um)
            hd = (SIGMA1 * p.diffusion_length_um / p.active_radius_um) ** 2
        return RateEquations(p.carrier_lifetime_ns * 1e-9, p.photon_lifetime_ps * 1e-12, p.gain_coefficient,
                             p.transparency_carriers, p.gain_compression, p.spontaneous_coupling,
                             p.linewidth_enhancement, p.optical_efficiency, p.wavelength_nm, *h, hd)

    def thermal_model(self, ambient_c: float | None = None) -> ThermalModel:
        p = self.params
        return ThermalModel(p.ambient_c if ambient_c is None else ambient_c, p.thermal_resistance_k_mw * 1e3,
                            tuple(p.leakage_coefficients), p.diode_ideality, p.saturation_current_a,
                            p.saturation_b1, p.saturation_b2, p.series_c1, p.series_c2)

    def setup(self) -> None:
        super().setup()
        self.th = self.thermal_model()
        if self.params.thermal:
            t, il, _ = thermal_operating_point(self.re, self.th, self.params.bias_ma * 1e-3)
            self.op = (float(il[0]), float(t[0]))
        else:
            self.op = (float(self.th.leakage_current(self.params.ambient_c)), self.params.ambient_c)

    def operating_point(self) -> tuple[float, float | None]:
        return self.op

    def extra_meta(self) -> dict:
        return {"spectral_width_nm": self.params.spectral_width_nm}

    def light_current(self, current_a: np.ndarray, ambient_c: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Static P-I curve with self-heating: output power (W) and junction temperature (C)."""
        t, _, power = thermal_operating_point(self.re, self.thermal_model(ambient_c), current_a)
        return power, t
