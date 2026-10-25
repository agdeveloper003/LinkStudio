"""Short-reach optics: VCSEL transmitters and graded-index multimode fiber (data-center links).

A VCSEL launches light into hundreds of fiber modes and has a spectrum hundreds of GHz
wide, so the modes add in power, not in field. The multimode fiber is therefore a
linear system acting on the optical *power*, with an impulse response set by

* modal dispersion: the mode groups of a graded-index (alpha-profile) core arrive at
  different times (Gloge & Marcatili 1973; Olshansky & Keck 1976),
* chromatic dispersion of the wide VCSEL spectrum: a Gaussian response whose width is
  the IEEE 802.3 link-model expression (Cunningham & Dawe 2002).

Two fiber models are available: ``bandwidth`` (Gaussian response from the effective
modal bandwidth of the fiber class, as in the IEEE 802.3 link model) and ``mode_groups``
(discrete mode-group delays from the WKB theory of alpha-profile fibers, with the launch
condition deciding how much power each group carries).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy.optimize import brentq

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block
from linkstudio.phy.dsp.filters import apply_frequency_response, dc_group_delay
from linkstudio.phy.optical import C_LIGHT, dbm_to_w, intensity_noise, laser_response

# Gaussian impulse response: 3-dB optical bandwidth B and RMS width sigma obey B * sigma = sqrt(ln 2 / 2) / pi.
GAUSS_BW_RMS = np.sqrt(np.log(2) / 2) / np.pi  # 0.1874, the "0.187" of the IEEE 802.3 link model

# Fiber classes (ISO/IEC 11801, TIA-568.3, IEC 60793-2-10) at 850 nm:
# core radius (um), NA, overfilled-launch bandwidth (MHz km), effective modal bandwidth (MHz km), cabled loss (dB/km).
MMF_PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "OM1": (31.25, 0.275, 200.0, 200.0, 3.5),
    "OM2": (25.0, 0.200, 500.0, 500.0, 3.5),
    "OM3": (25.0, 0.200, 1500.0, 2000.0, 3.5),
    "OM4": (25.0, 0.200, 3500.0, 4700.0, 3.5),
    "OM5": (25.0, 0.200, 3500.0, 4700.0, 3.5),
}


# ------------------------------------------------------------------- theory
def profile_contrast(na: float, n1: float) -> float:
    """Relative index difference Delta = NA^2 / (2 n1^2)."""
    return na ** 2 / (2 * n1 ** 2)


def mode_count(core_radius_um: float, na: float, wavelength_nm: float, alpha: float) -> float:
    """WKB number of guided modes (both polarizations): M = alpha / (alpha + 2) * V^2 / 2."""
    v = 2 * np.pi * core_radius_um * 1e-6 * na / (wavelength_nm * 1e-9)
    return alpha / (alpha + 2) * v ** 2 / 2


def group_delay(u: np.ndarray, alpha: float, delta: float, eps: float = 0.0) -> np.ndarray:
    """Relative group delay tau / (L N1 / c) - 1 of the modes at mode-count fraction u = m / M.

    Second-order expansion of the WKB delay (Olshansky & Keck 1976) with x = u^(alpha/(alpha+2)):
    Delta C1 x + Delta^2 C2 x^2, C1 = (alpha - 2 - eps)/(alpha + 2),
    C2 = (3 alpha - 2 - 2 eps) / (2 (alpha + 2)); eps is the profile-dispersion parameter.
    """
    x = np.asarray(u, dtype=float) ** (alpha / (alpha + 2))
    c1 = (alpha - 2 - eps) / (alpha + 2)
    c2 = (3 * alpha - 2 - 2 * eps) / (2 * (alpha + 2))
    return delta * c1 * x + delta ** 2 * c2 * x ** 2


def overfilled_rms(alpha: float, delta: float, eps: float = 0.0) -> float:
    """RMS width of the relative delay for uniform power over all modes (continuum limit).

    The moments of x = u^q (u uniform on [0, 1]) are E[x^k] = 1 / (k q + 1).
    """
    q = alpha / (alpha + 2)
    c1 = (alpha - 2 - eps) / (alpha + 2)
    c2 = (3 * alpha - 2 - 2 * eps) / (2 * (alpha + 2))
    m = [1 / (k * q + 1) for k in range(5)]
    var = (c1 ** 2 * (m[2] - m[1] ** 2) + 2 * delta * c1 * c2 * (m[3] - m[1] * m[2])
           + delta ** 2 * c2 ** 2 * (m[4] - m[2] ** 2))
    return delta * float(np.sqrt(var))


def optimum_alpha(delta: float, eps: float = 0.0) -> float:
    """Profile exponent that minimizes the overfilled RMS width (Olshansky & Keck 1976)."""
    return 2 + eps - delta * (4 + eps) * (3 + eps) / (5 + 2 * eps)


def chromatic_rms_ps_per_km(wavelength_nm: float, rms_width_nm: float, lambda0_nm: float, s0: float) -> float:
    """RMS width (ps/km) of the chromatic-dispersion response of a wide source spectrum.

    IEEE 802.3 link model: sigma = sigma_lambda sqrt(D1^2 + D2^2), with the dispersion
    D1 = S0 lambda / 4 (1 - (lambda0 / lambda)^4) and the spectral-width term D2 = 0.7 S0 sigma_lambda.
    """
    d1 = s0 * wavelength_nm / 4 * (1 - (lambda0_nm / wavelength_nm) ** 4)
    d2 = 0.7 * s0 * rms_width_nm
    return rms_width_nm * float(np.hypot(d1, d2))


def gaussian_launch_weights(n_groups: int, waist_um: float, offset_um: float, fundamental_um: float) -> np.ndarray:
    """Power coupled into each principal mode group by a Gaussian beam (parabolic-core eigenmodes).

    The modes of a parabolic core are the Hermite-Gauss modes of a 2-D harmonic oscillator
    with fundamental field radius w0; group p holds the modes with nx + ny = p. A beam of
    field radius ``waist_um`` displaced by ``offset_um`` along x separates in x and y, so the
    group powers follow from 1-D overlap integrals. Returns the fraction of the launched power
    in each group (the remainder is not guided).
    """
    xi = np.linspace(-12, 12, 4001)  # in units of w0 / sqrt(2) (oscillator length)
    dx = xi[1] - xi[0]
    s = waist_um / fundamental_um
    d = offset_um / fundamental_um * np.sqrt(2)
    hermite = np.zeros((n_groups, xi.size))  # normalized oscillator eigenfunctions
    hermite[0] = np.pi ** -0.25 * np.exp(-xi ** 2 / 2)
    if n_groups > 1:
        hermite[1] = np.sqrt(2) * xi * hermite[0]
    for k in range(2, n_groups):
        hermite[k] = np.sqrt(2 / k) * xi * hermite[k - 1] - np.sqrt((k - 1) / k) * hermite[k - 2]

    def beam(centre: float) -> np.ndarray:
        g = np.exp(-(xi - centre) ** 2 / (2 * s ** 2))
        return g / np.sqrt(np.sum(g ** 2) * dx)

    cx = (hermite @ beam(d) * dx) ** 2
    cy = (hermite @ beam(0.0) * dx) ** 2
    return np.array([np.sum(cx[:p + 1] * cy[p::-1]) for p in range(n_groups)])


# ----------------------------------------------------------- multimode fiber
@register_block("multimode_fiber", category="optical")
class MultimodeFiber(Block):
    """Graded-index multimode fiber (OM1-OM5) for short-reach IM/DD links.

    Acts on the optical power: P_out = (P_in * h)(t) x 10^(-loss/10), with h the modal
    response convolved with the Gaussian chromatic-dispersion response of the source
    spectrum (``spectral_width_nm`` metadata of a ``vcsel``, or ``source_rms_width_nm``).

    * ``bandwidth``: Gaussian modal response with the effective modal bandwidth of the
      fiber class divided by the length (IEEE 802.3 link model).
    * ``mode_groups``: one delay per principal mode group from the WKB theory of an
      alpha-profile core. Without an explicit ``alpha`` the profile is detuned from the
      optimum so that the overfilled-launch bandwidth equals the class specification.
      ``launch`` sets the group powers: ``overfilled`` (all modes equally), ``gaussian``
      (beam of ``launch_waist_um`` at ``launch_offset_um`` from the axis) or ``custom``.

    Incoherent model: the output field is sqrt(P_out) with the input phase; dual-polarization
    inputs are summed into one intensity.
    """

    class Params(BlockParams):
        fiber_type: Literal["OM1", "OM2", "OM3", "OM4", "OM5"] = "OM4"
        length_m: float = Field(100.0, ge=0)
        model: Literal["bandwidth", "mode_groups"] = "bandwidth"
        modal_bandwidth_mhz_km: float | None = Field(None, gt=0, description="Override the class EMB (bandwidth model).")
        attenuation_db_km: float | None = Field(None, ge=0, description="Override the cabled loss of the class.")
        alpha: float | None = Field(None, gt=1, lt=10, description="Core profile exponent (mode_groups).")
        profile_dispersion: float = Field(0.0, description="Profile-dispersion parameter epsilon.")
        launch: Literal["overfilled", "gaussian", "custom"] = "overfilled"
        launch_waist_um: float = Field(4.0, gt=0, description="Field radius of the launched beam (gaussian).")
        launch_offset_um: float = Field(0.0, ge=0, description="Radial offset of the launched beam (gaussian).")
        group_weights: list[float] = Field([], description="Relative power per mode group, lowest first (custom).")
        core_index: float = Field(1.467, gt=1, description="Core-centre refractive index n1.")
        group_index: float = Field(1.482, gt=1, description="Core group index N1.")
        zero_dispersion_nm: float = Field(1320.0, gt=0, description="Zero-dispersion wavelength lambda0.")
        dispersion_slope: float = Field(0.101, ge=0, description="S0 in ps/(nm^2 km).")
        source_rms_width_nm: float | None = Field(None, ge=0, description="Override the source RMS spectral width.")

    inputs = {"in": Port(Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES)}

    def setup(self) -> None:
        p = self.params
        radius, na, self.ofl_bw, self.emb, self.loss_db_km = MMF_PRESETS[p.fiber_type]
        if p.attenuation_db_km is not None:
            self.loss_db_km = p.attenuation_db_km
        self.radius_um, self.na = radius, na
        self.delta = profile_contrast(na, p.core_index)
        self.groups: tuple[np.ndarray, np.ndarray] | None = None
        if p.model == "mode_groups":
            self.groups = self.mode_groups(850.0)

    # -- mode-group model
    def profile_alpha(self, wavelength_nm: float) -> float:
        """Explicit alpha, or the detuned profile meeting the overfilled-bandwidth specification."""
        p = self.params
        if p.alpha is not None:
            return p.alpha
        a0 = optimum_alpha(self.delta, p.profile_dispersion)
        target = GAUSS_BW_RMS / (self.ofl_bw * 1e6) / (p.group_index / C_LIGHT * 1e3)  # relative RMS over 1 km
        return brentq(lambda a: self._rms(self._weights_delays(a, wavelength_nm, "overfilled")) - target,
                      a0, a0 + 1.0, xtol=1e-12)

    def _weights_delays(self, alpha: float, wavelength_nm: float, launch: str) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        g = max(1, int(round(np.sqrt(mode_count(self.radius_um, self.na, wavelength_nm, alpha)))))
        mu = np.arange(1, g + 1)
        delays = group_delay(((mu - 0.5) / g) ** 2, alpha, self.delta, p.profile_dispersion)
        if launch == "overfilled":
            w = (2 * mu - 1).astype(float)  # group mu holds 2 mu - 1 of the g^2 modes
        elif launch == "gaussian":
            w0 = np.sqrt(self.radius_um * 1e-6 * wavelength_nm * 1e-9 / (np.pi * self.na)) * 1e6
            w = gaussian_launch_weights(g, p.launch_waist_um, p.launch_offset_um, w0)
        else:
            if not p.group_weights:
                raise BlockError("Custom launch needs group_weights.")
            w = np.zeros(g)
            k = min(g, len(p.group_weights))
            w[:k] = np.asarray(p.group_weights[:k], dtype=float)
        return w, delays

    @staticmethod
    def _rms(wd: tuple[np.ndarray, np.ndarray]) -> float:
        w, d = wd
        w = w / w.sum()
        mean = np.sum(w * d)
        return float(np.sqrt(np.sum(w * (d - mean) ** 2)))

    def mode_groups(self, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        """Group powers (sum = coupled fraction) and delays in seconds per metre, mean delay removed."""
        w, d = self._weights_delays(self.profile_alpha(wavelength_nm), wavelength_nm, self.params.launch)
        if w.sum() <= 0:
            raise BlockError("The launch couples no power into guided modes.")
        d = d - np.sum(w * d) / w.sum()
        return w, d * self.params.group_index / C_LIGHT

    # -- response
    def modal_rms_s(self) -> float:
        """RMS width of the modal impulse response over ``length_m``."""
        p = self.params
        km = p.length_m / 1e3
        if self.groups is None:
            return GAUSS_BW_RMS / ((p.modal_bandwidth_mhz_km or self.emb) * 1e6 / max(km, 1e-12))
        w, d = self.groups
        return self._rms((w, d)) * p.length_m

    def response(self, f: np.ndarray, sigma_cd_s: float) -> np.ndarray:
        """Power transfer function (loss excluded) for a chromatic RMS width ``sigma_cd_s``."""
        p = self.params
        h = np.exp(-2 * (np.pi * f * sigma_cd_s) ** 2)
        if self.groups is None:
            return h * np.exp(-2 * (np.pi * f * self.modal_rms_s()) ** 2)
        w, d = self.groups
        modal = np.exp(-2j * np.pi * np.multiply.outer(f, d * p.length_m)) @ (w / w.sum())
        return h * modal

    def chromatic_rms_s(self, x: Signal) -> float:
        p = self.params
        lam = x.meta.get("wavelength_nm", 850.0)
        width = p.source_rms_width_nm if p.source_rms_width_nm is not None else x.meta.get("spectral_width_nm", 0.0)
        return chromatic_rms_ps_per_km(lam, width, p.zero_dispersion_nm, p.dispersion_slope) * 1e-12 * p.length_m / 1e3

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Baseband (intensity) transfer function, fiber loss excluded."""
        return self.response(f, self.chromatic_rms_s(x))

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        if not x.fs:
            raise BlockError("multimode_fiber needs the sample rate (fs) of the signal.")
        sigma_cd = self.chromatic_rms_s(x)
        e = x.data
        power = np.sum(np.abs(e) ** 2, axis=1) if e.ndim == 3 else np.abs(e) ** 2
        phase = np.angle(e[:, 0] if e.ndim == 3 else e)
        coupled = 1.0
        if self.groups is not None:
            coupled = float(self.groups[0].sum()) if p.launch == "gaussian" else 1.0
        loss = 10 ** (-self.loss_db_km * p.length_m / 1e3 / 10) * coupled
        out = apply_frequency_response(power, lambda f: self.response(f, sigma_cd), x.fs, pad=64) * loss
        field = np.sqrt(np.maximum(out, 0.0)) * np.exp(1j * phase)
        self._last = {"modal_rms_ps": self.modal_rms_s() * 1e12, "chromatic_rms_ps": sigma_cd * 1e12,
                      "coupled_fraction": coupled}
        return {"out": x.evolve(field, power=(x.meta.get("power") or float(np.mean(power))) * loss)}

    def metrics(self) -> dict[str, float]:
        if not hasattr(self, "_last"):
            return {}
        m = dict(self._last)
        total = np.hypot(m["modal_rms_ps"], m["chromatic_rms_ps"])
        m["bandwidth_ghz"] = GAUSS_BW_RMS / (total * 1e-12) / 1e9 if total > 0 else float("inf")
        return m


# ---------------------------------------------------------------------- VCSEL
vcsel_response = laser_response  # the VCSEL uses the generic small-signal laser response


@register_block("vcsel", category="optical")
class VCSEL(Block):
    """Directly modulated 850 nm VCSEL for multimode data-center links.

    The real drive (NRZ, PAM-4...) is normalized to [-1, 1], filtered by the small-signal
    laser response and mapped to optical power P = P_avg (1 + m x) with modulation depth
    m = (ER - 1) / (ER + 1); power below threshold is clipped to zero. Relative intensity
    noise is added, and the RMS spectral width is published as ``spectral_width_nm`` for
    the chromatic dispersion of a ``multimode_fiber``. The group delay of the laser
    response is added to the ``delay`` metadata so that receivers sample at the eye centre.
    """

    class Params(BlockParams):
        power_dbm: float = Field(0.0, description="Average optical power.")
        extinction_ratio_db: float = Field(4.0, gt=0)
        resonance_ghz: float = Field(20.0, gt=0, description="Relaxation-oscillation frequency f_r.")
        damping_per_ns: float = Field(100.0, gt=0, description="Damping rate gamma (1/ns).")
        parasitic_ghz: float | None = Field(20.0, gt=0, description="Parasitic RC pole (None = ideal chip).")
        rin_db_hz: float | None = Field(-140.0, description="Relative intensity noise.")
        wavelength_nm: float = Field(850.0, gt=0)
        spectral_width_nm: float = Field(0.4, ge=0, description="RMS spectral width (multimode VCSEL).")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port(Kind.SAMPLES, doc="Optical field (sqrt(W)).")}

    def response(self, f: np.ndarray) -> np.ndarray:
        p = self.params
        return vcsel_response(f, p.resonance_ghz * 1e9, p.damping_per_ns * 1e9,
                              p.parasitic_ghz * 1e9 if p.parasitic_ghz else None)

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray:
        """Small-signal modulation response (drive current to optical power)."""
        return self.response(f)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        p = self.params
        if not x.fs:
            raise BlockError("vcsel needs the sample rate (fs) of the signal.")
        drive = np.real(x.data)
        drive = drive / (np.max(np.abs(drive)) or 1.0)
        drive = apply_frequency_response(drive, self.response, x.fs, pad=256)
        er = 10 ** (p.extinction_ratio_db / 10)
        p_avg = dbm_to_w(p.power_dbm)
        power = p_avg * (1 + (er - 1) / (er + 1) * drive)
        power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, x.fs), 0.0)
        meta = {**x.meta, "optical": True, "wavelength_nm": p.wavelength_nm, "power": p_avg,
                "spectral_width_nm": p.spectral_width_nm, "cd_ps_nm": 0.0, "ase_psd": 0.0, "unit": "√W",
                "delay": x.meta.get("delay", 0.0) + dc_group_delay(self.response, x.fs) * x.fs}
        return {"out": Signal(np.sqrt(power).astype(np.complex128), Kind.SAMPLES, fs=x.fs, meta=meta)}


def vcsel_bandwidth_hz(resonance_hz: float, damping_per_s: float, parasitic_hz: float | None) -> float:
    """3-dB modulation bandwidth of :func:`vcsel_response` (|H|^2 = 1/2), found numerically."""
    def g(f: float) -> float:
        return float(np.abs(vcsel_response(np.array([f]), resonance_hz, damping_per_s, parasitic_hz))[0] ** 2) - 0.5
    return brentq(g, 1.0, 10 * resonance_hz)

