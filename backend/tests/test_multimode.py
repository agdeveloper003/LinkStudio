"""Multimode fiber: checks against the closed-form theory, and the VCSEL data-center lane."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import linkstudio as cs
from linkstudio.core import Kind, Signal
from linkstudio.phy.multimode import (
    GAUSS_BW_RMS, chromatic_rms_ps_per_km, gaussian_launch_weights, group_delay,
    optimum_alpha, overfilled_rms, profile_contrast,
)

DELTA = profile_contrast(0.2, 1.467)


def fiber(**params):
    block = cs.create_block("multimode_fiber", **params)
    block.setup()
    return block


def test_optimum_profile_and_minimum_width_match_olshansky_keck():
    res = minimize_scalar(lambda a: overfilled_rms(a, DELTA), bounds=(1.8, 2.2), method="bounded",
                          options={"xatol": 1e-10})
    assert res.x == pytest.approx(optimum_alpha(DELTA), abs=1e-3)
    # sigma_min = L N1 Delta^2 / (20 sqrt(3) c), relative to L N1 / c
    assert res.fun == pytest.approx(DELTA ** 2 / (20 * np.sqrt(3)), rel=0.03)


def test_delay_expansion_matches_exact_wkb_delay():
    u = np.linspace(0, 1, 101)
    for alpha in (1.9, 2.0, 2.2):
        x = u ** (alpha / (alpha + 2))
        exact = (1 - 4 * DELTA * x / (alpha + 2)) / np.sqrt(1 - 2 * DELTA * x) - 1
        assert np.max(np.abs(group_delay(u, alpha, DELTA) - exact)) < 5 * DELTA ** 3


def test_discrete_mode_groups_converge_to_continuum_width():
    g = 400
    mu = np.arange(1, g + 1)
    w = (2 * mu - 1) / g ** 2
    for alpha in (1.95, 2.0, 2.1):
        d = group_delay(((mu - 0.5) / g) ** 2, alpha, DELTA)
        rms = np.sqrt(np.sum(w * d ** 2) - np.sum(w * d) ** 2)
        assert rms == pytest.approx(overfilled_rms(alpha, DELTA), rel=0.01)


@pytest.mark.parametrize("fiber_type, ofl_mhz_km", [("OM3", 1500.0), ("OM4", 3500.0)])
def test_default_profile_meets_overfilled_bandwidth_of_class(fiber_type, ofl_mhz_km):
    b = fiber(fiber_type=fiber_type, model="mode_groups", length_m=1000.0)
    assert GAUSS_BW_RMS / b.modal_rms_s() == pytest.approx(ofl_mhz_km * 1e6, rel=1e-6)
    assert len(b.groups[0]) in (18, 19)  # about V/2 principal groups for 50 um, NA 0.2 at 850 nm


def test_offset_launch_excites_poisson_group_powers():
    # A displaced fundamental Gaussian is a coherent state: Poisson with mean (r0 / w0)^2.
    w0 = 5.8
    w = gaussian_launch_weights(20, w0, 1.5 * w0, w0)
    assert np.allclose(w[:8], poisson.pmf(np.arange(8), 2.25), atol=1e-4)


def test_centred_mismatched_launch_excites_even_groups_geometrically():
    w0, waist = 5.8, 4.0
    t = ((waist ** 2 - w0 ** 2) / (waist ** 2 + w0 ** 2)) ** 2
    expected = [(1 - t) * t ** (p // 2) if p % 2 == 0 else 0.0 for p in range(8)]
    assert np.allclose(gaussian_launch_weights(20, waist, 0.0, w0)[:8], expected, atol=1e-4)


def test_restricted_launch_has_higher_bandwidth_than_overfilled():
    ofl = fiber(model="mode_groups", length_m=300.0)
    centre = fiber(model="mode_groups", length_m=300.0, launch="gaussian", launch_waist_um=4.0)
    assert centre.modal_rms_s() < ofl.modal_rms_s() / 3


def test_chromatic_width_follows_ieee_link_model():
    # D(850 nm) = S0 lambda / 4 (1 - (lambda0/lambda)^4), about -100 ps/(nm km).
    d1 = 0.101 * 850 / 4 * (1 - (1320 / 850) ** 4)
    assert -110 < d1 < -95
    assert chromatic_rms_ps_per_km(850, 0.4, 1320, 0.101) == pytest.approx(
        0.4 * np.hypot(d1, 0.7 * 0.101 * 0.4))


def gaussian_power_pulse(width_s, fs=2e12, n=8192):
    t = (np.arange(n) - n / 2) / fs
    power = 1e-3 * np.exp(-t ** 2 / (2 * width_s ** 2))
    meta = {"optical": True, "wavelength_nm": 850.0, "power": 1e-3, "spectral_width_nm": 0.4}
    return Signal(np.sqrt(power).astype(np.complex128)[None, :], Kind.SAMPLES, fs=fs, meta=meta), t


def rms_width(power, t):
    w = power / power.sum()
    mean = np.sum(w * t)
    return np.sqrt(np.sum(w * (t - mean) ** 2))


@pytest.mark.parametrize("model", ["bandwidth", "mode_groups"])
def test_pulse_variance_adds_modal_and_chromatic_variances(model):
    sig, t = gaussian_power_pulse(10e-12)
    b = fiber(model=model, length_m=300.0)
    y = b.process({"in": sig}, None)["out"]
    p_out = np.abs(y.data[0]) ** 2
    m = b.metrics()
    expected = np.sqrt(10.0 ** 2 + m["modal_rms_ps"] ** 2 + m["chromatic_rms_ps"] ** 2) * 1e-12
    assert rms_width(p_out, t) == pytest.approx(expected, rel=0.02)
    assert p_out.sum() / (np.abs(sig.data[0]) ** 2).sum() == pytest.approx(10 ** (-3.5 * 0.3 / 10), rel=1e-6)


def test_photodiode_bandwidth_scales_reported_noise():
    fs, n = 100e9, 1 << 15
    sig = Signal(np.zeros((8, n), dtype=np.complex128), Kind.SAMPLES, fs=fs, meta={"optical": True})
    pd = cs.create_block("photodiode", thermal_noise_pa_rthz=20.0, shot_noise=False, dark_current_na=0.0,
                         bandwidth_ghz=10.0)
    pd.setup()
    pd.rng = np.random.default_rng(1)
    y = pd.process({"in": sig}, None)["out"]
    assert np.var(y.data) == pytest.approx(y.meta["noise_var"], rel=0.05)
    # Bessel-Thomson noise bandwidth is a little above the 3-dB bandwidth.
    assert 1.0 < y.meta["noise_var"] / ((20e-12) ** 2 * 10e9) < 1.2


def test_datacenter_link_reach_depends_on_fiber_class():
    sc = cs.load_scenario("scenarios/optical/datacenter_vcsel_mmf.yaml")
    g = sc.graph

    def ber(fiber_type, length_m):
        g.set_param("fib.fiber_type", fiber_type)
        g.set_param("fib.length_m", length_m)
        return cs.Simulation(g, batch_size=4, seed=1).run(n_batches=2)["ber.ber"]

    assert ber("OM4", 100.0) < 1e-3
    assert ber("OM3", 300.0) > 1e-2
