"""Rate-equation lasers (DML and VCSEL): checks against closed-form steady states and linearization."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.core import Kind, Signal
from linkstudio.phy import lasers
from linkstudio.phy.lasers import (
    Q_E, RateEquations, SIGMA1, overlap_coefficients, simulate, small_signal_response, static_power,
    steady_state, threshold_current,
)


def dfb(**changes) -> RateEquations:
    p = dict(tau_n=1e-9, tau_p=2.5e-12, g0=3e4, n_tr=4.9e7, epsilon=0.0, beta=0.0, alpha=4.0, eta_opt=0.3,
             wavelength_nm=1550.0)
    p.update(changes)
    return RateEquations(**p)


def test_pi_curve_has_threshold_and_quantum_slope():
    re = dfb()
    ith = threshold_current(re)
    assert ith == pytest.approx(Q_E * (4.9e7 + 1 / (3e4 * 2.5e-12)) / 1e-9)
    # Without spontaneous emission and gain compression the carrier number clamps at
    # threshold: P = eta_opt h nu (I - I_th) / q exactly.
    i = np.array([1.5, 2.0, 4.0]) * ith
    p = static_power(re, i)
    slope = re.eta_opt * 6.62607015e-34 * 299792458.0 / 1550e-9 / Q_E
    assert p == pytest.approx(slope * (i - ith), rel=1e-6)
    assert static_power(re, 0.5 * ith)[0] < 1e-9 * p[0]


def test_dynamics_settle_to_the_static_solution():
    re = dfb(epsilon=1.6e-7, beta=1e-5)
    fs = 400e9
    i = np.where(np.arange(8000) < 4000, 30e-3, 60e-3)[None, :]
    p, _ = simulate(re, i, fs)
    lo, hi = static_power(re, np.array([30e-3, 60e-3]))
    assert p[0, 3990] == pytest.approx(lo, rel=1e-4)
    assert p[0, -10] == pytest.approx(hi, rel=1e-4)
    assert p[0, 4000:4400].max() > 1.1 * hi  # relaxation-oscillation overshoot


def test_small_signal_response_matches_simulated_modulation():
    re = dfb(epsilon=1.6e-7, beta=1e-5)
    fs, n = 1.6e12, 1 << 14
    t = np.arange(n) / fs
    bias, depth = 50e-3, 0.5e-3
    for f in (2e9, 8e9, 14e9):
        f = round(f * n / fs) * fs / n  # integer number of periods in the frame
        p, _ = simulate(re, bias + depth * np.sin(2 * np.pi * f * t)[None, :], fs, preroll=n // 2)
        amp = 2 * np.abs(np.mean(p[0] * np.exp(-2j * np.pi * f * t)))
        h = small_signal_response(re, bias, np.array([0.0, f]))
        p0 = static_power(re, bias)[0]
        dp_di = (static_power(re, bias + 1e-6)[0] - static_power(re, bias - 1e-6)[0]) / 2e-6
        assert amp == pytest.approx(dp_di * depth * abs(h[1]), rel=0.02)


def test_adiabatic_chirp_follows_gain_compression():
    re = dfb(epsilon=1.6e-7, beta=1e-6)
    fs = 400e9
    i = np.where(np.arange(12000) < 6000, 30e-3, 70e-3)[None, :]
    _, phi = simulate(re, i, fs)
    freq = np.diff(phi[0]) * fs / (2 * np.pi)
    (_, _, s0), (_, _, s1) = steady_state(re, 30e-3), steady_state(re, 70e-3)
    # g0 (N - N_T) = (1 + eps S) / tau_p at steady state, so the frequency step is
    # alpha / (4 pi) eps (S1 - S0) / tau_p (Koch-Bowers adiabatic chirp).
    expected = re.alpha / (4 * np.pi) * re.epsilon * (s1[0] - s0[0]) / re.tau_p
    assert np.median(freq[-500:]) - np.median(freq[5000:5900]) == pytest.approx(expected, rel=0.02)


def test_overlap_coefficients_closed_forms():
    h0, h1, h2, h3 = overlap_coefficients(4.0, 3.0)
    assert h0 == pytest.approx(1 - np.exp(-2 * 16 / 9), rel=1e-6)
    from scipy.special import j0
    assert h2 == pytest.approx(h1 / j0(SIGMA1) ** 2)
    # A mode much narrower than the disc sees J0 ~ 1 - (s1 r / R)^2 / 4 with <r^2> = R_m^2 / 2.
    t0, t1, t2, t3 = overlap_coefficients(10.0, 0.3)
    assert t0 == pytest.approx(1.0, abs=1e-4)
    assert t1 == pytest.approx(1 - SIGMA1 ** 2 * 0.045 / 400, rel=1e-4)
    assert t3 == pytest.approx(t2, rel=5e-3)


@pytest.mark.skipif(lasers._njit is None, reason="numba not installed")
def test_compiled_and_python_integrators_agree():
    re = dfb(epsilon=1.6e-7, beta=1e-5)
    i = 40e-3 + 20e-3 * np.sign(np.sin(np.arange(600) / 15))[None, :]
    n0, n1, s = steady_state(re, i[:, 0])
    args = (re.tau_n, re.tau_p, re.g0, re.n_tr, re.epsilon, re.beta, re.alpha, re.h0, re.h1, re.h2, re.h3, 0.0, 0.0)
    a = lasers._integrate(np.ascontiguousarray(i), 2.5e-12, 1, n0.copy(), n1.copy(), np.log(s), *args)
    b = lasers._integrate.py_func(i, 2.5e-12, 1, n0.copy(), n1.copy(), np.log(s), *args)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def vcsel(**params):
    blk = cs.create_block("vcsel", **params)
    blk.reset(0, "laser")
    return blk


def test_vcsel_self_heating_raises_threshold_and_rolls_power_over():
    blk = vcsel()
    i = np.linspace(0, 30e-3, 61)
    hot, _ = blk.light_current(i, ambient_c=80.0)
    cold, t = blk.light_current(i, ambient_c=25.0)
    assert t[-1] > 25.0 + 100  # strong self-heating at 30 mA
    assert np.argmax(cold) < len(i) - 5 and cold[-1] < 0.9 * cold.max()  # thermal rollover
    assert hot.max() < 0.5 * cold.max()
    ith = [i[np.argmax(p > 0.05e-3)] for p in (cold, hot)]
    assert ith[1] > ith[0]  # leakage raises the threshold


def test_vcsel_spatial_hole_burning_changes_the_turn_off():
    fs = 800e9
    drive = np.where((np.arange(16000) // 4000) % 2 == 0, 1.0, -1.0)[None, :]
    sig = Signal(drive, Kind.SAMPLES, fs=fs, meta={})
    out = {}
    for shb in (True, False):
        blk = vcsel(bias_ma=2.75, modulation_ma=2.25, spatial_hole_burning=shb, thermal=False, rin_db_hz=None,
                    parasitic_ghz=None)
        out[shb] = np.abs(blk.process({"in": sig}, None)["out"].data[0]) ** 2
    # After the current falls to 0.5 mA (below threshold) carriers diffuse back into the mode
    # region: with hole burning the light lingers longer than in the uniform model.
    tail = slice(4000 + 200, 4000 + 1200)
    assert out[True][tail].sum() > 1.5 * out[False][tail].sum()


def test_vcsel_block_power_and_metrics():
    fs = 400e9
    drive = np.sin(2 * np.pi * np.arange(8192) / 128)[None, :]
    blk = vcsel(rin_db_hz=None)
    y = blk.process({"in": Signal(drive, Kind.SAMPLES, fs=fs, meta={})}, None)["out"]
    m = blk.metrics()
    leak, temp = blk.operating_point()
    assert temp > 25.0 and y.meta["spectral_width_nm"] == 0.4 and y.meta["unit"] == "√W"
    assert y.meta["power"] == pytest.approx(static_power(blk.re, 7e-3 - leak)[0], rel=0.03)
    assert m["extinction_ratio_db"] > 3 and m["threshold_ma"] < 1.5
