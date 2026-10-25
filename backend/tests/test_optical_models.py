"""Optical transmitters, amplifiers, links and detectors against the closed forms of the theory."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.core import Kind, Signal
from linkstudio.phy import theory
from linkstudio.phy.optical import RAMAN_NSP, RamanGain, ase_psd, beta2, dbm_to_w, mzm_field, saturated_gain

HF = theory.photon_energy(1550.0)


def run_block(type_name, sig, **params):
    blk = cs.create_block(type_name, **params)
    blk.reset(0, type_name)
    return blk, blk.process({"in": sig}, None)["out"]


# ------------------------------------------------------------------ transmitters
def test_mzm_sine_law_and_extinction_ratio():
    v = np.linspace(-4, 4, 9)
    assert np.allclose(np.abs(mzm_field(v, 4.0)) ** 2, np.sin(np.pi * v / 8) ** 2)
    assert np.allclose(np.angle(mzm_field(np.array([1.0, 3.0]), 4.0)), 0.0)  # push-pull: chirp-free
    e = np.abs(mzm_field(np.array([0.0, 4.0]), 4.0, extinction_ratio_db=20.0)) ** 2
    assert 10 * np.log10(e[1] / e[0]) == pytest.approx(20.0)
    single = mzm_field(np.array([1.0, 3.0]), 4.0, push_pull=False)
    assert np.allclose(np.angle(single), np.pi * np.array([1.0, 3.0]) / 8)  # single arm: phase pi V / (2 V_pi)


def test_mzm_block_ook_levels():
    drive = np.where(np.arange(4096) % 64 < 32, -1.0, 1.0)[None, :]
    sig = Signal(drive, Kind.SAMPLES, fs=100e9, meta={})
    blk, y = run_block("mach_zehnder_modulator", sig, laser_power_dbm=13.0, insertion_loss_db=4.0, linewidth_hz=0.0,
                       extinction_ratio_db=None)
    p = np.abs(y.data[0]) ** 2
    assert p.max() == pytest.approx(dbm_to_w(9.0), rel=1e-9) and p.min() == pytest.approx(0.0, abs=1e-15)
    assert blk.metrics()["power_dbm"] == pytest.approx(9.0 - 10 * np.log10(2), abs=1e-6)  # half the peak (OOK)


def test_intensity_modulator_pam_levels():
    drive = np.repeat(np.array([-1.0, -1 / 3, 1 / 3, 1.0]), 16)[None, :]
    _, y = run_block("intensity_modulator", Signal(drive, Kind.SAMPLES, fs=10e9, meta={}), power_dbm=0.0,
                     extinction_ratio_db=10 * np.log10(7.0))
    levels = np.unique(np.round(np.abs(y.data[0]) ** 2 * 1e3, 9))
    # P_avg = 1 mW, ER = 7: P_1 = 0.25 mW and equal steps of 0.5 mW (course PAM-M formulas).
    assert np.allclose(levels, [0.25, 0.75, 1.25, 1.75])


# ------------------------------------------------------------- amplifiers, links
def cw(power_w, n=256, fs=100e9, sps=None):
    meta = {"optical": True, "power": power_w, "wavelength_nm": 1550.0}
    if sps:
        meta["sps"] = sps
    return Signal(np.full((1, n), np.sqrt(power_w), dtype=complex), Kind.SAMPLES, fs=fs, meta=meta)


def test_amplifier_osnr_formula_and_saturation():
    blk, _ = run_block("optical_amplifier", cw(1e-6, sps=4), mode="gain", gain_db=20.0, noise_figure_db=5.0)
    g, f = 100.0, 10 ** 0.5
    m = blk.metrics()
    assert m["osnr_db"] == pytest.approx(10 * np.log10(g * 1e-6 / (HF * 12.5e9 * (g - 1) * f)), abs=1e-9)
    assert m["osnr_rs_db"] == pytest.approx(m["osnr_db"] + 10 * np.log10(12.5e9 / 25e9), abs=1e-9)
    assert ase_psd(g, 5.0, 1550.0) == pytest.approx(f / 2 * HF * (g - 1))
    gs = saturated_gain(1000.0, 1e-4, 1e-2)
    assert gs == pytest.approx(1000.0 * np.exp(-(gs - 1) * 1e-2), rel=1e-9) and gs < 1000.0


def test_attenuator_splitter_loss():
    blk, y = run_block("optical_attenuator", cw(1e-3), split_ways=8, excess_loss_db=1.0, loss_db=2.0)
    assert blk.total_loss_db() == pytest.approx(10 * np.log10(8) + 3.0)
    assert y.meta["power"] == pytest.approx(1e-3 * 10 ** (-blk.total_loss_db() / 10))


@pytest.mark.parametrize("gain_db", [None, 15.5])
def test_link_osnr_matches_multispan_formula(gain_db):
    blk, y = run_block("optical_link", cw(1e-3, sps=4), n_spans=10, span_km=75.0, attenuation_db_km=0.2,
                       amplifier_gain_db=gain_db)
    expected = theory.osnr_link_db(0.0, 10, 15.0, 5.0, 12.5e9, gain_db=gain_db)
    assert blk.metrics()["osnr_db"] == pytest.approx(expected, abs=1e-6)


def test_course_exercise_osnr_after_40_spans():
    """0.2 dB/km, 75 km spans, NF 5 dB, 0 dBm, B = R_s = 64 GHz: OSNR_1 ~ 31 dB, OSNR_40 ~ 15 dB."""
    assert theory.osnr_link_db(0.0, 1, 15.0, 5.0, 64e9) == pytest.approx(30.9, abs=0.15)
    assert theory.osnr_link_db(0.0, 40, 15.0, 5.0, 64e9) == pytest.approx(14.9, abs=0.15)
    # Distributed-amplification limit of the notes (1000 km, 100 GHz): 27.3 dB, reproduced with NF = 5 dB.
    assert theory.osnr_distributed_db(0.0, 0.2, 1000.0, 5.0, 100e9) == pytest.approx(27.3, abs=0.1)


def test_raman_gain_and_noise_closed_forms():
    r = RamanGain(0.5, 0.4, 0.25, 100e3)
    ap = 0.25 / (10 * np.log10(np.e)) / 1e3
    leff = (1 - np.exp(-ap * 100e3)) / ap
    assert r.on_off_gain() == pytest.approx(np.exp(0.4e-3 * 0.5 * leff))
    # Without fiber or pump loss the distributed amplifier is a lumped one: n_sp h nu (G - 1).
    flat = RamanGain(0.5, 0.4, 0.0, 20e3)
    assert flat.ase_psd(0.0, 1550.0) == pytest.approx(RAMAN_NSP * HF * (flat.on_off_gain() - 1), rel=1e-6)


def test_raman_assisted_link_lowers_the_equivalent_noise_figure():
    sig = cw(1e-3, sps=4)
    edfa, y0 = run_block("optical_link", sig, n_spans=5, span_km=100.0)
    hybrid, y1 = run_block("optical_link", sig, n_spans=5, span_km=100.0, raman_pump_mw=600.0)
    m = hybrid.metrics()
    assert m["raman_on_off_gain_db"] > 10
    assert m["equivalent_nf_db"] < 2.0  # below the 3 dB quantum limit of a lumped amplifier
    assert m["osnr_db"] > edfa.metrics()["osnr_db"] + 3
    assert y1.meta["power"] == pytest.approx(1e-3)  # still transparent


def test_gn_model_three_db_rule_and_link_metrics():
    rs, nch, df = 32e9, 9, 50e9
    eta = theory.gn_eta(1.3, 0.2, -21.7, rs, 80.0, nch, df)
    # Long spans: the closed form of the course notes with the field loss coefficient.
    a = 0.2 / (20 * np.log10(np.e)) / 1e3
    b2 = 21.7e-27
    long = 4 / (27 * np.pi) * 1.3e-3 ** 2 / (a * b2 * rs ** 2) * np.arcsinh(np.pi ** 2 * b2 * rs ** 2 * nch ** (2 * rs / df)
                                                                        / (4 * a))
    leff, la = (1 - np.exp(-2 * a * 80e3)) / (2 * a), 1 / (2 * a)
    assert eta == pytest.approx(long * (leff / la) ** 2, rel=1e-9)  # exact L_eff for a finite span
    p_ase = 2 * ase_psd(10 ** 1.6, 5.0, 1550.0) * rs
    p_opt = theory.gn_optimum_power(p_ase, eta)
    assert eta * p_opt ** 3 == pytest.approx(p_ase / 2)  # 3 dB rule
    assert -3.0 < 10 * np.log10(p_opt / 1e-3) < 2.0  # typical optimum launch power per channel
    sig = cw(dbm_to_w(1.0), sps=4, fs=4 * rs)
    sig.meta["wdm"] = {"n_channels": nch, "spacing_hz": df}
    blk, _ = run_block("optical_link", sig, n_spans=10, span_km=80.0, dispersion_ps_nm_km=17.0)
    m = blk.metrics()
    eta_b = theory.gn_eta(1.3, 0.2, beta2(17.0, 1550.0) * 1e27, rs, 80.0, nch, df)
    assert m["gsnr_gn_db"] == pytest.approx(float(theory.gsnr_db(dbm_to_w(1.0), 10, p_ase, eta_b)), abs=1e-6)
    assert m["gn_optimum_launch_dbm"] == pytest.approx(10 * np.log10(theory.gn_optimum_power(p_ase, eta_b) / 1e-3))


def test_fiber_type_presets():
    blk = cs.create_block("fiber", fiber_type="nzdsf", length_km=10.0)
    h = blk.frequency_response(np.array([0.0]), cw(1e-3))
    assert 20 * np.log10(abs(h[0])) == pytest.approx(-2.1)
    _, y = run_block("fiber", cw(1e-3), fiber_type="dcf", length_km=5.0, connector_loss_db=0.5)
    assert y.meta["cd_ps_nm"] == pytest.approx(-500.0) and y.meta["power"] == pytest.approx(1e-3 * 10 ** (-0.275))


# --------------------------------------------------------------------- theory
def test_imdd_reference_numbers_of_the_course():
    assert theory.cd_limit_km(16.0, 2.5e9) == pytest.approx(1000.0)
    assert theory.cd_limit_km(16.0, 10e9) == pytest.approx(62.5)
    assert theory.cd_limit_km(16.0, 10e9, bits_per_symbol=2) == pytest.approx(250.0)
    # Quantum limit: 1e-15 needs ~17 photons per bit on average.
    n_avg = -np.log(2e-15) / 2
    p = n_avg * HF * 10e9
    assert theory.ber_quantum_limit(p, 10e9) == pytest.approx(1e-15)
    assert n_avg == pytest.approx(16.9, abs=0.1)
    assert theory.responsivity(0.8, 1550.0) == pytest.approx(1.0, abs=0.01)
    # PAM-2 IM/DD reduces to the OOK Q factor with equal noise on both levels.
    s = 2e-6
    assert theory.pam_imdd_ber(2, 2e-5, 9.0, 0.8, s) == pytest.approx(
        theory.q_to_ber(theory.ook_q_factor(2e-5, 9.0, 0.8, s)))
    # PIN + TIA example of the notes: -20 dBm, 1 A/W, 20 pA/sqrt(Hz), 5 GHz -> 17 dB.
    assert 10 * np.log10(theory.snr_photodiode(1e-5, 1.0, 20e-12, 5e9, shot=False)) == pytest.approx(17.0, abs=0.02)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_pm_qam_approximations_track_the_exact_ber(order):
    snr = {4: 9.8, 16: 16.5, 64: 22.5}[order]  # BER around 1e-3
    exact = theory.ber_qam_esn0(order, snr)
    assert theory.ber_pm_qam(order, snr) == pytest.approx(exact, rel=0.1)


# ------------------------------------------------------------------ detectors
def test_photodiode_tia_voltage_and_responsivity():
    blk, y = run_block("photodiode", cw(1e-4), quantum_efficiency=0.8, transimpedance_ohm=2000.0,
                       thermal_noise_pa_rthz=0.0, shot_noise=False, dark_current_na=0.0)
    assert y.meta["unit"] == "V"
    assert np.mean(y.data) == pytest.approx(2000.0 * theory.responsivity(0.8, 1550.0) * 1e-4)
    apd = cs.create_block("photodiode", apd_gain=20.0, apd_excess_noise="power_law", apd_x=0.7)
    assert apd.excess_noise() == pytest.approx(20 ** 0.7)


def test_coherent_receiver_internal_snr():
    n = 1 << 14
    sig = Signal(np.full((4, n), np.sqrt(1e-4), dtype=complex), Kind.SAMPLES, fs=64e9,
                 meta={"optical": True, "power": 1e-4})
    _, y = run_block("coherent_receiver", sig, lo_linewidth_hz=0.0, shot_noise=False, thermal_noise_pa_rthz=0.0,
                     internal_snr_db=20.0)
    z = y.data
    snr = np.mean(np.abs(np.mean(z, axis=-1)) ** 2) / np.mean(np.var(z, axis=-1))
    assert 10 * np.log10(snr) == pytest.approx(20.0, abs=0.1)


def test_imdd_pam4_simulation_matches_course_formula():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=8000)
    g.create("map", "mapper", scheme="pam", order=4, symbol_rate=10e9)
    g.create("tx", "pulse_shaper", shape="rect", sps=4)
    g.create("mod", "intensity_modulator", power_dbm=-14.0, extinction_ratio_db=10.0)
    g.create("pd", "photodiode", thermal_noise_pa_rthz=20.0, shot_noise=False, dark_current_na=0.0)
    g.create("rx", "matched_filter", shape="none")
    g.create("agc", "agc")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "tx", "mod", "pd", "rx", "agc", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    res = cs.Simulation(g, batch_size=16, seed=4).run(stop=cs.StopCriteria.errors("ber.bit_errors", 400,
                                                                                  max_batches=40))
    sigma = 20e-12 * np.sqrt(40e9 / 2)  # noise over fs/2, one sample per symbol
    expected = theory.pam_imdd_ber(4, dbm_to_w(-14.0), 10.0, 0.8, sigma)
    assert 1e-4 < expected < 1e-2
    assert res["ber.ber"] == pytest.approx(expected, rel=0.25)
