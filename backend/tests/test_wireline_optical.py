"""Wireline (copper) and optical blocks."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.phy import theory
from linkstudio.phy.optical import beta2, dbm_to_w
from linkstudio.phy.wireline import (
    CABLE_PRESETS, LOW_FREQUENCY_TERM, cable_response, dielectric_response, waterfill_bits,
)
from helpers import ber_at


def capture(g, batch=4, seed=0):
    return cs.Simulation(g, batch_size=batch, seed=seed).run(n_batches=1, capture="*").captures


# ------------------------------------------------------------------- wireline
@pytest.mark.parametrize("code", ["nrz", "manchester", "ami"])
def test_line_codes_roundtrip(code):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=1000)
    g.create("enc", "line_encoder", code=code)
    g.create("dec", "line_decoder", code=code)
    g.chain("src", "enc", "dec")
    c = capture(g)
    assert np.array_equal(c["dec.out"].data, c["src.out"].data)
    s = c["enc.out"].data
    assert not np.iscomplexobj(s)
    if code in ("manchester", "ami"):
        assert abs(np.mean(s)) < 0.05  # DC-free codes
    if code == "manchester":
        assert s.shape[-1] == 2000


@pytest.mark.parametrize("code,rate", [("4b5b", 4 / 5), ("64b66b", 64 / 66)])
def test_block_line_codes(code, rate):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=640)
    g.create("enc", "line_block_encoder", code=code)
    g.create("dec", "line_block_decoder", code=code)
    g.chain("src", "enc", "dec")
    res = cs.Simulation(g, batch_size=4).run(n_batches=1, capture="*")
    c = res.captures
    assert np.array_equal(c["dec.out"].data, c["src.out"].data)
    assert np.isclose(c["enc.out"].meta["code_rate"], rate)
    assert res["dec.invalid_words"] == 0


def test_64b66b_error_multiplication():
    from linkstudio.phy.wireline import _self_sync_descramble, _self_sync_scramble
    d = np.random.default_rng(0).integers(0, 2, (1, 640), dtype=np.uint8)
    s = _self_sync_scramble(d)
    s[0, 100] ^= 1
    assert np.count_nonzero(_self_sync_descramble(s) != d) == 3  # n, n+39, n+58


def test_nrz_real_awgn_matches_bpsk_theory():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("lc", "line_encoder", code="nrz")
    g.create("awgn", "awgn", snr_db=6.0)
    g.create("ld", "line_decoder")
    g.create("ber", "ber_meter")
    g.chain("src", "lc", "awgn", "ld", "ber.rx")
    g.connect("src", "ber.tx")
    assert not np.iscomplexobj(capture(g)["awgn.out"].data)
    assert np.isclose(ber_at(g), theory.ber_bpsk(6.0), rtol=0.2)


def test_cable_insertion_loss_matches_preset():
    # TIA-568 Cat 5e per 100 m: 1.967 sqrt(f) + 0.023 f + 0.050 / sqrt(f), f in MHz.
    f = np.array([1e6, 10e6, 100e6])
    loss_db = -20 * np.log10(np.abs(cable_response(f, "cat5e", 100.0)))
    a, b = CABLE_PRESETS["cat5e"]
    expected = 0.1 * (a * np.sqrt(f / 1e6) + b * f / 1e6 + LOW_FREQUENCY_TERM["cat5e"] / np.sqrt(f / 1e6))
    assert np.allclose(loss_db, expected, rtol=2e-3)


def test_pic_loop_matches_g9961_insertion_loss():
    # G.996.1 Table 2, loop Mid-C (1829 m of 26 AWG) between 100-ohm terminations at 70 F.
    f = np.array([20, 40, 100, 200, 300, 500, 600, 780, 1100]) * 1e3
    table = np.array([13.3, 16.2, 20.0, 23.4, 26.8, 33.2, 36.3, 41.3, 49.1])
    loss_db = -20 * np.log10(np.abs(cable_response(f, "pic_26awg", 1828.8)))
    assert np.max(np.abs(loss_db - table)) < 0.35


def test_dielectric_loss_is_linear_and_causal():
    f = np.array([1e7, 1e8, 1e9])
    assert np.allclose(-20 * np.log10(np.abs(dielectric_response(f, 1e-9, 1.0))), f * 1e-9, rtol=0.01)
    fs, n = 200e9, 8192
    h = np.fft.ifft(cable_response(np.fft.fftfreq(n, 1 / fs), "backplane", 1.0)).real
    assert np.sum(h[n // 2:] ** 2) / np.sum(h ** 2) < 1e-5  # no energy before t = 0 (second half)
    for ff in (f, -f):  # Hermitian response for real signals
        assert np.allclose(dielectric_response(-ff, 1e-9, 1.0), np.conj(dielectric_response(ff, 1e-9, 1.0)))


def serdes_link(eq: bool):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="pam", order=4, symbol_rate=25e9)
    g.create("tx", "pulse_shaper", shape="rc", sps=8, rolloff=0.5, span=8)
    g.create("ch", "copper_line", preset="backplane", length_m=0.8)
    g.create("awgn", "awgn", mode="snr", snr_db=30.0)
    chain = ["src", "map", "tx", "ch", "awgn"]
    if eq:
        g.create("ctle", "ctle", zero_hz=2e9, pole1_hz=12e9, pole2_hz=40e9)
        chain.append("ctle")
    g.create("rx", "matched_filter", shape="none")
    g.create("agc", "agc", remove_dc=False)
    chain += ["rx", "agc"]
    if eq:
        g.create("dfe", "dfe_equalizer")
        chain.append("dfe")
        g.connect("map", "dfe.ref")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", skip_bits=800)
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def test_serdes_ctle_dfe_opens_the_eye():
    raw = cs.Simulation(serdes_link(False), batch_size=8, seed=1).run(n_batches=1)["ber.ber"]
    eq = cs.Simulation(serdes_link(True), batch_size=8, seed=1).run(n_batches=1)["ber.ber"]
    assert raw > 0.02
    assert eq < 1e-3


def test_ctle_peaking():
    blk = cs.create_block("ctle", zero_hz=1e9, pole1_hz=10e9, pole2_hz=1e12)
    h = np.abs(blk.response(np.array([1e6, 9e9])))
    assert 20 * np.log10(h[1] / h[0]) > 15  # ~20 dB of high-frequency boost


def test_waterfill_bits():
    snr = 10 ** (np.array([60, 40, 20, 5]) / 10)
    bits = waterfill_bits(snr, 9.8, 0.0, 14)
    assert list(bits) == sorted(bits, reverse=True) and bits[-1] == 0
    assert all(b in (0, 1) or b % 2 == 0 for b in bits)


def dmt_link(length=2000.0, loading="waterfill"):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=12000)
    g.create("dmt", "dmt_modulator", snr_db=60, line_length_m=length, loading=loading)
    g.create("ch", "copper_line", preset="dsl_0.4mm", length_m=length)
    g.create("awgn", "awgn", mode="snr", snr_db=60)
    g.create("rx", "dmt_demodulator")
    g.create("ber", "ber_meter")
    g.chain("src", "dmt", "ch", "awgn", "rx.in")
    g.connect("ch.csi", "rx.csi")
    g.connect("rx", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def test_dmt_waterfilling():
    res = cs.Simulation(dmt_link(), batch_size=4, seed=1).run(n_batches=2)
    assert res["ber.ber"] < 1e-4
    longer = cs.Simulation(dmt_link(4000.0), batch_size=4, seed=1).run(n_batches=1)
    assert longer["dmt.data_rate_mbps"] < res["dmt.data_rate_mbps"] / 2  # rate falls with loop length
    bits = cs.create_block("dmt_modulator", snr_db=60, line_length_m=2000).bits
    assert bits[0] > bits[-1]  # high tones carry fewer bits


def test_crosstalk_and_echo():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("lc", "line_encoder", symbol_rate=1e6)
    g.create("xt", "crosstalk", model="next", n_disturbers=20)
    g.create("echo", "hybrid_echo", echo_db=-6)
    g.create("ec", "echo_canceller")
    g.chain("src", "lc", "xt", "echo", "ec")
    g.connect("lc", "echo.tx")
    g.connect("lc", "ec.tx")
    res = cs.Simulation(g, batch_size=2).run(n_batches=1)
    assert res["xt.crosstalk_to_signal_db"] < -10
    assert res["ec.power_reduction_db"] > 0.3  # echo removed (residual = far-end signal)


def test_jitter_identity_and_effect():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=200)
    g.create("lc", "line_encoder", symbol_rate=1e9)
    g.create("tx", "pulse_shaper", shape="rc", sps=8)
    g.create("j", "jitter", rj_rms_ui=0.0)
    g.chain("src", "lc", "tx", "j")
    c = capture(g)
    assert np.allclose(c["j.out"].data, c["tx.out"].data)
    g.set_param("j.rj_rms_ui", 0.05)
    c = capture(g)
    assert not np.allclose(c["j.out"].data, c["tx.out"].data)


# -------------------------------------------------------------------- optical
def optical_source(n_bits=2000, rs=10e9, power_dbm=0.0):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("lc", "line_encoder", symbol_rate=rs)
    g.create("tx", "intensity_modulator", power_dbm=power_dbm)
    g.chain("src", "lc", "tx")
    return g


def test_fiber_attenuation_and_metadata():
    g = optical_source()
    g.create("fib", "fiber", length_km=50, dispersion_ps_nm_km=0)
    g.connect("tx", "fib")
    c = capture(g)
    p_in = np.mean(np.abs(c["tx.out"].data) ** 2)
    p_out = np.mean(np.abs(c["fib.out"].data) ** 2)
    assert np.isclose(10 * np.log10(p_in / p_out), 10.0, atol=1e-6)
    assert c["fib.out"].meta["cd_ps_nm"] == 0


def test_dispersion_broadens_gaussian_pulse_and_is_compensated():
    from linkstudio.core import Signal, Kind
    fs, n = 640e9, 8192
    t = (np.arange(n) - n / 2) / fs
    t0 = 10e-12
    pulse = np.exp(-t ** 2 / (2 * t0 ** 2)).astype(np.complex128)[None, :]
    sig = Signal(pulse, Kind.SAMPLES, fs=fs, meta={"wavelength_nm": 1550.0, "power": 1e-3})
    fib = cs.create_block("fiber", length_km=5, attenuation_db_km=0)
    out = fib.process({"in": sig}, None)["out"]

    def rms_width(x):
        p = np.abs(x[0]) ** 2
        mu = np.sum(t * p) / np.sum(p)
        return np.sqrt(np.sum((t - mu) ** 2 * p) / np.sum(p))

    expected = theory.gaussian_pulse_broadening(t0, beta2(17, 1550), 5e3)
    assert np.isclose(rms_width(out.data) / rms_width(pulse), expected, rtol=0.01)
    back = cs.create_block("cd_compensation").process({"in": out}, None)["out"]
    assert np.allclose(back.data, pulse, atol=1e-9)


def test_self_phase_modulation_of_cw():
    from linkstudio.core import Signal, Kind
    p = dbm_to_w(20)
    cw = np.full((1, 1024), np.sqrt(p), dtype=np.complex128)
    sig = Signal(cw, Kind.SAMPLES, fs=10e9, meta={"wavelength_nm": 1550.0, "power": p})
    fib = cs.create_block("fiber", length_km=20, nonlinear=True, dispersion_ps_nm_km=0, steps_per_km=2)
    out = fib.process({"in": sig}, None)["out"]
    alpha = 0.2 / (10 * np.log10(np.e)) / 1e3
    leff = (1 - np.exp(-alpha * 20e3)) / alpha
    assert np.isclose(np.angle(out.data[0, 0]), 1.3e-3 * p * leff, rtol=1e-3)


def test_edfa_osnr():
    g = optical_source(power_dbm=-20)
    g.create("amp", "optical_amplifier", mode="gain", gain_db=20, noise_figure_db=5)
    g.connect("tx", "amp")
    res = cs.Simulation(g, batch_size=2).run(n_batches=1)
    assert np.isclose(res["amp.osnr_db"], 58 - 20 - 5, atol=0.3)  # OSNR ~ 58 + Pin - NF


@pytest.mark.parametrize("p_dbm", [-24.0, -22.0])
def test_imdd_ook_matches_q_factor_theory(p_dbm):
    g = optical_source(n_bits=4000, power_dbm=p_dbm)
    g.set_param("tx.extinction_ratio_db", 10.0)
    g.create("pd", "photodiode", thermal_noise_pa_rthz=20, shot_noise=False, dark_current_na=0)
    g.create("agc", "agc")
    g.create("ld", "line_decoder")
    g.create("ber", "ber_meter")
    g.chain("tx", "pd", "agc", "ld", "ber.rx")
    g.connect("src", "ber.tx")
    sigma = 20e-12 * np.sqrt(10e9 / 2)
    expected = theory.q_to_ber(theory.ook_q_factor(dbm_to_w(p_dbm), 10.0, 0.8, sigma))
    assert np.isclose(ber_at(g, errors=300, batch_size=32), expected, rtol=0.25)


def test_imdd_pam4_with_dispersion_penalty():
    def link(km):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=4000)
        g.create("map", "mapper", scheme="pam", order=4, symbol_rate=28e9)
        g.create("tx", "pulse_shaper", shape="rc", sps=4, rolloff=0.3)
        g.create("mod", "intensity_modulator", power_dbm=3, extinction_ratio_db=8)
        g.create("fib", "fiber", length_km=km)
        g.create("pd", "photodiode")
        g.create("rx", "matched_filter", shape="none")
        g.create("agc", "agc")
        g.create("demap", "demapper", method="hard")
        g.create("ber", "ber_meter")
        g.chain("src", "map", "tx", "mod", "fib", "pd", "rx", "agc", "demap", "ber.rx")
        g.connect("src", "ber.tx")
        return g
    short = cs.Simulation(link(2), batch_size=8, seed=1).run(n_batches=1)["ber.ber"]
    long = cs.Simulation(link(15), batch_size=8, seed=1).run(n_batches=1)["ber.ber"]
    assert short < 1e-3
    assert long > 10 * max(short, 1e-5)  # dispersion-induced power fading closes the eye


def coherent_link(launch_dbm=-10.0, spans=20):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=8000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=32e9)
    g.create("dmx", "stream_demux", n_streams=2)
    g.create("tx", "pulse_shaper", sps=2, rolloff=0.1, span=32)
    g.create("mod", "iq_modulator", power_dbm=launch_dbm, linewidth_hz=100e3)
    g.create("link", "optical_link", n_spans=spans, span_km=80, polarization_rotation=True)
    g.create("rx", "coherent_receiver", frequency_offset_hz=1e9, lo_linewidth_hz=100e3)
    g.create("cdc", "cd_compensation")
    g.create("fo", "fo_compensation")
    g.create("mf", "matched_filter", decimate=False)
    g.create("agc", "agc", remove_dc=False)
    g.create("eq", "adaptive_mimo_equalizer")
    g.create("cpr", "phase_recovery")
    g.create("mux", "stream_mux")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", skip_bits=4 * 600)
    g.chain("src", "map", "dmx", "tx", "mod", "link", "rx", "cdc", "fo", "mf", "agc", "eq", "cpr", "mux",
            "demap", "ber.rx")
    g.connect("dmx", "eq.ref")
    g.connect("dmx", "cpr.ref")
    g.connect("src", "ber.tx")
    return g


def test_coherent_dp_qpsk_matches_osnr_theory():
    res = cs.Simulation(coherent_link(), batch_size=8, seed=2).run(
        stop=cs.StopCriteria.errors("ber.bit_errors", 300, max_batches=20))
    esn0 = theory.osnr_to_esn0_db(res["link.osnr_db"], 32e9)
    expected = theory.ber_qam_esn0(4, esn0)
    assert 1e-4 < expected < 1e-2
    # Adaptive-equalizer misadjustment and phase-noise tracking cost ~0.3 dB (BER < x2.5).
    assert 0.8 * expected < res["ber.ber"] < 2.5 * expected
    assert abs(res["fo.fo_estimate_hz"] + 1e9) < 20e6


def test_fso_turbulence_statistics():
    g = optical_source(n_bits=64)
    g.create("fso", "fso_turbulence", model="lognormal", log_intensity_variance=0.2)
    g.connect("tx", "fso")
    h = capture(g, batch=4000)["fso.csi"].data.real[:, 0] ** 2
    assert np.isclose(np.mean(h), 1.0, rtol=0.05)
    assert np.isclose(np.var(h), np.exp(0.2) - 1, rtol=0.2)  # scintillation index


def test_mlt3_roundtrip_and_levels():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=1000)
    g.create("enc", "line_encoder", code="mlt3")
    g.create("dec", "line_decoder", code="mlt3")
    g.chain("src", "enc", "dec")
    c = capture(g)
    assert np.array_equal(c["dec.out"].data, c["src.out"].data)
    assert set(np.unique(c["enc.out"].data)) <= {-1.0, 0.0, 1.0}


def test_ber_confidence_interval():
    from linkstudio.phy.metrics import wilson_interval
    lo, hi = wilson_interval(100, 100000)
    assert lo < 1e-3 < hi and hi - lo < 5e-4


def test_wdm_cross_phase_modulation():
    def mer(n_channels):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=6000)
        g.create("map", "mapper", scheme="psk", order=4, symbol_rate=32e9)
        g.create("dmx", "stream_demux", n_streams=2)
        g.create("tx", "pulse_shaper", sps=8, rolloff=0.1, span=32)
        g.create("mod", "iq_modulator", power_dbm=2.0, linewidth_hz=0.0)
        g.create("wdm", "wdm_multiplexer", n_channels=n_channels, spacing_ghz=50)
        g.create("link", "optical_link", n_spans=10, span_km=80, nonlinear=True, steps_per_km=0.25)
        g.create("demux", "wdm_demultiplexer")
        g.create("rx", "coherent_receiver", lo_linewidth_hz=0.0)
        g.create("cdc", "cd_compensation")
        g.create("mf", "matched_filter", decimate=False)
        g.create("ds", "downsampler", factor=4)
        g.create("agc", "agc", remove_dc=False)
        g.create("eq", "adaptive_mimo_equalizer")
        g.create("cpr", "phase_recovery")
        g.create("mux", "stream_mux")
        g.create("evm", "evm_meter", skip_symbols=1200)
        g.chain("src", "map", "dmx", "tx", "mod", "wdm", "link", "demux", "rx", "cdc", "mf", "ds", "agc", "eq",
                "cpr", "mux", "evm.in")
        g.connect("dmx", "eq.ref")
        g.connect("dmx", "cpr.ref")
        g.connect("map", "evm.ref")
        return cs.Simulation(g, batch_size=2, seed=1).run(n_batches=1)["evm.mer_db"]
    single, wdm = mer(1), mer(5)
    assert single > 18 and wdm < single - 2  # XPM from the neighbours


def test_ofdm_sync_recovers_delay_and_cfo():
    def link(sync):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=4000)
        g.create("map", "mapper", scheme="qam", order=16)
        g.create("ofdm", "ofdm_modulator", n_fft=64, cp_len=16, n_active=52, pilot_spacing=4, tail_zeros=80,
                 subcarrier_spacing=312.5e3)
        g.create("tim", "timing_offset", delay_samples=37.0)
        g.create("cfo", "carrier_offset", cfo_hz=60e3, random_phase=True)
        g.create("awgn", "awgn", mode="esn0", snr_db=25)
        chain = ["src", "map", "ofdm", "tim", "cfo", "awgn"]
        if sync:
            g.create("sync", "ofdm_sync")
            chain.append("sync")
        g.create("demod", "ofdm_demodulator")
        g.create("est", "ofdm_channel_estimator")
        g.create("eq", "ofdm_equalizer")
        g.create("demap", "demapper", method="hard")
        g.create("ber", "ber_meter")
        g.chain(*chain, "demod", "est")
        g.connect("demod", "eq.in")
        g.connect("est", "eq.csi")
        g.chain("eq", "demap", "ber.rx")
        g.connect("src", "ber.tx")
        return cs.Simulation(g, batch_size=8, seed=1).run(n_batches=1)
    assert link(False)["ber.ber"] > 0.3
    res = link(True)
    assert res["ber.ber"] < 1e-3
    assert abs(res["sync.cfo_estimate"] - 60e3) < 3e3


# ------------------------------------------------- fiber, transmitter and FSO upgrades
def test_dispersion_slope_gives_the_group_delay_of_d_and_s():
    from linkstudio.phy.optical import C_LIGHT, dispersion_operator
    fs, n = 4e12, 1 << 14
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    phase = np.unwrap(np.angle(np.fft.fftshift(dispersion_operator(n, fs, 17.0, 1550.0, 0.057))))
    tau = -np.gradient(phase, 2 * np.pi * f)
    nu0 = C_LIGHT / 1550e-9
    dlam = (C_LIGHT / (nu0 - f) - 1550e-9) * 1e9  # positive NumPy frequency = lower optical frequency
    expected = (17.0 * dlam + 0.057 / 2 * dlam ** 2) * 1e-12  # 1 km
    sel = np.abs(f) < 1e12
    assert np.max(np.abs(tau[sel] - expected[sel])) < 0.05e-12  # of up to 138 ps


def test_waveplate_pmd_has_maxwellian_dgd():
    from linkstudio.phy.optical import pmd_transfer
    dw = 2 * np.pi * 1e8
    t = pmd_transfer(np.random.default_rng(0), 4000, np.array([-dw, 0.0, dw]), 10e-12, 10)
    td = (t[:, 2] - t[:, 0]) / (2 * dw)
    m = 1j * np.einsum("bij,bkj->bik", td, t[:, 1].conj())
    ev = np.linalg.eigvalsh((m + np.conj(np.transpose(m, (0, 2, 1)))) / 2)
    dgd = ev[:, 1] - ev[:, 0]
    assert np.mean(dgd) == pytest.approx(10e-12, rel=0.03)
    assert np.mean(dgd ** 2) / np.mean(dgd) ** 2 == pytest.approx(3 * np.pi / 8, rel=0.03)


def test_nonlinear_phase_step_criterion_keeps_the_soliton():
    from linkstudio.phy.optical import propagate
    t0 = 5e-12
    b2 = abs(beta2(17.0, 1550.0))
    p0 = b2 / (1.3e-3 * t0 ** 2)
    fs, n = 2e12, 4096
    t = (np.arange(n) - n / 2) / fs
    e0 = (np.sqrt(p0) / np.cosh(t / t0))[None, :].astype(complex)
    length = 4 * t0 ** 2 / b2 / 1e3
    out = propagate(e0, fs, length, 0.0, 17.0, 1550.0, True, 1.3, 0.5, 0.0, 0.05)
    assert np.linalg.norm(np.abs(out) - np.abs(e0)) / np.linalg.norm(np.abs(e0)) < 1e-3


def test_cd_compensation_removes_dispersion_slope():
    from linkstudio.core import Kind, Signal
    fs, n = 1e12, 4096
    t = (np.arange(n) - n / 2) / fs
    e = np.exp(-t ** 2 / (2 * (2e-12) ** 2))[None, :].astype(complex) * 1e-2
    sig = Signal(e, Kind.SAMPLES, fs=fs, meta={"optical": True, "wavelength_nm": 1550.0, "power": 1e-4})
    fib = cs.create_block("fiber", length_km=2.0, attenuation_db_km=0.0, dispersion_slope_ps_nm2_km=0.057)
    fib.reset(0, "f")
    y = fib.process({"in": sig}, None)["out"]
    z = cs.create_block("cd_compensation").process({"in": y}, None)["out"]
    assert np.allclose(z.data, e, atol=1e-8)
    assert z.meta["cd_slope_ps_nm2"] == pytest.approx(0.0)


def test_coherent_receiver_splits_lo_between_polarizations():
    from linkstudio.core import Kind, Signal
    e = np.full((1, 2, 64), np.sqrt(0.5e-3), dtype=complex)  # 1 mW over two polarizations
    sig = Signal(e, Kind.SAMPLES, fs=10e9, meta={"optical": True, "power": 1e-3})
    rx = cs.create_block("coherent_receiver", lo_power_dbm=10.0, lo_linewidth_hz=0.0, shot_noise=False,
                         thermal_noise_pa_rthz=0.0, responsivity=1.0)
    rx.reset(0, "rx")
    y = rx.process({"in": sig}, None)["out"]
    # Each polarization beats with half of the 10 mW LO: |i| = R sqrt(P_LO / 2) |E|.
    assert np.allclose(np.abs(y.data), np.sqrt(5e-3) * np.sqrt(0.5e-3))


def test_adiabatic_chirp_shifts_frequency_with_power():
    from linkstudio.core import Kind, Signal
    fs, n = 100e9, 4000
    drive = np.where(np.arange(n) < n // 2, -1.0, 1.0)[None, :]
    sig = Signal(drive, Kind.SAMPLES, fs=fs, meta={})
    tx = cs.create_block("intensity_modulator", power_dbm=0.0, extinction_ratio_db=10.0, adiabatic_chirp_ghz_mw=2.0)
    tx.reset(0, "tx")
    e = tx.process({"in": sig}, None)["out"].data[0]
    freq = np.diff(np.unwrap(np.angle(e))) * fs / (2 * np.pi)
    p1, p0 = 2e-3 / (1 + 0.1), 2e-3 / (1 + 10)  # levels for ER = 10 dB around 1 mW
    assert np.median(freq[n // 2 + 10:]) - np.median(freq[:n // 2 - 10]) == pytest.approx(2e9 * (p1 - p0) / 1e-3,
                                                                                          rel=1e-3)


def test_gamma_gamma_parameters_from_rytov_variance():
    from linkstudio.phy.optical import gamma_gamma_parameters, rytov_variance
    a, b = gamma_gamma_parameters(1.0)
    assert (a, b) == (pytest.approx(4.39, abs=0.01), pytest.approx(2.56, abs=0.01))
    k = 2 * np.pi / 1550e-9
    assert rytov_variance(1e-14, 1550.0, 1000.0) == pytest.approx(1.23e-14 * k ** (7 / 6) * 1000 ** (11 / 6))
    blk = cs.create_block("fso_turbulence", model="gamma_gamma", rytov_variance=1.0)
    blk.reset(1, "t")
    h = blk.draw(200000)
    assert np.var(h) == pytest.approx(1 / a + 1 / b + 1 / (a * b), rel=0.05)


def test_fso_channel_budget_and_pointing_loss():
    from linkstudio.core import Kind, Signal
    from linkstudio.phy.optical import geometric_loss_db, kim_attenuation_db_km
    assert kim_attenuation_db_km(1.0, 1550.0) == pytest.approx(17 * (550 / 1550) ** 0.5)
    assert geometric_loss_db(1000.0, 2.0, 0.08) == pytest.approx(20 * np.log10(2.0 / 0.08))
    assert geometric_loss_db(10.0, 2.0, 0.08) == 0.0  # the aperture catches the whole beam
    sig = Signal(np.ones((20000, 4), dtype=complex), Kind.SAMPLES, fs=1e9,
                 meta={"optical": True, "wavelength_nm": 1550.0, "power": 1.0})
    ch = cs.create_block("fso_channel", distance_m=1000.0, visibility_km=2.0, turbulence="none",
                         pointing_jitter_m=0.3)
    ch.reset(2, "fso")
    y = ch.process({"in": sig}, None)["out"]
    m = ch.metrics()
    assert m["atmospheric_loss_db"] == pytest.approx(kim_attenuation_db_km(2.0, 1550.0))
    # E[exp(-2 r^2 / w^2)] for Rayleigh r with sigma_s: w^2 / (w^2 + 4 sigma_s^2)
    from scipy.special import erf
    wz, a = 1.0, 0.04
    v = np.sqrt(np.pi / 2) * a / wz
    w2 = wz ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2))
    assert m["pointing_loss_db"] == pytest.approx(-10 * np.log10(w2 / (w2 + 4 * 0.09)), abs=0.05)
    assert y.meta["power"] == pytest.approx(10 ** (-(m["atmospheric_loss_db"] + m["geometric_loss_db"]) / 10))
