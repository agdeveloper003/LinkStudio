"""End-to-end 'physics' tests: simulated error rates must match closed-form theory."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.phy import theory
from helpers import ber_at, uncoded_link

RTOL = 0.2  # ~300 errors -> ~6% standard deviation; generous but still meaningful


@pytest.mark.parametrize("scheme,order,ebn0,ref", [
    ("psk", 2, 6.0, theory.ber_bpsk),
    ("psk", 4, 6.0, theory.ber_bpsk),
    ("psk", 8, 10.0, lambda e: theory.ber_psk(8, e)),
    ("qam", 16, 10.0, lambda e: theory.ber_qam(16, e)),
    ("qam", 64, 14.0, lambda e: theory.ber_qam(64, e)),
    ("pam", 4, 10.0, lambda e: theory.ber_pam(4, e)),
])
def test_awgn_ber_matches_theory(scheme, order, ebn0, ref):
    k = int(np.log2(order))
    g = uncoded_link(scheme, order, ebn0, n_bits=k * 1200)
    assert np.isclose(ber_at(g), ref(ebn0), rtol=RTOL)


def test_soft_demapping_signs_consistent_with_hard():
    g_hard = uncoded_link("qam", 16, 8.0, demap="hard")
    g_soft = uncoded_link("qam", 16, 8.0, demap="maxlog")
    assert ber_at(g_hard) == ber_at(g_soft)


def test_pulse_shaped_link_matches_theory():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("tx", "pulse_shaper", shape="rrc", sps=8, rolloff=0.25, span=12)
    g.create("awgn", "awgn", snr_db=6.0, signal_power=1.0 / 8)  # per-sample power of the RRC waveform
    g.create("mf", "matched_filter")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "tx", "awgn", "mf", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    assert np.isclose(ber_at(g), theory.ber_bpsk(6.0), rtol=RTOL)


def test_matched_filter_noise_metadata():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("tx", "pulse_shaper", sps=4)
    g.create("awgn", "awgn", snr_db=10.0, mode="esn0", signal_power=0.25)
    g.create("mf", "matched_filter")
    g.chain("src", "map", "tx", "awgn", "mf")
    caps = cs.Simulation(g, batch_size=32).run(n_batches=1, capture="*").captures
    err = caps["mf.out"].data - caps["map.out"].data
    assert np.isclose(np.mean(np.abs(err) ** 2), 0.1, rtol=0.05)
    assert np.isclose(caps["mf.out"].meta["noise_var"], 0.1, rtol=1e-6)


def test_rayleigh_bpsk_with_zf_matches_theory():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=1000)
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("fad", "flat_fading", variation="fast")
    g.create("awgn", "awgn", snr_db=10.0, signal_power=1.0)
    g.create("eq", "one_tap_equalizer")
    g.create("demap", "demapper", method="maxlog")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "fad", "awgn", "eq", "demap", "ber.rx")
    g.connect("fad.csi", "eq.csi")
    g.connect("src", "ber.tx")
    assert np.isclose(ber_at(g), theory.ber_bpsk_rayleigh(10.0), rtol=RTOL)


def test_ofdm_awgn_matches_theory():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4 * 1000)
    g.create("map", "mapper", scheme="qam", order=16)
    g.create("ofdm", "ofdm_modulator", n_fft=64, cp_len=16, n_active=52, pilot_spacing=6)
    g.create("awgn", "awgn", snr_db=10.0)
    g.create("demod", "ofdm_demodulator")
    g.create("eq", "ofdm_equalizer")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "ofdm", "awgn", "demod", "eq", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    assert np.isclose(ber_at(g, errors=1000), theory.ber_qam(16, 10.0), rtol=0.1)


def test_nominal_power_tracking():
    """The nominal 'power' metadata must match the measured power along a waveform chain."""
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=8000)
    g.create("map", "mapper", scheme="qam", order=16)
    g.create("tx", "pulse_shaper", sps=4, span=16)
    g.create("ofdm", "ofdm_modulator")
    g.chain("src", "map", "tx")
    g.connect("map", "ofdm")
    caps = cs.Simulation(g, batch_size=64).run(n_batches=1, capture="*").captures
    for key in ("map.out", "tx.out", "ofdm.out"):
        sig = caps[key]
        assert np.isclose(np.mean(np.abs(sig.data) ** 2), sig.meta["power"], rtol=0.03), key


@pytest.mark.parametrize("method", ["perfect", "ls"])
def test_ofdm_multipath_equalization(method):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2 * 2000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("ofdm", "ofdm_modulator", n_fft=128, cp_len=32, n_active=100, pilot_spacing=4,
             subcarrier_spacing=15e3)
    g.create("ch", "multipath", preset="EPA")
    g.create("awgn", "awgn", snr_db=20.0, mode="esn0")
    g.create("demod", "ofdm_demodulator")
    g.create("est", "ofdm_channel_estimator", method=method, average_symbols=True)
    g.create("eq", "ofdm_equalizer")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "ofdm", "ch", "awgn", "demod", "est")
    g.connect("demod", "eq.in")
    g.connect("est", "eq.csi")
    g.connect("ch.csi", "est.csi")
    g.chain("eq", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    ber = cs.Simulation(g, batch_size=64, seed=3).run(n_batches=4)["ber.ber"]
    # Per-subcarrier Rayleigh fading at 20 dB Es/N0 (QPSK): theory ~ 5e-3.
    assert ber < 2e-2


@pytest.mark.parametrize("eq", ["linear_equalizer", "lms_equalizer"])
def test_isi_equalizers(eq):
    def link(with_eq: bool) -> cs.Graph:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=4000)
        g.create("map", "mapper", scheme="psk", order=4)
        g.create("ch", "multipath", delays=[0, 1, 2], powers_db=[0, -2, -6], fading=False)
        g.create("awgn", "awgn", snr_db=14.0)
        g.create("demap", "demapper", method="hard")
        g.create("ber", "ber_meter", skip_bits=400)  # exclude the LMS training period
        g.chain("src", "map", "ch", "awgn")
        if with_eq:
            g.create("eq", eq, n_taps=15)
            g.connect("awgn", "eq.in")
            g.connect("ch.csi" if eq == "linear_equalizer" else "map.out",
                      "eq.csi" if eq == "linear_equalizer" else "eq.ref")
            g.chain("eq", "demap", "ber.rx")
        else:
            g.chain("awgn", "demap", "ber.rx")
        g.connect("src", "ber.tx")
        return g

    raw = cs.Simulation(link(False), batch_size=16, seed=2).run(n_batches=2)["ber.ber"]
    equalized = cs.Simulation(link(True), batch_size=16, seed=2).run(n_batches=2)["ber.ber"]
    assert raw > 0.05
    assert equalized < 0.01


def test_conv_coded_soft_vs_hard():
    def link(method: str) -> cs.Graph:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=1000)
        g.create("enc", "conv_encoder")
        g.create("map", "mapper", scheme="psk", order=2)
        g.create("awgn", "awgn", snr_db=3.0, signal_power=1.0)
        g.create("demap", "demapper", method=method)
        g.create("dec", "viterbi_decoder")
        g.create("ber", "ber_meter")
        g.chain("src", "enc", "map", "awgn", "demap", "dec", "ber.rx")
        g.connect("src", "ber.tx")
        return g

    soft = cs.Simulation(link("maxlog"), batch_size=32, seed=5).run(n_batches=10)["ber.ber"]
    hard = cs.Simulation(link("hard"), batch_size=32, seed=5).run(n_batches=10)["ber.ber"]
    assert soft < hard / 3  # ~2 dB soft-decision gain
