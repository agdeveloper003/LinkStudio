"""Synchronization, LDPC/polar codes, puncturing, MIMO and time-varying multipath."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.phy import theory
from linkstudio.phy.coding.ldpc import QCLDPCCode
from linkstudio.phy.coding.polar import frozen_mask, polar_encode, sc_decode
from helpers import ber_at


# ------------------------------------------------------------- synchronization
def sync_link(delay=9.37, cfo=3000.0, ebn0=6.0, linewidth=0.0, synced=True):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=1e6)
    g.create("pre", "preamble", length=64, guard=16)
    g.create("tx", "pulse_shaper", sps=4, span=10, rolloff=0.35)
    g.create("tim", "timing_offset", delay_samples=delay)
    g.create("cfo", "carrier_offset", cfo_hz=cfo, random_phase=True)
    g.create("pn", "phase_noise", linewidth_hz=linewidth)
    g.create("awgn", "awgn", snr_db=ebn0)
    chain = ["src", "map", "pre", "tx", "tim", "cfo", "pn", "awgn"]
    if synced:
        g.create("mf", "matched_filter", decimate=False)
        g.create("tr", "timing_recovery")
        g.create("fs", "frame_sync")
        g.create("cr", "carrier_recovery")
        chain += ["mf", "tr", "fs", "cr"]
    else:
        g.create("mf", "matched_filter")
        g.create("strip", "frame_sync", search=0, correct_cfo=False)
        chain += ["mf", "strip"]
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


@pytest.mark.parametrize("delay,cfo", [(0.0, 0.0), (9.37, 3000.0), (3.5, -6000.0)])
def test_synchronized_receiver_close_to_theory(delay, cfo):
    res = cs.Simulation(sync_link(delay, cfo), batch_size=32, seed=1).run(n_batches=4)
    assert res["ber.ber"] < 1.6 * theory.ber_bpsk(6.0)
    assert abs(res["fs.cfo_estimate"] - cfo) < 100.0


def test_unsynchronized_receiver_fails():
    res = cs.Simulation(sync_link(synced=False), batch_size=16, seed=1).run(n_batches=1)
    assert res["ber.ber"] > 0.2


def test_carrier_recovery_tracks_phase_noise():
    res = cs.Simulation(sync_link(delay=2.0, cfo=1000.0, linewidth=20.0), batch_size=32, seed=2).run(n_batches=3)
    assert res["ber.ber"] < 2.5 * theory.ber_bpsk(6.0)


def test_timing_recovery_estimates_fraction():
    g = sync_link(delay=1.0, cfo=0.0, ebn0=20.0)
    res = cs.Simulation(g, batch_size=16, seed=0).run(n_batches=1)
    assert abs(res["tr.timing_offset_sym"] - 0.25) < 0.05  # 1 sample at sps=4
    assert res["fs.frame_offset"] == 0


def test_preamble_layout():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=200)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("pre", "preamble", length=32, guard=8)
    g.chain("src", "map", "pre")
    sig = cs.Simulation(g, batch_size=2).run(n_batches=1, capture=["pre.out"]).captures["pre.out"]
    assert sig.length == 32 + 100 + 8
    assert np.allclose(sig.data[:, :16], sig.data[:, 16:32])
    assert sig.meta["n_payload_symbols"] == 100


# ---------------------------------------------------------------------- codes
def test_ldpc_codewords_satisfy_parity_checks():
    code = QCLDPCCode(4, 4, 16, col_weight=3, seed=3)
    u = np.random.default_rng(0).integers(0, 2, (8, code.k), dtype=np.uint8)
    c = code.encode(u)
    assert c.shape == (8, code.n)
    assert np.array_equal(c[:, :code.k], u)
    assert code.syndrome_ok(c).all()
    bits, iters = code.decode(4.0 * (1.0 - 2.0 * c))
    assert np.array_equal(bits, u) and np.all(iters == 1)


@pytest.mark.parametrize("method", ["min_sum", "sum_product"])
def test_ldpc_corrects_noise(method):
    code = QCLDPCCode(8, 8, 48, seed=1)
    rng = np.random.default_rng(1)
    u = rng.integers(0, 2, (16, code.k), dtype=np.uint8)
    x = 1.0 - 2.0 * code.encode(u)
    sigma2 = 1 / (2 * 0.5 * 10 ** 0.3)  # Eb/N0 = 3 dB, rate 1/2, BPSK
    llr = 2 * (x + np.sqrt(sigma2) * rng.standard_normal(x.shape)) / sigma2
    bits, _ = code.decode(llr, method=method)
    assert np.mean(bits != u) < 1e-3


def test_polar_noiseless_and_frozen_set():
    frozen = frozen_mask(8, 128, 0.0)
    assert (~frozen).sum() == 128
    assert frozen[0] and not frozen[-1]  # first channel worst, last channel best
    rng = np.random.default_rng(0)
    u = np.zeros((4, 256), dtype=np.uint8)
    u[:, ~frozen] = rng.integers(0, 2, (4, 128))
    x = polar_encode(u)
    assert np.array_equal(polar_encode(x), u)  # G_N is an involution
    assert np.array_equal(sc_decode(5.0 * (1.0 - 2.0 * x), frozen), u)


def coded_bpsk(enc, dec, ebn0, n_bits, params=None, puncture=False):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("enc", enc, **(params or {}))
    chain = ["src", "enc"]
    if puncture:
        g.create("punc", "puncturer")
        chain.append("punc")
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("awgn", "awgn", snr_db=ebn0)
    g.create("demap", "demapper", method="exact")
    chain += ["map", "awgn", "demap"]
    if puncture:
        g.create("dpunc", "depuncturer")
        chain.append("dpunc")
    g.create("dec", dec, **(params or {}))
    g.create("ber", "ber_meter")
    g.chain(*chain, "dec", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def test_ldpc_link_gain():
    g = coded_bpsk("ldpc_encoder", "ldpc_decoder", 2.5, 384 * 4)
    res = cs.Simulation(g, batch_size=32, seed=1).run(n_batches=3)
    assert res["ber.ber"] < 3e-3  # uncoded BPSK at 2.5 dB: 2.3e-2
    assert res["dec.avg_iterations"] < 20


def test_polar_link_gain():
    p = {"block_length": 256, "k": 128, "design_snr_db": 1.0}
    res = cs.Simulation(coded_bpsk("polar_encoder", "polar_decoder", 3.0, 512, p), batch_size=32,
                        seed=1).run(n_batches=3)
    assert res["ber.ber"] < 6e-3  # uncoded BPSK at 3 dB: 2.3e-2


def test_puncturing_rate_and_decoding():
    g = coded_bpsk("conv_encoder", "viterbi_decoder", 6.0, 1002, puncture=True)
    caps = cs.Simulation(g, batch_size=8, seed=0).run(n_batches=1, capture=["punc.out", "dpunc.out"]).captures
    assert np.isclose(caps["punc.out"].meta["code_rate"], 1002 / caps["punc.out"].length)
    assert caps["dpunc.out"].length == 2 * (1002 + 6)
    assert cs.Simulation(g, batch_size=16, seed=0).run(n_batches=2)["ber.ber"] < 1e-3


# ----------------------------------------------------------------------- MIMO
def mimo_link(kind, ebn0, nr=2, method="zf", variation="fast"):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=2)
    chain = ["src", "map"]
    if kind == "smux":
        g.create("dmx", "stream_demux", n_streams=2)
        chain.append("dmx")
    elif kind == "alamouti":
        g.create("stbc", "alamouti_encoder")
        chain.append("stbc")
    g.create("ch", "mimo_channel", n_rx=nr, variation=variation)
    g.create("awgn", "awgn", snr_db=ebn0)
    chain += ["ch", "awgn"]
    g.create("det", "alamouti_decoder" if kind == "alamouti" else "mimo_detector",
             **({} if kind == "alamouti" else {"method": method}))
    chain.append("det")
    g.connect("ch.csi", "det.csi")
    if kind != "alamouti":
        g.create("mux", "stream_mux")
        chain.append("mux")
    g.create("demap", "demapper", method="maxlog")
    g.create("ber", "ber_meter")
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def test_simo_mrc_matches_theory():
    assert np.isclose(ber_at(mimo_link("simo", 8.0), errors=500), theory.ber_bpsk_mrc(2, 8.0), rtol=0.2)


def test_spatial_multiplexing_zf_matches_theory():
    assert np.isclose(ber_at(mimo_link("smux", 10.0), errors=500), theory.ber_bpsk_rayleigh(10.0), rtol=0.2)


def test_mimo_detector_ordering():
    zf = ber_at(mimo_link("smux", 10.0, method="zf"))
    mmse = ber_at(mimo_link("smux", 10.0, method="mmse"))
    ml = ber_at(mimo_link("smux", 10.0, method="ml"))
    assert ml < mmse < zf


def test_alamouti_matches_theory():
    g = mimo_link("alamouti", 10.0, nr=1, variation="block")
    stop = cs.StopCriteria.errors("ber.bit_errors", 400, max_batches=1000)
    g.set_param("src.n_bits", 100)  # short frames -> many block-fading realizations
    ber = cs.Simulation(g, batch_size=256, seed=3).run(stop=stop)["ber.ber"]
    assert np.isclose(ber, theory.ber_bpsk_mrc(2, 10.0 - 10 * np.log10(2)), rtol=0.25)


# ------------------------------------------------------ time-varying multipath
def test_time_varying_multipath_with_ofdm():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("ofdm", "ofdm_modulator", n_fft=128, cp_len=16, n_active=96, pilot_spacing=4)
    g.create("ch", "multipath", preset="EVA", variation="jakes", doppler_hz=100)
    g.create("awgn", "awgn", snr_db=20)
    g.create("demod", "ofdm_demodulator")
    g.create("est", "ofdm_channel_estimator", method="perfect")
    g.create("eq", "ofdm_equalizer")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "ofdm", "ch", "awgn", "demod", "est")
    g.connect("ch.csi", "est.csi")
    g.connect("demod", "eq.in")
    g.connect("est", "eq.csi")
    g.chain("eq", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    caps = cs.Simulation(g, batch_size=4, seed=0).run(n_batches=1, capture=["ch.csi"]).captures
    assert caps["ch.csi"].meta["csi_type"] == "cir_tv"
    ber = cs.Simulation(g, batch_size=16, seed=1).run(n_batches=3)["ber.ber"]
    assert ber < 2 * theory.ber_bpsk_rayleigh(20.0)
