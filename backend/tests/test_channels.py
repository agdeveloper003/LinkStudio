"""Channel and impairment blocks."""

import numpy as np
import pytest

import linkstudio as cs


def _symbols_graph(*blocks, n_bits=4000, order=4):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme="psk", order=order)
    names = ["src", "map"]
    for i, (btype, params) in enumerate(blocks):
        g.create(f"b{i}", btype, **params)
        names.append(f"b{i}")
    g.chain(*names)
    return g


def _capture(g, batch=32, seed=0):
    return cs.Simulation(g, batch_size=batch, seed=seed).run(n_batches=1, capture="*").captures


@pytest.mark.parametrize("mode,expected", [("snr", 0.1), ("esn0", 0.1), ("ebn0", 0.05)])
def test_awgn_noise_variance(mode, expected):
    g = _symbols_graph(("awgn", {"snr_db": 10.0, "mode": mode, "signal_power": 1.0}))
    caps = _capture(g)
    noise = caps["b0.out"].data - caps["map.out"].data
    assert np.isclose(np.mean(np.abs(noise) ** 2), expected, rtol=0.03)
    assert np.isclose(caps["b0.out"].meta["noise_var"], expected)


def test_flat_fading_unit_power():
    for variation in ("block", "fast"):
        g = _symbols_graph(("flat_fading", {"variation": variation}))
        h = _capture(g, batch=2000)["b0.csi"].data
        assert np.isclose(np.mean(np.abs(h) ** 2), 1.0, rtol=0.1)


def test_jakes_correlation():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=20000)
    g.create("map", "mapper", scheme="psk", order=2, symbol_rate=1e4)
    g.create("fad", "flat_fading", variation="jakes", doppler_hz=10.0, n_sinusoids=32)
    g.chain("src", "map", "fad")
    h = _capture(g, batch=200)["fad.csi"].data
    assert np.isclose(np.mean(np.abs(h) ** 2), 1.0, rtol=0.1)
    # Slowly varying: adjacent samples almost identical (f_D/fs = 1e-3).
    assert np.mean(np.abs(h[:, 1:] - h[:, :-1]) ** 2) < 1e-3


def test_rician_k_factor():
    g = _symbols_graph(("flat_fading", {"model": "rician", "k_factor_db": 20.0}))
    h = _capture(g, batch=2000)["b0.csi"].data
    assert np.std(np.abs(h)) < 0.1


def test_multipath_preset_and_static_taps():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=30.72e6)
    g.create("ch", "multipath", preset="EVA")
    g.chain("src", "map", "ch")
    caps = _capture(g, batch=500)
    cir = caps["ch.csi"].data
    # Last tap at 77.1 samples: band-limited kernel spanning 8 samples on each side.
    assert cir.shape[1] == int(np.floor(2510e-9 * 30.72e6)) + 9
    assert np.isclose(np.mean(np.sum(np.abs(cir) ** 2, axis=1)), 1.0, rtol=0.1)
    g.set_param("ch.fractional_delays", False)
    cir = _capture(g, batch=500)["ch.csi"].data
    assert cir.shape[1] == int(round(2510e-9 * 30.72e6)) + 1
    assert np.isclose(np.mean(np.sum(np.abs(cir) ** 2, axis=1)), 1.0, rtol=0.1)

    g2 = _symbols_graph(("multipath", {"delays": [0, 2], "powers_db": [0, -6], "fading": False,
                                        "normalize": False}))
    caps = _capture(g2, batch=2)
    x, y = caps["map.out"].data, caps["b0.out"].data
    expected = x.copy()
    expected[:, 2:] += np.sqrt(10 ** -0.6) * x[:, :-2]
    assert np.allclose(y, expected)


@pytest.mark.parametrize("btype,params", [
    ("carrier_offset", {}), ("iq_imbalance", {}), ("timing_offset", {}), ("gain", {}),
    ("power_amplifier", {"input_backoff_db": 40.0}), ("quantizer", {"bits": 16}),
])
def test_neutral_settings_are_transparent(btype, params):
    g = _symbols_graph((btype, params))
    caps = _capture(g, batch=4)
    assert np.allclose(caps["b0.out"].data, caps["map.out"].data, atol=1e-3)


def test_carrier_offset_rotates():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=200)
    g.create("map", "mapper", scheme="psk", order=2, symbol_rate=1000.0)
    g.create("cfo", "carrier_offset", cfo_hz=10.0)
    g.chain("src", "map", "cfo")
    caps = _capture(g, batch=1)
    ratio = caps["cfo.out"].data / caps["map.out"].data
    assert np.allclose(np.angle(ratio[0, 1:] / ratio[0, :-1]), 2 * np.pi * 10 / 1000)


def test_iq_imbalance_creates_image():
    g = _symbols_graph(("iq_imbalance", {"amplitude_db": 1.0, "phase_deg": 5.0}))
    caps = _capture(g, batch=4)
    assert not np.allclose(caps["b0.out"].data, caps["map.out"].data)


def test_pa_compresses_peaks():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4096)
    g.create("map", "mapper", scheme="qam", order=16)
    g.create("ofdm", "ofdm_modulator")
    g.create("pa", "power_amplifier", input_backoff_db=0.0, model="clip")
    g.create("pin", "power_meter")
    g.create("pout", "power_meter")
    g.chain("src", "map", "ofdm", "pa", "pout")
    g.connect("ofdm", "pin")
    res = cs.Simulation(g, batch_size=8).run(n_batches=1)
    assert res["pout.papr_db_mean"] < res["pin.papr_db_mean"] - 3


def test_fractional_timing_offset():
    g = _symbols_graph(("timing_offset", {"delay_samples": 3.0}))
    caps = _capture(g, batch=2)
    assert np.allclose(caps["b0.out"].data[:, 3:], caps["map.out"].data[:, :-3])


# ------------------------------------------------- fading generator and TDL profiles
def test_zheng_xiao_generator_matches_clarke():
    from scipy.special import j0
    from linkstudio.phy.channel.fading import sum_of_sinusoids
    fd = 0.01
    h = sum_of_sinusoids(np.random.default_rng(1), 2000, 600, fd, 8)
    lags = np.arange(0, 200, 20)
    acf = np.array([np.mean(h[:, k:] * np.conj(h[:, :h.shape[1] - k])).real for k in lags])
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.02)
    assert np.allclose(acf, j0(2 * np.pi * fd * lags), atol=0.02)
    cross = [np.mean(h[:, k:].real * h[:, :h.shape[1] - k].imag) for k in lags]
    assert np.max(np.abs(cross)) < 0.02  # in-phase and quadrature uncorrelated


@pytest.mark.parametrize("preset", ["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"])
def test_nr_tdl_profiles_scale_to_the_delay_spread(preset):
    blk = cs.create_block("multipath", preset=preset, delay_spread_ns=300.0, fractional_delays=False)
    fs = 1e12  # fine grid: rounding is negligible
    d, p, _ = blk.tap_profile(fs)
    mean = np.sum(p * d) / fs
    rms = np.sqrt(np.sum(p * (d / fs) ** 2) - mean ** 2)
    assert rms == pytest.approx(300e-9, rel=0.01)


def test_tdl_d_first_tap_is_ricean_with_the_standard_k_factor():
    g = _symbols_graph(("multipath", {"preset": "TDL-D", "delay_spread_ns": 1000.0, "fractional_delays": False}))
    g.set_param("map.symbol_rate", 30.72e6)
    h0 = _capture(g, batch=4000)["b0.csi"].data[:, 0]
    pw = np.abs(h0) ** 2
    gamma = np.var(pw) / np.mean(pw) ** 2  # = (1 + 2K) / (1 + K)^2 for a Ricean amplitude
    k = np.sqrt(1 - gamma) / (1 - np.sqrt(1 - gamma))
    assert 10 * np.log10(k) == pytest.approx(13.3, abs=1.0)


def test_fractional_delay_kernel_is_band_limited_delay():
    from linkstudio.phy.channel.fading import delay_kernel
    first, taps = delay_kernel(5.3)
    w = np.linspace(0, 0.75 * np.pi, 40)
    resp = np.array([np.sum(taps * np.exp(-1j * x * (first + np.arange(taps.size)))) for x in w])
    assert np.max(np.abs(resp - np.exp(-1j * w * 5.3))) < 0.05
    assert delay_kernel(4.0) == (4, pytest.approx(np.ones(1)))
