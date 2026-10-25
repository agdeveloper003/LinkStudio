"""Multi-core runs, accumulator merging, block frequency responses and probe units."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.api.probes import compute_view
from linkstudio.api.response import block_response
from linkstudio.core import Kind, Signal
from linkstudio.phy import theory


def qpsk_graph(snr_db=4.0, n_bits=2000):
    g = cs.Graph("qpsk")
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("awgn", "awgn", snr_db=snr_db)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", bits_per_symbol=2)
    g.create("pm", "power_meter")
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    g.connect("awgn", "pm")
    return g


def test_block_state_is_merged_like_one_run():
    a, b = qpsk_graph(), qpsk_graph()
    sim_a = cs.Simulation(a, batch_size=4, seed=1)
    sim_b = cs.Simulation(b, batch_size=4, seed=2)
    sim_a.run(n_batches=3)
    sim_b.run(n_batches=2)
    merged = qpsk_graph()
    merged.reset(0)
    merged.merge_state(a.export_state())
    merged.merge_state(b.export_state())
    m, ma, mb = merged.metrics(), a.metrics(), b.metrics()
    assert m["ber.bit_errors"] == ma["ber.bit_errors"] + mb["ber.bit_errors"]
    assert m["ber.bits"] == ma["ber.bits"] + mb["ber.bits"] == 5 * 4 * 2000
    assert m["ber.ber"] == pytest.approx(m["ber.bit_errors"] / m["ber.bits"])
    assert m["pm.papr_db_max"] == pytest.approx(max(ma["pm.papr_db_max"], mb["pm.papr_db_max"]))


def test_parallel_run_is_independent_of_worker_count():
    g = qpsk_graph()
    r2 = cs.Simulation(g, batch_size=4, seed=3).run(n_batches=12, workers=2)
    r4 = cs.Simulation(g, batch_size=4, seed=3).run(n_batches=12, workers=4)
    assert r2.n_batches == r4.n_batches == 12
    assert r2.metrics == r4.metrics
    assert r2["ber.bits"] == 12 * 4 * 2000
    p = r2["ber.ber"]  # the awgn default is Eb/N0
    assert p == pytest.approx(theory.ber_bpsk(4.0), rel=0.25)


def test_parallel_stop_on_errors_and_capture():
    g = qpsk_graph()
    stop = cs.StopCriteria.errors("ber.bit_errors", 300, max_batches=200)
    res = cs.Simulation(g, batch_size=2, seed=5).run(stop=stop, workers=3, capture=["awgn.out"])
    assert res.stop_reason.startswith("ber.bit_errors")
    assert res["ber.bit_errors"] >= 300
    assert res.captures["awgn.out"].data.shape == (2, 1000)


def test_parallel_sweep_matches_serial():
    params = {"awgn.snr_db": [0.0, 3.0, 6.0]}
    serial = cs.Sweep(qpsk_graph(), params, batch_size=4, seed=1, n_batches=3).run()
    parallel = cs.Sweep(qpsk_graph(), params, batch_size=4, seed=1, n_batches=3).run(workers=3)
    assert [r["ber.ber"] for r in serial] == [r["ber.ber"] for r in parallel]


def test_ctle_response_peaking():
    from linkstudio.core.scenario import load_scenario

    sc = load_scenario("scenarios/wireline/10gbase_kr_backplane.yaml")
    res = block_response(sc.graph, "ctle")
    assert res["available"] and res["source"] == "model"
    mag = np.array(res["magnitude_db"], dtype=float)
    assert mag[0] == pytest.approx(sc.graph["ctle"].params.dc_gain_db, abs=1e-6)
    assert mag.max() > mag[0] + 6  # high-frequency boost
    ch = block_response(sc.graph, "ch")
    assert np.all(np.diff(np.array(ch["magnitude_db"], dtype=float)) <= 1e-9)  # loss grows with f


def test_fiber_response_group_delay_is_dispersion():
    from linkstudio.core.scenario import load_scenario

    sc = load_scenario("scenarios/optical/imdd_pam4.yaml")
    fib = sc.graph["fib"]
    res = block_response(sc.graph, "fib")
    f = np.array(res["f"])
    gd = np.array(res["group_delay_s"])
    assert f[0] < 0 < f[-1]
    # tau(f) = -beta2 L w: slope -2 pi beta2 L (envelope convention of dispersion_operator)
    from linkstudio.phy.optical import beta2

    b2l = beta2(fib.params.dispersion_ps_nm_km * fib.params.length_km, sc.graph["mod"].params.wavelength_nm) * 1e3
    slope = np.polyfit(f, gd, 1)[0]
    assert slope == pytest.approx(-2 * np.pi * b2l, rel=1e-3)
    assert np.allclose(res["magnitude_db"], -fib.params.attenuation_db_km * fib.params.length_km, atol=1e-6)


def test_multipath_response_is_estimated():
    from linkstudio.core.scenario import load_scenario

    sc = load_scenario("scenarios/wireless/ofdm_eva_ls.yaml")
    res = block_response(sc.graph, "ch")
    assert res["available"] and res["source"] == "estimate"
    assert min(res["f"]) < 0  # complex baseband: two-sided


def test_probe_views_carry_si_units():
    fs = 10e9
    x = np.cos(2 * np.pi * 1e9 * np.arange(1024) / fs)[None, :] * 1e-3
    sig = Signal(x, Kind.SAMPLES, fs=fs, meta={"unit": "A", "sps": 8})
    spec = compute_view(sig, "spectrum")
    assert spec["xunit"] == "Hz" and spec["yunit"] == "dBA²/Hz" and min(spec["x"]) >= 0
    assert spec["x"][int(np.argmax(spec["series"]["PSD"]))] == pytest.approx(1e9, rel=0.02)
    time = compute_view(sig, "time")
    assert time["xunit"] == "s" and time["yunit"] == "A" and time["x"][1] == pytest.approx(1e-10)
    eye = compute_view(sig, "eye")
    assert eye["tunit"] == "s" and eye["ui"] == pytest.approx(8 / fs) and eye["unit"] == "A"
    assert max(max(t) for t in eye["traces"]) == pytest.approx(1e-3, rel=0.05)  # raw amplitude, not normalized
    opt = compute_view(Signal(np.ones((1, 64), complex) * 1e-2, Kind.SAMPLES, fs=fs, meta={"unit": "√W"}), "spectrum")
    assert opt["yunit"] == "dBm/Hz"


def test_scope_view_follows_the_signal():
    from linkstudio.api.probes import compute_probes

    sig = Signal(np.ones((1, 64), complex), Kind.SAMPLES, fs=1e9, meta={"sps": 4})
    values = {"tx.out": sig}
    auto = compute_probes(values, [{"key": "tx.out", "view": "auto"}])["tx.out"]
    assert auto["view"] == "spectrum" and "eye" in auto["views"] and "alt" not in auto
    # A view that does not suit the signal (an LLR histogram of samples) falls back to the natural one.
    assert compute_probes(values, [{"key": "tx.out", "view": "histogram"}])["tx.out"]["view"] == "spectrum"
    assert compute_probes(values, [{"key": "tx.out", "view": "eye"}])["tx.out"]["type"] == "eye"
