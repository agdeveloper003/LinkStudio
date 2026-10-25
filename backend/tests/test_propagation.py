"""Path loss (ITU-R P.525, 3GPP TR 38.901) and thermal noise."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.core import Kind, Signal
from linkstudio.phy import theory
from linkstudio.phy.channel.propagation import (
    SCENARIO_DEFAULTS, free_space_loss_db, los_probability, tr38901_path_loss_db,
)


def test_free_space_loss_matches_p525():
    # P.525: L = 32.45 + 20 log10(f / MHz) + 20 log10(d / km)
    assert free_space_loss_db(1000.0, 2.4) == pytest.approx(32.45 + 20 * np.log10(2400), abs=0.01)


@pytest.mark.parametrize("scenario", ["uma", "umi"])
def test_dual_slope_los_is_continuous_at_the_breakpoint(scenario):
    h_bs, h_ut, _ = SCENARIO_DEFAULTS[scenario]
    dbp = 4 * (h_bs - 1) * (h_ut - 1) * 3.5e9 / 3e8
    pl, _ = tr38901_path_loss_db(scenario, np.array([True, True]), np.array([dbp * 0.999, dbp * 1.001]), 3.5,
                                 h_bs, h_ut)
    assert abs(pl[1] - pl[0]) < 0.3


@pytest.mark.parametrize("scenario", ["uma", "umi", "inh"])
def test_nlos_is_never_below_los(scenario):
    h_bs, h_ut, (dmin, dmax) = SCENARIO_DEFAULTS[scenario]
    d = np.geomspace(dmin, dmax, 50)
    los, _ = tr38901_path_loss_db(scenario, np.ones(50, bool), d, 3.5, h_bs, h_ut)
    nlos, _ = tr38901_path_loss_db(scenario, np.zeros(50, bool), d, 3.5, h_bs, h_ut)
    assert np.all(nlos >= los)


def test_umi_los_values_follow_table():
    # UMi LOS below the breakpoint: 32.4 + 21 log10(d3D) + 20 log10(fc)
    d3d = np.hypot(100.0, 10.0 - 1.5)
    pl, sigma = tr38901_path_loss_db("umi", np.array([True]), np.array([100.0]), 3.5, 10.0, 1.5)
    assert pl[0] == pytest.approx(32.4 + 21 * np.log10(d3d) + 20 * np.log10(3.5))
    assert sigma[0] == 4.0


def test_block_shadowing_and_los_statistics():
    blk = cs.create_block("path_loss", model="umi", distance_m=60.0, frequency_ghz=3.5)
    blk.setup()
    blk.reset(3, "pl")
    loss, los = blk.draw_loss_db(40000)
    assert np.mean(los) == pytest.approx(float(los_probability("umi", 60.0, 1.5)), abs=0.01)
    pl_los, _ = tr38901_path_loss_db("umi", np.array([True]), np.array([60.0]), 3.5, 10.0, 1.5)
    assert np.std(loss[los]) == pytest.approx(4.0, rel=0.03)
    assert np.mean(loss[los]) == pytest.approx(pl_los[0], abs=0.1)


def test_out_of_range_distance_is_rejected():
    with pytest.raises(cs.BlockError):
        cs.create_block("path_loss", model="inh", distance_m=500.0).setup()


def test_thermal_noise_variance_is_ktf_bandwidth():
    fs = 20e6
    sig = Signal(np.zeros((16, 4096), dtype=np.complex128), Kind.SAMPLES, fs=fs)
    awgn = cs.create_block("awgn", mode="thermal", noise_figure_db=7.0)
    awgn.reset(1, "n")
    y = awgn.process({"in": sig}, None)["out"]
    expected = 1.380649e-23 * 290 * 10 ** 0.7 * fs
    assert y.meta["noise_var"] == pytest.approx(expected)
    assert np.var(y.data) == pytest.approx(expected, rel=0.02)


def test_free_space_link_budget_gives_theoretical_ber():
    # QPSK at 1 Msym/s, 2.4 GHz over 2 km: Es/N0 = P_rx / (k T F R_s).
    rs, p_tx_dbm, nf = 1e6, 7.0, 5.0
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=rs)
    g.create("tx", "gain", gain_db=p_tx_dbm - 30)
    g.create("pl", "path_loss", model="free_space", frequency_ghz=2.4, distance_m=2000.0)
    g.create("awgn", "awgn", mode="thermal", noise_figure_db=nf)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "tx", "pl", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    p_rx = 10 ** ((p_tx_dbm - 30 - float(free_space_loss_db(2000.0, 2.4))) / 10)
    esn0 = p_rx / (1.380649e-23 * 290 * 10 ** (nf / 10) * rs)
    stop = cs.StopCriteria.errors("ber.bit_errors", 400, max_batches=200)
    ber = cs.Simulation(g, batch_size=32, seed=2).run(stop=stop)["ber.ber"]
    expected = float(theory.ber_psk(4, 10 * np.log10(esn0 / 2)))
    assert ber == pytest.approx(expected, rel=0.25)
