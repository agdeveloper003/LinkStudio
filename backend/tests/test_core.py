"""Engine core: graph construction, validation, execution, parameters, subsystems, scenarios."""

import numpy as np
import pytest

import linkstudio as cs
from helpers import uncoded_link


def test_chain_and_run():
    g = uncoded_link(snr_db=20)
    res = cs.Simulation(g, batch_size=8, seed=0).run(n_batches=3)
    assert res.n_batches == 3
    assert res["ber.bits"] == 3 * 8 * 2400
    assert res["ber.ber"] == 0.0


def test_reproducible_with_same_seed():
    g = uncoded_link(snr_db=4)
    a = cs.Simulation(g, batch_size=16, seed=42).run(n_batches=5)["ber.bit_errors"]
    b = cs.Simulation(g, batch_size=16, seed=42).run(n_batches=5)["ber.bit_errors"]
    c = cs.Simulation(g, batch_size=16, seed=43).run(n_batches=5)["ber.bit_errors"]
    assert a == b
    assert a != c


def test_missing_input_is_reported():
    g = cs.Graph()
    g.create("map", "mapper")
    with pytest.raises(cs.GraphError, match="not connected"):
        g.validate()


def test_kind_mismatch_is_reported():
    g = cs.Graph()
    g.create("src", "random_bits")
    g.create("awgn", "awgn")
    g.connect("src", "awgn")
    with pytest.raises(cs.GraphError, match="Kind mismatch"):
        g.validate()


def test_cycle_is_reported():
    g = cs.Graph()
    g.create("a", "gain")
    g.create("b", "gain")
    g.connect("a", "b")
    g.connect("b", "a")
    with pytest.raises(cs.GraphError, match="cycle"):
        g.validate()


def test_double_driven_input_rejected():
    g = cs.Graph()
    g.create("s1", "random_bits")
    g.create("s2", "random_bits")
    g.create("map", "mapper")
    g.connect("s1", "map")
    with pytest.raises(cs.GraphError, match="already driven"):
        g.connect("s2", "map")


def test_set_param_validates_and_rebuilds():
    g = uncoded_link()
    g.set_param("map.order", 16)
    g.set_param("map.scheme", "qam")
    assert g["map"].constellation.order == 16
    with pytest.raises(Exception):
        g.set_param("awgn.mode", "nonsense")
    with pytest.raises(cs.BlockError):
        g.set_param("awgn.does_not_exist", 1)


def test_stop_on_errors():
    g = uncoded_link(snr_db=2)
    stop = cs.StopCriteria.errors("ber.bit_errors", 500, max_batches=1000)
    res = cs.Simulation(g, batch_size=4, seed=1).run(stop=stop)
    assert res["ber.bit_errors"] >= 500
    assert res.stop_reason.startswith("ber.bit_errors")


def test_capture_signals():
    g = uncoded_link()
    res = cs.Simulation(g, batch_size=4).run(n_batches=1, capture=["awgn.out", "map.out"])
    assert res.captures["awgn.out"].data.shape == (4, 1200)
    assert res.captures["map.out"].kind == cs.Kind.SYMBOLS


def test_sweep_restores_params_and_is_monotonic():
    g = uncoded_link()
    sweep = cs.Sweep(g, {"awgn.snr_db": [0, 4, 8]}, batch_size=16, n_batches=4)
    res = sweep.run()
    ber = res.column("ber.ber")
    assert np.all(np.diff(ber) < 0)
    assert g.get_param("awgn.snr_db") == 6.0


def test_subsystem_hierarchy():
    tx = cs.Graph("tx")
    tx.create("src", "random_bits", n_bits=1000)
    tx.create("map", "mapper", scheme="psk", order=2)
    tx.chain("src", "map")
    rx = cs.Graph("rx")
    rx.create("demap", "demapper", method="hard")
    rx.create("ber", "ber_meter")
    rx.connect("demap", "ber.rx")

    top = cs.Graph("link")
    top.add("tx", cs.Subsystem(tx, outputs={"bits": "src.out", "symbols": "map.out"}))
    top.create("awgn", "awgn", snr_db=30)
    top.add("rx", cs.Subsystem(rx, inputs={"in": "demap.in", "ref": "ber.tx"}))
    top.connect("tx.symbols", "awgn")
    top.connect("awgn", "rx.in")
    top.connect("tx.bits", "rx.ref")

    res = cs.Simulation(top, batch_size=4).run(n_batches=2, capture=["tx/map.out"])
    assert res["rx/ber.ber"] == 0.0
    assert "tx/map.out" in res.captures
    top.set_param("awgn.snr_db", -5)
    top.set_param("rx/demap.method", "maxlog")
    assert top.get_param("rx/demap.method") == "maxlog"
    assert cs.Simulation(top, batch_size=4).run(n_batches=2)["rx/ber.ber"] > 0.1


def test_scenario_roundtrip(tmp_path):
    g = uncoded_link()
    sc = cs.Scenario(graph=g, name="rt", seed=3, batch_size=8, simulation={"n_batches": 2})
    path = tmp_path / "s.yaml"
    cs.save_scenario(sc, path)
    loaded = cs.load_scenario(path)
    assert set(loaded.graph.blocks) == set(g.blocks)
    assert loaded.graph.get_param("map.scheme") == "psk"
    r1 = sc.run()
    r2 = loaded.run()
    assert r1["ber.bit_errors"] == r2["ber.bit_errors"]


def test_block_catalogue_has_schemas():
    cat = cs.block_catalogue()
    assert "awgn" in cat and "properties" in cat["awgn"]["params"]
    assert cat["mapper"]["outputs"]["out"]["kinds"] == "SYMBOLS"
