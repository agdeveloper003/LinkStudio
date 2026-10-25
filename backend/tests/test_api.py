"""HTTP / WebSocket API used by the user interface."""

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from linkstudio.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def simple_graph(snr=6.0):
    return {
        "name": "api-test", "seed": 1, "batch_size": 8,
        "blocks": {
            "src": {"type": "random_bits", "params": {"n_bits": 400}},
            "map": {"type": "mapper", "params": {"scheme": "psk", "order": 4}},
            "awgn": {"type": "awgn", "params": {"snr_db": snr}},
            "demap": {"type": "demapper", "params": {"method": "hard"}},
            "ber": {"type": "ber_meter", "params": {}},
            "const": {"type": "scope", "params": {"view": "constellation"}},
        },
        "edges": [{"source": "src.out", "target": "map.in"}, {"source": "map.out", "target": "awgn.in"},
                  {"source": "awgn.out", "target": "demap.in"}, {"source": "demap.out", "target": "ber.rx"},
                  {"source": "src.out", "target": "ber.tx"}, {"source": "awgn.out", "target": "const.in1"}],
        "probes": [{"key": "awgn.out", "view": "constellation"}],
    }


def wait(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise TimeoutError


def test_catalogue(client):
    data = client.get("/api/blocks").json()
    assert "awgn" in data["blocks"] and "bpsk" in data["theory"]


def test_scenarios_roundtrip(client, tmp_path):
    items = client.get("/api/scenarios").json()
    paths = [i["path"] for i in items]
    assert "qpsk_awgn_sweep.yaml" in paths
    sc = client.get("/api/scenarios/coded_link_subsystems.yaml").json()
    assert sc["blocks"]["tx"]["type"] == "subsystem"
    assert sc["validation"]["ok"]
    assert {p["name"] for p in sc["validation"]["ports"]["tx"]["outputs"]} == {"bits", "symbols"}


def test_save_and_reload(client):
    import shutil

    from linkstudio.api.app import SCENARIO_DIR

    g = simple_graph()
    g["blocks"]["awgn"]["position"] = {"x": 123.0, "y": 45.0}
    g["blocks"]["awgn"]["params"] = {"snr_db": 3.5, "mode": "ebn0"}
    try:
        assert client.put("/api/scenarios/_pytest_tmp/saved.yaml", json=g).json()["path"] == "_pytest_tmp/saved.yaml"
        loaded = client.get("/api/scenarios/_pytest_tmp/saved.yaml").json()
        assert loaded["blocks"]["awgn"]["params"]["snr_db"] == 3.5
        assert loaded["blocks"]["awgn"]["position"] == {"x": 123.0, "y": 45.0}
        assert loaded["probes"] == g["probes"]
        assert client.get("/api/scenarios/..%2Fpyproject.toml").status_code == 400  # no path traversal
    finally:
        shutil.rmtree(SCENARIO_DIR / "_pytest_tmp", ignore_errors=True)


def test_validate_reports_errors(client):
    g = simple_graph()
    g["edges"] = g["edges"][1:]
    res = client.post("/api/validate", json=g).json()
    assert not res["ok"] and "not connected" in res["errors"][0]
    assert client.post("/api/validate", json=simple_graph()).json()["ok"]


def test_simulation_job(client):
    job = client.post("/api/jobs", json={"mode": "simulation", "graph": simple_graph(),
                                         "config": {"n_batches": 3}}).json()
    job = wait(client, job["id"])
    assert job["status"] == "done", job
    assert job["result"]["metrics"]["ber.bits"] == 3 * 8 * 400
    probe = job["result"]["probes"]["const.in1"]
    assert probe["type"] == "scatter"
    # The other views of the signal come along, so the scope can switch plot after the run.
    assert set(probe["alt"]) == set(probe["views"]) - {"constellation"}


def test_sweep_job_with_theory(client):
    job = client.post("/api/jobs", json={"mode": "sweep", "graph": simple_graph(), "config": {
        "params": {"awgn.snr_db": [0, 4, 8]}, "n_batches": 2, "theory": "bpsk"}}).json()
    job = wait(client, job["id"])
    assert job["status"] == "done", job
    assert len(job["records"]) == 3
    assert job["result"]["theory"]["label"].startswith("BPSK")


def test_live_websocket(client):
    with client.websocket_connect("/api/live") as ws:
        ws.send_json({"type": "start", "graph": simple_graph(), "rate": 50})
        assert ws.receive_json()["type"] == "started"
        frame = ws.receive_json()
        assert frame["type"] == "frame" and "ber.ber" in frame["metrics"]
        assert frame["probes"]["const.in1"]["view"] == "constellation"
        ws.send_json({"type": "set_param", "path": "const.view", "value": "time"})
        for _ in range(3):
            frame = ws.receive_json()
        assert frame["probes"]["const.in1"]["view"] == "time"
        ws.send_json({"type": "set_param", "path": "awgn.snr_db", "value": 30.0})
        for _ in range(3):
            frame = ws.receive_json()
        assert frame["metrics"]["ber.ber"] == 0.0
        ws.send_json({"type": "stop"})


def test_layout_and_python_export(client, tmp_path):
    g = simple_graph()
    pos = client.post("/api/layout", json=g).json()["positions"]
    assert set(pos) == set(g["blocks"]) and pos["src"]["x"] < pos["ber"]["x"]
    code = client.post("/api/export/python", json=g).json()["code"]
    assert "cs.Graph(" in code and "g.connect('src.out', 'map.in')" in code
    namespace: dict = {}
    exec(compile(code.replace("print(result.summary())", "RESULT = result"), "export.py", "exec"), namespace)
    assert namespace["RESULT"]["ber.bits"] > 0


def test_scenario_categories(client):
    items = client.get("/api/scenarios").json()
    cats = {i["category"] for i in items}
    assert {"Tutorials", "Wireless", "Coding", "Wireline", "Optical", "Subsystems"} <= cats
