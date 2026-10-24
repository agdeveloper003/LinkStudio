"""Every scenario file must load, validate and run."""

from pathlib import Path

import numpy as np
import pytest
import yaml

import linkstudio as cs

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
FILES = sorted(p for p in SCENARIO_DIR.rglob("*.yaml") if "subsystems" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=[p.relative_to(SCENARIO_DIR).as_posix() for p in FILES])
def test_scenario_runs(path):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert spec.get("category"), "every scenario declares a category"
    sc = cs.load_scenario(path)
    sc.graph.validate()
    res = cs.Simulation(sc.graph, batch_size=min(sc.batch_size, 4), seed=sc.seed).run(n_batches=1)
    if spec.get("sweep"):
        return  # sweep scenarios start at low SNR on purpose
    for key, value in res.metrics.items():
        if key.endswith(".ber"):
            assert np.isnan(value) or value < 0.05, f"{key} = {value}"
