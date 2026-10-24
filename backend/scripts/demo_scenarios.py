"""Demo: scenario files, reusable subsystems and serialization.

    python scripts/demo_scenarios.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import SCENARIO_DIR, DemoContext, section

import numpy as np

import linkstudio as cs
from linkstudio.phy import theory


def run(ctx: DemoContext) -> None:
    _common.banner("Scenarios and subsystems")

    section("scenarios/qpsk_awgn_sweep.yaml (sweep defined in YAML)")
    sc = cs.load_scenario(SCENARIO_DIR / "qpsk_awgn_sweep.yaml")
    print(sc.graph.describe())
    res = sc.run(progress=True)
    print(res.table())
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(ctx.out_dir / "scenarios_qpsk_sweep.csv")
    ebn0 = res.column("awgn.snr_db")
    dev = np.max(np.abs(res.column("ber.ber") / theory.ber_bpsk(ebn0) - 1))
    ctx.check("YAML sweep reproduces QPSK theory", dev < 0.3, f"max relative deviation {dev:.1%}")

    section("scenarios/coded_link_subsystems.yaml (TX and RX loaded as subsystems)")
    sc = cs.load_scenario(SCENARIO_DIR / "coded_link_subsystems.yaml")
    print(sc.graph.describe())
    res = sc.run()
    print(res.summary())
    ctx.check("Subsystem link decodes (coded 16-QAM at 6 dB)", res["ber.ber"] < 1e-3, f"BER {res['ber.ber']:.2e}")

    section("Live parameter change inside a subsystem")
    sc.graph.set_param("rx/demap.method", "hard")
    hard = cs.Simulation(sc.graph, batch_size=sc.batch_size, seed=sc.seed).run(n_batches=5)["ber.ber"]
    sc.graph.set_param("rx/demap.method", "maxlog")
    soft = cs.Simulation(sc.graph, batch_size=sc.batch_size, seed=sc.seed).run(n_batches=5)["ber.ber"]
    print(f"  rx/demap.method=hard   -> BER {hard:.2e}")
    print(f"  rx/demap.method=maxlog -> BER {soft:.2e}")
    ctx.check("Soft demapping inside the RX subsystem helps", soft < hard)

    section("scenarios/rrc_16qam_impairments.yaml (waveform chain with RF impairments)")
    sc = cs.load_scenario(SCENARIO_DIR / "rrc_16qam_impairments.yaml")
    res = sc.run()
    print(res.summary())
    ctx.check("Impaired link still works", res["ber.ber"] < 1e-2, f"EVM {res['evm.evm_pct']:.1f}%")

    section("Save a programmatic graph to YAML and load it back")
    g = cs.Graph("roundtrip")
    g.create("src", "prbs", order=15, n_bits=1024)
    g.create("scr", "scrambler")
    g.create("map", "mapper", scheme="qam", order=16)
    g.create("awgn", "awgn", snr_db=12.0)
    g.create("demap", "demapper")
    g.create("dscr", "scrambler")
    g.create("ber", "ber_meter")
    g.chain("src", "scr", "map", "awgn", "demap", "dscr", "ber.rx")
    g.connect("src", "ber.tx")
    path = ctx.out_dir / "roundtrip.yaml"
    cs.save_scenario(cs.Scenario(graph=g, name="roundtrip", seed=5, batch_size=16,
                                 simulation={"n_batches": 3}), path)
    print(path.read_text())
    a = cs.Simulation(g, batch_size=16, seed=5).run(n_batches=3)["ber.bit_errors"]
    b = cs.load_scenario(path).run()["ber.bit_errors"]
    ctx.check("Saved scenario reproduces the same errors", a == b, f"{int(a)} errors")


if __name__ == "__main__":
    _common.run_standalone("scenarios", run, __doc__)
