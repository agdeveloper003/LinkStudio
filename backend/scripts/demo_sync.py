"""Demo: receiver synchronization - timing recovery, frame/CFO sync, carrier tracking.

    python scripts/demo_sync.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory


def sync_link(delay: float = 9.37, cfo: float = 3000.0, linewidth: float = 10.0) -> cs.Graph:
    """QPSK with preamble through delay + CFO + phase noise, fully synchronized receiver."""
    g = cs.Graph("synchronized-qpsk")
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=1e6)
    g.create("pre", "preamble", length=64, guard=16)
    g.create("tx", "pulse_shaper", sps=4, span=10, rolloff=0.35)
    g.create("tim", "timing_offset", delay_samples=delay)
    g.create("cfo", "carrier_offset", cfo_hz=cfo, random_phase=True)
    g.create("pn", "phase_noise", linewidth_hz=linewidth)
    g.create("awgn", "awgn", mode="ebn0", snr_db=8.0)
    g.create("mf", "matched_filter", decimate=False)
    g.create("tr", "timing_recovery")
    g.create("fs", "frame_sync")
    g.create("cr", "carrier_recovery")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "pre", "tx", "tim", "cfo", "pn", "awgn", "mf", "tr", "fs", "cr", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Synchronization: timing, frame, carrier")

    section("Constellation after each receiver stage (Eb/N0 = 15 dB, delay 9.37 samples, CFO 3 kHz)")
    g = sync_link()
    g.set_param("awgn.snr_db", 15.0)
    res = cs.Simulation(g, batch_size=4, seed=3).run(n_batches=1, capture=["tr.out", "fs.out", "cr.out"])
    naive = cs.Graph("naive")
    for name in ["src", "map", "pre", "tx", "tim", "cfo", "pn", "awgn"]:
        naive.add(name, type(g[name])(**g[name].params.model_dump()))
    naive.create("mf", "matched_filter")
    naive.chain("src", "map", "pre", "tx", "tim", "cfo", "pn", "awgn", "mf")
    raw = cs.Simulation(naive, batch_size=4, seed=3).run(n_batches=1, capture=["mf.out"]).captures["mf.out"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    stages = [("Matched filter, no sync", raw), ("After timing recovery", res.captures["tr.out"]),
              ("After frame + CFO sync", res.captures["fs.out"]), ("After carrier PLL", res.captures["cr.out"])]
    for ax, (title, sig) in zip(axes, stages):
        viz.plot_constellation(sig, ax=ax, title=title, max_points=3000)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    ctx.save(fig, "stages")
    print(f"  estimated CFO {res['fs.cfo_estimate']:.0f} Hz (true 3000 Hz), "
          f"timing offset {res['tr.timing_offset_sym']:.3f} symbols, frame offset {res['fs.frame_offset']:.1f}")
    ctx.check("CFO estimate within 100 Hz", abs(res["fs.cfo_estimate"] - 3000) < 100, f"{res['fs.cfo_estimate']:.0f} Hz")

    section("BER of the synchronized receiver vs ideal coherent detection")
    ebn0 = np.arange(0, 9, 2)
    stop = cs.StopCriteria.errors("ber.bit_errors", 150 if ctx.quick else 400, max_batches=200)
    sweep = cs.Sweep(sync_link(), {"awgn.snr_db": ebn0.tolist()}, batch_size=32, stop=stop,
                     metrics=["ber.ber", "ber.bit_errors", "fs.cfo_estimate"]).run()
    print(sweep.table())
    # Below ~3 dB the decision-directed loops start to slip (threshold effect).
    th = theory.ber_bpsk(ebn0)
    loss = (sweep.column("ber.ber") / th)[ebn0 >= 4]
    ctx.check("Synchronized BER within 2x of theory (Eb/N0 >= 4 dB)", bool(np.all(loss < 2.0)),
              f"worst ratio {loss.max():.2f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    fine = np.linspace(0, 8, 50)
    viz.plot_ber({"Synchronized receiver (sim)": (ebn0, sweep.column("ber.ber"))},
                 {"Coherent QPSK (theory)": (fine, theory.ber_bpsk(fine))}, ax=ax,
                 title="Delay 9.37 samples, CFO 3 kHz, phase noise 10 Hz")
    ctx.save(fig, "ber")


if __name__ == "__main__":
    _common.run_standalone("sync", run, __doc__)
