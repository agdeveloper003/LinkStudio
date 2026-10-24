"""Demo: digital modulations over AWGN, simulated BER vs closed-form theory.

    python scripts/demo_modulation.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory

SCHEMES = [  # (label, scheme, order, Eb/N0 range, theory function)
    ("BPSK", "psk", 2, np.arange(0, 11, 2), theory.ber_bpsk),
    ("QPSK", "psk", 4, np.arange(0, 11, 2), theory.ber_bpsk),
    ("8-PSK", "psk", 8, np.arange(0, 15, 2), lambda e: theory.ber_psk(8, e)),
    ("16-QAM", "qam", 16, np.arange(0, 15, 2), lambda e: theory.ber_qam(16, e)),
    ("64-QAM", "qam", 64, np.arange(4, 19, 2), lambda e: theory.ber_qam(64, e)),
]


def build_link(scheme: str, order: int, n_symbols: int = 1200) -> cs.Graph:
    """Bits -> mapper -> AWGN -> hard demapper -> BER meter."""
    k = int(np.log2(order))
    g = cs.Graph(f"{scheme}{order}-awgn")
    g.create("src", "random_bits", n_bits=k * n_symbols)
    g.create("map", "mapper", scheme=scheme, order=order)
    g.create("awgn", "awgn", mode="ebn0", snr_db=10.0, signal_power=1.0)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", bits_per_symbol=k)
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Modulation over AWGN: BER vs theory")
    errors = 100 if ctx.quick else 400
    stop = cs.StopCriteria.errors("ber.bit_errors", errors, max_batches=60 if ctx.quick else 300)

    curves, refs = {}, {}
    for label, scheme, order, ebn0, ref in SCHEMES:
        section(f"{label}: Eb/N0 sweep {ebn0[0]}..{ebn0[-1]} dB")
        g = build_link(scheme, order)
        res = cs.Sweep(g, {"awgn.snr_db": ebn0.tolist()}, batch_size=64, seed=1, stop=stop,
                       metrics=["ber.ber", "ber.ser", "ber.bit_errors"]).run()
        sim_ber = res.column("ber.ber")
        th = ref(ebn0)
        print(res.table(["awgn.snr_db", "ber.ber", "ber.ser", "ber.bit_errors", "n_batches"]))
        reliable = res.column("ber.bit_errors") >= 50
        dev = np.max(np.abs(sim_ber[reliable] / th[reliable] - 1)) if reliable.any() else np.nan
        ctx.check(f"{label} BER matches theory", dev < 0.25, f"max relative deviation {dev:.1%}")
        curves[f"{label} (sim)"] = (ebn0, sim_ber)
        fine = np.linspace(ebn0[0], ebn0[-1], 60)
        refs[f"{label} (theory)"] = (fine, ref(fine))

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_ber(curves, refs, ax=ax, title="Uncoded BER over AWGN")
    ax.set_ylim(1e-6, 0.5)
    ctx.save(fig, "ber_awgn")

    section("Constellations at Es/N0 = 20 dB")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (scheme, order) in zip(axes, [("psk", 4), ("psk", 8), ("qam", 16), ("qam", 64)]):
        g = build_link(scheme, order, n_symbols=500)
        g.set_param("awgn.mode", "esn0")
        g.set_param("awgn.snr_db", 20.0)
        sig = cs.Simulation(g, batch_size=4).run(n_batches=1, capture=["awgn.out"]).captures["awgn.out"]
        viz.plot_constellation(sig, ax=ax, title=sig.meta["constellation"].name)
    ctx.save(fig, "constellations")


if __name__ == "__main__":
    _common.run_standalone("modulation", run, __doc__)
