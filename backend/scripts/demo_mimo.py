"""Demo: multi-antenna links - receive diversity, Alamouti, spatial multiplexing detectors.

    python scripts/demo_mimo.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory


def mimo_link(kind: str, nr: int = 2, method: str = "zf") -> cs.Graph:
    """BPSK over i.i.d. Rayleigh (fast fading): siso | simo | alamouti | smux."""
    g = cs.Graph(f"mimo-{kind}")
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=2)
    chain = ["src", "map"]
    if kind == "smux":
        g.create("dmx", "stream_demux", n_streams=2)
        chain.append("dmx")
    elif kind == "alamouti":
        g.create("stbc", "alamouti_encoder")
        chain.append("stbc")
    g.create("ch", "mimo_channel", n_rx=nr, variation="fast" if kind != "alamouti" else "block")
    g.create("awgn", "awgn", mode="ebn0", snr_db=10.0)
    chain += ["ch", "awgn"]
    if kind == "alamouti":
        g.create("det", "alamouti_decoder")
    else:
        g.create("det", "mimo_detector", method=method)
    chain.append("det")
    g.connect("ch.csi", "det.csi")
    if kind != "alamouti":
        g.create("mux", "stream_mux")
        chain.append("mux")
    g.create("demap", "demapper", method="maxlog")
    g.create("ber", "ber_meter")
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    if kind == "alamouti":
        g.set_param("src.n_bits", 100)  # short frames: many block-fading realizations
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("MIMO")
    ebn0 = np.arange(0, 21, 5 if ctx.quick else 2.5)
    stop = cs.StopCriteria.errors("ber.bit_errors", 150 if ctx.quick else 400, max_batches=400)
    configs = [("SISO 1x1", "simo", 1, "zf"), ("SIMO 1x2 (MRC)", "simo", 2, "zf"),
               ("Alamouti 2x1", "alamouti", 1, "zf"), ("2x2 SM, ZF", "smux", 2, "zf"),
               ("2x2 SM, MMSE", "smux", 2, "mmse"), ("2x2 SM, ML", "smux", 2, "ml")]
    curves = {}
    for label, kind, nr, method in configs:
        section(label)
        batch = 256 if kind == "alamouti" else 64
        res = cs.Sweep(mimo_link(kind, nr, method), {"awgn.snr_db": ebn0.tolist()}, batch_size=batch,
                       stop=stop, metrics=["ber.ber", "ber.bit_errors"]).run()
        print(res.table())
        curves[label] = (ebn0, res.column("ber.ber"))

    fine = np.linspace(0, 20, 80)
    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_ber(curves, {"Rayleigh L=1 (theory)": (fine, theory.ber_bpsk_rayleigh(fine)),
                          "MRC L=2 (theory)": (fine, theory.ber_bpsk_mrc(2, fine)),
                          "Alamouti 2x1 (theory)": (fine, theory.ber_bpsk_mrc(2, fine - 10 * np.log10(2)))},
                 ax=ax, xlabel="Eb/N0 per receive antenna [dB]", title="BPSK, i.i.d. Rayleigh fading")
    ax.set_ylim(1e-5, 0.5)
    ctx.save(fig, "ber")

    def dev(label: str, ref: np.ndarray) -> float:
        sim = curves[label][1]
        ok = sim > 1e-4
        return float(np.max(np.abs(sim[ok] / ref[ok] - 1)))

    d = dev("SIMO 1x2 (MRC)", theory.ber_bpsk_mrc(2, ebn0))
    ctx.check("1x2 MRC matches theory", d < 0.3, f"max relative deviation {d:.1%}")
    d = dev("2x2 SM, ZF", theory.ber_bpsk_rayleigh(ebn0))
    ctx.check("2x2 ZF matches diversity-1 theory", d < 0.3, f"max relative deviation {d:.1%}")
    at = int(np.argmin(np.abs(ebn0 - 10)))
    b = {k: v[1][at] for k, v in curves.items()}
    ctx.check("ML < MMSE < ZF at 10 dB", b["2x2 SM, ML"] < b["2x2 SM, MMSE"] < b["2x2 SM, ZF"])


if __name__ == "__main__":
    _common.run_standalone("mimo", run, __doc__)
