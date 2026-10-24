"""Demo: channel coding - Hamming, convolutional (hard/soft Viterbi), CRC error detection.

    python scripts/demo_coding.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory


def coded_link(code: str, demap: str = "hard", n_bits: int = 1008) -> cs.Graph:
    """BPSK link with an optional code: none | hamming | conv."""
    g = cs.Graph(f"bpsk-{code}-{demap}")
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("awgn", "awgn", mode="ebn0", snr_db=4.0, signal_power=1.0)
    g.create("demap", "demapper", method=demap)
    g.create("ber", "ber_meter")
    if code == "none":
        g.chain("src", "map", "awgn", "demap", "ber.rx")
    else:
        enc, dec = {"hamming": ("hamming_encoder", "hamming_decoder"),
                    "conv": ("conv_encoder", "viterbi_decoder"),
                    "ldpc": ("ldpc_encoder", "ldpc_decoder"),
                    "polar": ("polar_encoder", "polar_decoder")}[code]
        params = {"polar": {"block_length": 512, "k": 256, "design_snr_db": 1.0}}.get(code, {})
        g.create("enc", enc, **params)
        g.create("dec", dec, **params)
        g.chain("src", "enc", "map", "awgn", "demap", "dec", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def crc_link() -> cs.Graph:
    """CRC-24 + conv code + interleaver + QPSK: the CRC must flag every erroneous frame."""
    g = cs.Graph("crc")
    g.create("src", "random_bits", n_bits=976)
    g.create("crc", "crc_encoder", polynomial="crc24a")
    g.create("enc", "conv_encoder")
    g.create("il", "interleaver", kind="random")
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("awgn", "awgn", mode="ebn0", snr_db=2.0)
    g.create("demap", "demapper", method="maxlog")
    g.create("dil", "interleaver", kind="random", inverse=True)
    g.create("dec", "viterbi_decoder")
    g.create("crcd", "crc_decoder", polynomial="crc24a")
    g.create("ber", "ber_meter")
    g.chain("src", "crc", "enc", "il", "map", "awgn", "demap", "dil", "dec", "crcd", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Channel coding")
    errors = 100 if ctx.quick else 300
    ebn0 = np.arange(0, 9, 2 if ctx.quick else 1)

    configs = [("Uncoded", "none", "hard"), ("Hamming(7,4) hard", "hamming", "hard"),
               ("Conv K=7 r=1/2 hard", "conv", "hard"), ("Conv K=7 r=1/2 soft", "conv", "maxlog"),
               ("LDPC n=768 r=1/2", "ldpc", "exact"), ("Polar N=512 r=1/2 (SC)", "polar", "exact")]
    curves = {}
    for label, code, demap in configs:
        section(label)
        stop = cs.StopCriteria.errors("ber.bit_errors", errors,
                                      max_batches=40 if code in ("conv", "ldpc", "polar") else 200)
        n_bits = {"ldpc": 384 * 3, "polar": 256 * 4}.get(code, 1008)
        res = cs.Sweep(coded_link(code, demap, n_bits=n_bits), {"awgn.snr_db": ebn0.tolist()}, batch_size=64,
                       stop=stop, metrics=["ber.ber", "ber.bit_errors"]).run()
        print(res.table())
        curves[label] = (ebn0, res.column("ber.ber"))

    fine = np.linspace(0, 8, 60)
    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_ber(curves, {"Uncoded (theory)": (fine, theory.ber_bpsk(fine)),
                          "Hamming(7,4) hard (approx.)": (fine, theory.ber_hamming_hard(3, fine))},
                 ax=ax, title="Coding gain, BPSK over AWGN")
    ax.set_ylim(1e-6, 0.5)
    ctx.save(fig, "ber_coding")

    at = int(np.where(ebn0 == 4)[0][0])
    b = {label: curves[label][1][at] for label in curves}
    ctx.check("Uncoded BER matches theory at 4 dB",
              abs(b["Uncoded"] / theory.ber_bpsk(4.0) - 1) < 0.25, f"{b['Uncoded']:.2e}")
    ctx.check("Soft Viterbi beats hard Viterbi beats uncoded at 4 dB",
              b["Conv K=7 r=1/2 soft"] < b["Conv K=7 r=1/2 hard"] < b["Uncoded"],
              f"soft {b['Conv K=7 r=1/2 soft']:.1e}, hard {b['Conv K=7 r=1/2 hard']:.1e}")
    # Both codes have rate 1/2; the LDPC waterfall overtakes the convolutional code above ~2.5 dB.
    at = int(np.where(ebn0 == 4)[0][0])
    ldpc, conv = curves["LDPC n=768 r=1/2"][1][at], curves["Conv K=7 r=1/2 soft"][1][at]
    ctx.check("LDPC beats the convolutional code at 4 dB", ldpc < conv, f"LDPC {ldpc:.1e}, conv {conv:.1e}")

    section("CRC-24A error detection (coded QPSK, Eb/N0 = 2 dB)")
    res = cs.Simulation(crc_link(), batch_size=64, seed=2).run(n_batches=5 if ctx.quick else 20)
    print(f"  frames {int(res['ber.frames'])}, frame errors {int(res['ber.frame_errors'])}, "
          f"CRC failures {int(res['crcd.crc_failures'])}, BER {res['ber.ber']:.2e}")
    # Every erroneous payload must be flagged; a few extra failures are frames whose
    # payload decoded correctly but whose CRC bits were hit.
    ctx.check("CRC flags every erroneous frame", res["crcd.crc_failures"] >= res["ber.frame_errors"],
              f"FER {res['ber.fer']:.3f} vs CRC BLER {res['crcd.bler']:.3f}")


if __name__ == "__main__":
    _common.run_standalone("coding", run, __doc__)
