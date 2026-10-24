"""Demo: copper communications - line codes, cable loss, SerDes equalization, DSL (DMT).

    python scripts/demo_wireline.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy.dsp.filters import fir_filter
from linkstudio.phy.wireline import CABLE_PRESETS, cable_response


def serdes_link(stage: str) -> cs.Graph:
    """25 GBd PAM-4 over a 0.8 m backplane trace: none | ctle | ctle+dfe."""
    g = cs.Graph(f"serdes-{stage}")
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="pam", order=4, symbol_rate=25e9)
    g.create("tx", "pulse_shaper", shape="rc", sps=8, rolloff=0.5, span=8)
    g.create("ch", "copper_line", preset="backplane", length_m=0.8)
    g.create("awgn", "awgn", mode="snr", snr_db=30.0)
    chain = ["src", "map", "tx", "ch", "awgn"]
    if stage != "none":
        g.create("ctle", "ctle", zero_hz=2e9, pole1_hz=12e9, pole2_hz=40e9)
        chain.append("ctle")
    g.create("rx", "matched_filter", shape="none")
    g.create("agc", "agc", remove_dc=False)
    chain += ["rx", "agc"]
    if stage == "ctle+dfe":
        g.create("dfe", "dfe_equalizer")
        chain.append("dfe")
        g.connect("map", "dfe.ref")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", skip_bits=800)
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def dmt_link(length_m: float) -> cs.Graph:
    g = cs.Graph("adsl")
    g.create("src", "random_bits", n_bits=12000)
    g.create("dmt", "dmt_modulator", snr_db=60, line_length_m=length_m)
    g.create("ch", "copper_line", preset="dsl_0.4mm", length_m=length_m)
    g.create("awgn", "awgn", mode="snr", snr_db=60)
    g.create("rx", "dmt_demodulator")
    g.create("ber", "ber_meter")
    g.chain("src", "dmt", "ch", "awgn", "rx.in")
    g.connect("ch.csi", "rx.csi")
    g.connect("rx", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Wireline (copper) communications")

    section("Line code spectra (NRZ, Manchester, AMI)")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for code in ("nrz", "manchester", "ami"):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=4000)
        g.create("lc", "line_encoder", code=code, symbol_rate=1e8 * (2 if code == "manchester" else 1))
        g.create("tx", "pulse_shaper", shape="rect", sps=8 // (2 if code == "manchester" else 1))
        g.chain("src", "lc", "tx")
        sig = cs.Simulation(g, batch_size=16).run(n_batches=1, capture=["tx.out"]).captures["tx.out"]
        viz.plot_psd(sig, ax=ax, label=code, nperseg=1024)
    ax.set_xlim(0, 4e8)
    ax.set_ylim(-40, 3)
    ax.set_title("Line codes at 100 Mb/s: AMI and Manchester have no DC content")
    ctx.save(fig, "line_codes_psd")

    section("Cable insertion loss")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    f = np.logspace(5, 10, 300)
    for preset, length in [("dsl_0.4mm", 1000), ("cat5e", 100), ("cat6a", 100), ("rg6_coax", 100), ("backplane", 1)]:
        loss = -20 * np.log10(np.abs(cable_response(f, preset, length)))
        ax.semilogx(f, loss, label=f"{preset}, {length} m")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Insertion loss [dB]")
    ax.set_ylim(0, 80)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ctx.save(fig, "cable_loss")
    a, b = CABLE_PRESETS["cat5e"]
    loss100 = 0.1 * (a * 10 + b * 100)
    print(f"  Cat5e, 100 m at 100 MHz: {loss100:.1f} dB (TIA-568 limit ~22 dB)")
    ctx.check("Cat5e loss at 100 MHz close to the TIA-568 value", 19 < loss100 < 24, f"{loss100:.1f} dB")

    section("SerDes: 25 GBd PAM-4 over a backplane, CTLE + DFE")
    results = {}
    for stage in ("none", "ctle", "ctle+dfe"):
        results[stage] = cs.Simulation(serdes_link(stage), batch_size=8, seed=1).run(n_batches=2)["ber.ber"]
        print(f"  {stage:<9} BER = {results[stage]:.2e}")
    ctx.check("CTLE + DFE equalize the backplane", results["ctle+dfe"] < 1e-3 < results["none"],
              f"{results['none']:.1e} -> {results['ctle+dfe']:.1e}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, stage in zip(axes, ("none", "ctle")):
        caps = cs.Simulation(serdes_link(stage), batch_size=2, seed=1).run(n_batches=1, capture="*").captures
        sig = caps["ctle.out" if stage == "ctle" else "awgn.out"]
        viz.plot_eye(np.real(sig.data) / np.max(np.abs(sig.data)), sps=8, delay=sig.meta.get("delay", 0.0), ax=ax)
        ax.set_title("Received eye, no equalization" if stage == "none" else "After CTLE")
    ctx.save(fig, "serdes_eye")

    section("ADSL-like DMT with water-filling bit loading (0.4 mm pair)")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lengths = [1000, 2000, 3000, 4000] if not ctx.quick else [1000, 3000]
    for L in lengths:
        mod = cs.create_block("dmt_modulator", snr_db=60, line_length_m=L)
        ax.step(mod.tones * mod.params.sample_rate / mod.params.n_fft / 1e6, mod.bits, where="mid",
                label=f"{L / 1000:.0f} km: {mod.metrics()['data_rate_mbps']:.2f} Mb/s")
    ax.set_xlabel("Tone frequency [MHz]")
    ax.set_ylabel("Bits per tone")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Bit loading falls with frequency and loop length")
    ctx.save(fig, "dmt_bitloading")
    res = cs.Simulation(dmt_link(2000), batch_size=4, seed=1).run(n_batches=2)
    print(f"  2 km: {res['dmt.data_rate_mbps']:.2f} Mb/s, BER {res['ber.ber']:.1e} (3 dB margin)")
    ctx.check("DMT link error free at the planned rate", res["ber.ber"] < 1e-4, f"BER {res['ber.ber']:.1e}")


if __name__ == "__main__":
    _common.run_standalone("wireline", run, __doc__)
