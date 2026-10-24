"""Demo: channel models and impairments - fading, multipath/ISI equalization, RF impairments.

    python scripts/demo_channels.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory


def impairment_link(btype: str | None, params: dict) -> cs.Graph:
    """16-QAM at Es/N0 = 30 dB through one impairment, measured with an EVM meter."""
    g = cs.Graph("impairment")
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="qam", order=16, symbol_rate=1e6)
    chain = ["src", "map"]
    if btype:
        g.create("imp", btype, **params)
        chain.append("imp")
    g.create("awgn", "awgn", mode="esn0", snr_db=30.0, signal_power=1.0)
    g.create("evm", "evm_meter")
    g.chain(*chain, "awgn", "evm.in")
    g.connect("map", "evm.ref")
    return g


def rayleigh_link(equalize: bool = True) -> cs.Graph:
    g = cs.Graph("bpsk-rayleigh")
    g.create("src", "random_bits", n_bits=1000)
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("fad", "flat_fading", variation="fast")
    g.create("awgn", "awgn", mode="ebn0", snr_db=10.0, signal_power=1.0)
    g.create("eq", "one_tap_equalizer", method="zf" if equalize else "phase")
    g.create("demap", "demapper", method="maxlog")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "fad", "awgn", "eq", "demap", "ber.rx")
    g.connect("fad.csi", "eq.csi")
    g.connect("src", "ber.tx")
    return g


def isi_link(equalizer: str | None) -> cs.Graph:
    g = cs.Graph("qpsk-isi")
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("ch", "multipath", delays=[0, 1, 2], powers_db=[0.0, -2.0, -6.0], fading=False)
    g.create("awgn", "awgn", mode="ebn0", snr_db=10.0)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", skip_bits=400)
    g.chain("src", "map", "ch", "awgn")
    if equalizer is None:
        g.chain("awgn", "demap")
    else:
        g.create("eq", equalizer, n_taps=15)
        g.connect("awgn", "eq.in")
        if equalizer == "lms_equalizer":
            g.connect("map", "eq.ref")
        else:
            g.connect("ch.csi", "eq.csi")
        g.chain("eq", "demap")
    g.connect("demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Channels: impairments, fading, multipath")
    stop = cs.StopCriteria.errors("ber.bit_errors", 100 if ctx.quick else 300, max_batches=300)

    section("RF impairments on 16-QAM (Es/N0 = 30 dB)")
    cases = [
        ("Clean", None, {}),
        ("Phase offset 10 deg", "carrier_offset", {"phase_deg": 10.0}),
        ("CFO 200 Hz", "carrier_offset", {"cfo_hz": 200.0}),
        ("Phase noise 20 Hz", "phase_noise", {"linewidth_hz": 20.0}),
        ("IQ imbalance 1 dB / 5 deg", "iq_imbalance", {"amplitude_db": 1.0, "phase_deg": 5.0}),
        ("PA Saleh, IBO 4 dB", "power_amplifier", {"model": "saleh", "input_backoff_db": 4.0}),
        ("ADC 4 bits", "quantizer", {"bits": 4}),
        ("DC offset", "iq_imbalance", {"dc_i": 0.1, "dc_q": -0.05}),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    evms = {}
    for ax, (label, btype, params) in zip(axes.ravel(), cases):
        g = impairment_link(btype, params)
        res = cs.Simulation(g, batch_size=4).run(n_batches=1, capture=["awgn.out"])
        evms[label] = res["evm.evm_pct"]
        viz.plot_constellation(res.captures["awgn.out"], ax=ax, title=f"{label}\nEVM {res['evm.evm_pct']:.1f}%")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        print(f"  {label:<28} EVM = {res['evm.evm_pct']:6.2f} %   MER = {res['evm.mer_db']:5.1f} dB")
    ctx.save(fig, "impairments")
    ctx.check("Every impairment increases EVM", all(v > evms["Clean"] for k, v in evms.items() if k != "Clean"))

    section("Flat Rayleigh fading, BPSK with coherent ZF equalization")
    ebn0 = np.arange(0, 21, 4 if ctx.quick else 2)
    res = cs.Sweep(rayleigh_link(), {"awgn.snr_db": ebn0.tolist()}, batch_size=64, stop=stop,
                   metrics=["ber.ber", "ber.bit_errors"]).run()
    print(res.table())
    th = theory.ber_bpsk_rayleigh(ebn0)
    dev = np.max(np.abs(res.column("ber.ber") / th - 1))
    ctx.check("Rayleigh BER matches theory", dev < 0.25, f"max relative deviation {dev:.1%}")
    fig, ax = plt.subplots(figsize=(7, 5))
    fine = np.linspace(0, 20, 60)
    viz.plot_ber({"Rayleigh + ZF (sim)": (ebn0, res.column("ber.ber"))},
                 {"Rayleigh (theory)": (fine, theory.ber_bpsk_rayleigh(fine)),
                  "AWGN (theory)": (fine, theory.ber_bpsk(fine))}, ax=ax, title="BPSK: AWGN vs Rayleigh")
    ax.set_ylim(1e-6, 0.5)
    ctx.save(fig, "ber_rayleigh")

    section("Time-varying fading (Jakes model), symbol rate 10 kHz")
    fig, ax = plt.subplots(figsize=(8, 4))
    for fd in (5.0, 50.0):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=10000)
        g.create("map", "mapper", scheme="psk", order=2, symbol_rate=1e4)
        g.create("fad", "flat_fading", variation="jakes", doppler_hz=fd, n_sinusoids=32)
        g.chain("src", "map", "fad")
        h = cs.Simulation(g, batch_size=1, seed=4).run(n_batches=1, capture=["fad.csi"]).captures["fad.csi"]
        t = np.arange(h.length) / 1e4
        ax.plot(t * 1e3, 20 * np.log10(np.abs(h.data[0])), label=f"f_D = {fd:g} Hz")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("|h(t)| [dB]")
    ax.set_ylim(-30, 10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Rayleigh envelope")
    ctx.save(fig, "jakes")

    section("Static ISI channel [0, -2, -6] dB: equalizers (QPSK, Eb/N0 = 10 dB)")
    results = {}
    for label, eq in [("No equalizer", None), ("Linear MMSE (known CIR)", "linear_equalizer"),
                      ("NLMS adaptive (200 training symbols)", "lms_equalizer")]:
        results[label] = cs.Simulation(isi_link(eq), batch_size=32, seed=3).run(n_batches=4)["ber.ber"]
        print(f"  {label:<40} BER = {results[label]:.3e}")
    print(f"  {'AWGN only (theory)':<40} BER = {theory.ber_bpsk(10.0):.3e}")
    ctx.check("Equalizers remove most of the ISI",
              results["Linear MMSE (known CIR)"] < results["No equalizer"] / 10
              and results["NLMS adaptive (200 training symbols)"] < results["No equalizer"] / 10)


if __name__ == "__main__":
    _common.run_standalone("channels", run, __doc__)
