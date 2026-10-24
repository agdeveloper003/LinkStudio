"""Demo: single-carrier waveforms - pulse shaping, spectra, eye diagrams, timing errors.

    python scripts/demo_pulse_shaping.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory
from linkstudio.phy.dsp.filters import fir_filter

SPS = 8


def build_link(rolloff: float = 0.35, shape: str = "rrc", ebn0_db: float = 6.0) -> cs.Graph:
    """QPSK -> pulse shaper -> timing offset -> AWGN -> matched filter -> demapper."""
    g = cs.Graph("qpsk-rrc")
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=1e6)
    g.create("tx", "pulse_shaper", shape=shape, sps=SPS, span=12, rolloff=rolloff)
    g.create("tim", "timing_offset", delay_samples=0.0)
    g.create("awgn", "awgn", mode="ebn0", snr_db=ebn0_db, signal_power=1.0 / SPS)
    g.create("mf", "matched_filter")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "tx", "tim", "awgn", "mf", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Pulse shaping: spectra, eye diagrams, matched filtering")
    stop = cs.StopCriteria.errors("ber.bit_errors", 100 if ctx.quick else 300, max_batches=200)

    section("Spectrum vs roll-off (RRC) and Gaussian pulse")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rolloff in (0.1, 0.35, 1.0):
        g = build_link(rolloff)
        sig = cs.Simulation(g, batch_size=32).run(n_batches=1, capture=["tx.out"]).captures["tx.out"]
        viz.plot_psd(sig, ax=ax, label=f"RRC beta={rolloff}", nperseg=512)
    g = build_link(shape="gaussian")
    sig = cs.Simulation(g, batch_size=32).run(n_batches=1, capture=["tx.out"]).captures["tx.out"]
    viz.plot_psd(sig, ax=ax, label="Gaussian BT=0.3", nperseg=512)
    ax.set_ylim(-60, 3)
    ax.set_title("Transmit PSD (symbol rate 1 MHz)")
    ctx.save(fig, "psd")

    section("Eye diagrams after the receive filter (Es/N0 = 25 dB)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, shape, rolloff, rx) in zip(axes, [
        ("RRC tx + RRC rx (Nyquist)", "rrc", 0.35, "matched"),
        ("RRC tx, no rx filter (ISI)", "rrc", 0.35, "none"),
        ("Gaussian BT=0.3 + matched", "gaussian", 0.35, "matched"),
    ]):
        g = build_link(rolloff, shape)
        g.set_param("awgn.mode", "esn0")
        g.set_param("awgn.snr_db", 25.0)
        sig = cs.Simulation(g, batch_size=2).run(n_batches=1, capture=["awgn.out"]).captures["awgn.out"]
        tx_taps = sig.meta["pulse_taps"]
        rx_taps = tx_taps[::-1] if rx == "matched" else np.ones(1)
        filtered = fir_filter(sig.data, rx_taps)
        delay = sig.meta["delay"] + (len(rx_taps) - 1) / 2
        viz.plot_eye(filtered / np.max(np.abs(filtered)), sps=SPS, delay=delay, ax=ax)
        ax.set_title(label)
    ctx.save(fig, "eye")

    section("BER with RRC + matched filter vs theory")
    g = build_link(0.35)
    ebn0 = [2.0, 4.0, 6.0, 8.0]
    res = cs.Sweep(g, {"awgn.snr_db": ebn0}, batch_size=64, stop=stop, metrics=["ber.ber", "ber.bit_errors"]).run()
    print(res.table())
    th = theory.ber_bpsk(np.array(ebn0))
    dev = np.max(np.abs(res.column("ber.ber") / th - 1))
    ctx.check("RRC-shaped QPSK BER matches theory", dev < 0.25, f"max relative deviation {dev:.1%}")

    section("Sensitivity to timing offset (no timing recovery), Eb/N0 = 8 dB")
    g = build_link(0.35, ebn0_db=8.0)
    offsets = [0.0, 1.0, 2.0, 3.0, 4.0]  # samples (sps = 8)
    res = cs.Sweep(g, {"tim.delay_samples": offsets}, batch_size=64,
                   stop=cs.StopCriteria.errors("ber.bit_errors", 200, max_batches=100),
                   metrics=["ber.ber"]).run()
    for off, ber in zip(offsets, res.column("ber.ber")):
        print(f"  offset {off / SPS:5.3f} T  ->  BER {ber:.3e}")
    ber = res.column("ber.ber")
    ctx.check("Timing offset degrades BER monotonically", bool(np.all(np.diff(ber) > 0)),
              f"{ber[0]:.1e} -> {ber[-1]:.1e}")


if __name__ == "__main__":
    _common.run_standalone("pulse_shaping", run, __doc__)
