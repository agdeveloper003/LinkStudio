"""Demo: OFDM over AWGN and frequency-selective fading with pilot-based channel estimation.

    python scripts/demo_ofdm.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory

OFDM = dict(n_fft=128, cp_len=16, n_active=96, pilot_spacing=4, subcarrier_spacing=15e3)  # fs = 1.92 MHz


def ofdm_link(channel: str | None = None, estimator: str | None = None, order: int = 4,
              average: bool = True) -> cs.Graph:
    """Mapper -> OFDM -> [multipath] -> AWGN -> FFT -> [estimation] -> equalizer -> demapper."""
    scheme = "psk" if order == 4 else "qam"
    k = int(np.log2(order))
    g = cs.Graph("ofdm")
    g.create("src", "random_bits", n_bits=k * 2000)
    g.create("map", "mapper", scheme=scheme, order=order)
    g.create("ofdm", "ofdm_modulator", **OFDM)
    chain = ["src", "map", "ofdm"]
    if channel:
        g.create("ch", "multipath", preset=channel)
        chain.append("ch")
    g.create("awgn", "awgn", mode="ebn0", snr_db=10.0)
    g.create("demod", "ofdm_demodulator")
    g.create("eq", "ofdm_equalizer")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain(*chain, "awgn", "demod", "eq", "demap", "ber.rx")
    if estimator:
        g.create("est", "ofdm_channel_estimator", method=estimator, average_symbols=average)
        g.connect("demod", "est")
        g.connect("est", "eq.csi")
        if estimator == "perfect":
            g.connect("ch.csi", "est.csi")
    g.connect("src", "ber.tx")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("OFDM")
    stop = cs.StopCriteria.errors("ber.bit_errors", 100 if ctx.quick else 300, max_batches=100)

    section("16-QAM OFDM over AWGN vs theory")
    ebn0 = np.arange(0, 15, 4 if ctx.quick else 2)
    res = cs.Sweep(ofdm_link(order=16), {"awgn.snr_db": ebn0.tolist()}, batch_size=32, stop=stop,
                   metrics=["ber.ber", "ber.bit_errors"]).run()
    print(res.table())
    th = theory.ber_qam(16, ebn0)
    ok = res.column("ber.bit_errors") >= 50
    dev = np.max(np.abs(res.column("ber.ber")[ok] / th[ok] - 1))
    ctx.check("OFDM/AWGN BER matches 16-QAM theory", dev < 0.25, f"max relative deviation {dev:.1%}")

    section("QPSK OFDM over 3GPP EVA multipath: perfect CSI vs LS pilot estimation")
    ebn0 = np.arange(0, 31, 10 if ctx.quick else 5)
    curves = {}
    for label, est, avg in [("Perfect CSI", "perfect", True), ("LS + interpolation, averaged", "ls", True),
                            ("LS + interpolation, per symbol", "ls", False)]:
        r = cs.Sweep(ofdm_link("EVA", est, average=avg), {"awgn.snr_db": ebn0.tolist()}, batch_size=32,
                     stop=stop, metrics=["ber.ber", "ber.bit_errors"]).run()
        curves[label] = (ebn0, r.column("ber.ber"))
        print(f"  {label:<32} " + "  ".join(f"{e:>4.0f}dB:{b:.2e}" for e, b in zip(ebn0, r.column("ber.ber"))))
    fine = np.linspace(0, 30, 60)
    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_ber(curves, {"Rayleigh per subcarrier (theory)": (fine, theory.ber_bpsk_rayleigh(fine)),
                          "AWGN (theory)": (fine, theory.ber_bpsk(fine))}, ax=ax,
                 title="QPSK-OFDM over EVA (block fading)")
    ax.set_ylim(1e-5, 0.5)
    ctx.save(fig, "ber_eva")
    perfect = curves["Perfect CSI"][1]
    ref = theory.ber_bpsk_rayleigh(ebn0)
    good = perfect > 0
    dev = np.max(np.abs(perfect[good] / ref[good] - 1))
    ctx.check("Perfect-CSI OFDM follows per-subcarrier Rayleigh theory", dev < 0.35,
              f"max relative deviation {dev:.1%}")
    ctx.check("LS estimation within a few dB of perfect CSI",
              curves["LS + interpolation, averaged"][1][-2] < 10 * perfect[-2])

    section("Spectrum, channel estimate and equalized constellation (Eb/N0 = 20 dB)")
    g = ofdm_link("EVA", "ls", order=16)
    g.set_param("awgn.snr_db", 20.0)
    caps = cs.Simulation(g, batch_size=8, seed=11).run(
        n_batches=1, capture=["ofdm.out", "ch.csi", "est.out", "eq.out"]).captures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    viz.plot_psd(caps["ofdm.out"], ax=axes[0], nperseg=256)
    axes[0].set_title("OFDM transmit spectrum")
    lay = caps["est.out"].meta["ofdm"]
    true_h = np.fft.fftshift(np.fft.fft(caps["ch.csi"].data[0], n=lay.n_fft))
    est_h = np.fft.fftshift(caps["est.out"].data[0, 0])
    f = np.arange(-lay.n_fft // 2, lay.n_fft // 2)
    active = np.isin(f, lay.active_freqs)
    axes[1].plot(f, 20 * np.log10(np.abs(true_h)), label="true |H(f)|")
    axes[1].plot(f[active], 20 * np.log10(np.abs(est_h[active])), ".", label="LS estimate")
    axes[1].set_xlabel("Subcarrier")
    axes[1].set_ylabel("dB")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Channel frequency response (frame 0)")
    viz.plot_constellation(caps["eq.out"], ax=axes[2], title="Equalized 16-QAM")
    axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2)
    ctx.save(fig, "ofdm_overview")


if __name__ == "__main__":
    _common.run_standalone("ofdm", run, __doc__)
