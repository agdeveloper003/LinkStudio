"""Demo: optical communications - IM/DD, chromatic dispersion, coherent DP-QPSK, Kerr nonlinearity.

    python scripts/demo_optical.py [--quick] [--show]
"""

from __future__ import annotations

import _common
from _common import DemoContext, section

import matplotlib.pyplot as plt
import numpy as np

import linkstudio as cs
from linkstudio import viz
from linkstudio.phy import theory
from linkstudio.phy.optical import dbm_to_w


def imdd_ook(power_dbm: float = -22.0) -> cs.Graph:
    g = cs.Graph("imdd-ook")
    g.create("src", "random_bits", n_bits=4000)
    g.create("lc", "line_encoder", symbol_rate=10e9)
    g.create("tx", "intensity_modulator", power_dbm=power_dbm, extinction_ratio_db=10)
    g.create("pd", "photodiode", thermal_noise_pa_rthz=20, shot_noise=False, dark_current_na=0)
    g.create("agc", "agc")
    g.create("ld", "line_decoder")
    g.create("ber", "ber_meter")
    g.chain("src", "lc", "tx", "pd", "agc", "ld", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def imdd_pam4(km: float) -> cs.Graph:
    g = cs.Graph("imdd-pam4")
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="pam", order=4, symbol_rate=28e9)
    g.create("tx", "pulse_shaper", shape="rc", sps=4, rolloff=0.3)
    g.create("mod", "intensity_modulator", power_dbm=3, extinction_ratio_db=8)
    g.create("fib", "fiber", length_km=km)
    g.create("pd", "photodiode")
    g.create("rx", "matched_filter", shape="none")
    g.create("agc", "agc")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "tx", "mod", "fib", "pd", "rx", "agc", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def coherent(launch_dbm: float = -10.0, spans: int = 20, nonlinear: bool = False) -> cs.Graph:
    """32 GBd DP-QPSK over an amplified SSMF link with the full coherent DSP chain."""
    g = cs.Graph("coherent-dp-qpsk")
    g.create("src", "random_bits", n_bits=8000)
    g.create("map", "mapper", scheme="psk", order=4, symbol_rate=32e9)
    g.create("dmx", "stream_demux", n_streams=2)
    g.create("tx", "pulse_shaper", sps=2, rolloff=0.1, span=32)
    g.create("mod", "iq_modulator", power_dbm=launch_dbm, linewidth_hz=100e3)
    g.create("link", "optical_link", n_spans=spans, span_km=80, polarization_rotation=True, nonlinear=nonlinear)
    g.create("rx", "coherent_receiver", frequency_offset_hz=1e9, lo_linewidth_hz=100e3)
    g.create("cdc", "cd_compensation")
    g.create("fo", "fo_compensation")
    g.create("mf", "matched_filter", decimate=False)
    g.create("agc", "agc", remove_dc=False)
    g.create("eq", "adaptive_mimo_equalizer")
    g.create("cpr", "phase_recovery")
    g.create("mux", "stream_mux")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter", skip_bits=4 * 600)
    g.create("evm", "evm_meter", skip_symbols=1200)
    g.chain("src", "map", "dmx", "tx", "mod", "link", "rx", "cdc", "fo", "mf", "agc", "eq", "cpr", "mux",
            "demap", "ber.rx")
    g.connect("dmx", "eq.ref")
    g.connect("dmx", "cpr.ref")
    g.connect("src", "ber.tx")
    g.connect("map", "evm.ref")
    g.connect("mux", "evm.in")
    return g


def run(ctx: DemoContext) -> None:
    _common.banner("Optical communications")

    section("10 Gb/s OOK IM/DD, thermal-noise-limited receiver: BER vs received power")
    powers = np.arange(-26, -19, 1 if not ctx.quick else 2).astype(float)
    stop = cs.StopCriteria.errors("ber.bit_errors", 150 if ctx.quick else 300, max_batches=150)
    res = cs.Sweep(imdd_ook(), {"tx.power_dbm": powers.tolist()}, batch_size=32, stop=stop,
                   metrics=["ber.ber", "ber.bit_errors"]).run()
    print(res.table())
    sigma = 20e-12 * np.sqrt(10e9 / 2)
    th = theory.q_to_ber(np.array([theory.ook_q_factor(dbm_to_w(p), 10, 0.8, sigma) for p in powers]))
    ok = res.column("ber.bit_errors") >= 50
    dev = np.max(np.abs(res.column("ber.ber")[ok] / th[ok] - 1))
    ctx.check("OOK BER matches the Q-factor theory", dev < 0.3, f"max relative deviation {dev:.1%}")
    fine = np.linspace(powers[0], powers[-1], 60)
    th_f = theory.q_to_ber(np.array([theory.ook_q_factor(dbm_to_w(p), 10, 0.8, sigma) for p in fine]))
    fig, ax = plt.subplots(figsize=(7, 5))
    viz.plot_ber({"simulation": (powers, res.column("ber.ber"))}, {"Q-factor theory": (fine, th_f)}, ax=ax,
                 xlabel="Received power [dBm]", title="OOK, ER 10 dB, 20 pA/sqrt(Hz)")
    ctx.save(fig, "ook_sensitivity")

    section("28 GBd PAM-4 IM/DD: chromatic-dispersion penalty vs fiber length")
    for km in (0, 2, 5, 10, 15):
        ber = cs.Simulation(imdd_pam4(km), batch_size=8, seed=1).run(n_batches=2)["ber.ber"]
        print(f"  {km:>3} km  BER {ber:.2e}")
        if km == 2:
            short = ber
        if km == 15:
            long = ber
    ctx.check("Dispersion closes the IM/DD eye at 15 km (C band)", long > 10 * max(short, 1e-5),
              f"2 km {short:.1e}, 15 km {long:.1e}")

    section("32 GBd DP-QPSK, 20 x 80 km, coherent DSP stages")
    g = coherent(launch_dbm=0.0)
    caps = cs.Simulation(g, batch_size=2, seed=3).run(
        n_batches=1, capture=["rx.out", "mf.out", "eq.out", "cpr.out"]).captures
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (key, title) in zip(axes, [("rx.out", "Received (CD, FO, pol. rotation)"), ("mf.out", "After CD + FO comp."),
                                       ("eq.out", "After 2x2 MIMO equalizer"), ("cpr.out", "After phase recovery")]):
        d = caps[key].data[0, 0]
        if key in ("rx.out", "mf.out"):
            d = d[::2] / np.sqrt(np.mean(np.abs(d) ** 2))
        viz.plot_constellation(d, ax=ax, title=title, max_points=2000)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    ctx.save(fig, "coherent_stages")

    section("DP-QPSK BER vs OSNR (launch power sweep, linear fiber)")
    launch = np.arange(-14, -5, 2 if not ctx.quick else 4).astype(float)
    res = cs.Sweep(coherent(), {"mod.power_dbm": launch.tolist()}, batch_size=8, seed=2,
                   stop=cs.StopCriteria.errors("ber.bit_errors", 150 if ctx.quick else 300, max_batches=20),
                   metrics=["ber.ber", "ber.bit_errors", "link.osnr_db"]).run()
    print(res.table())
    osnr = res.column("link.osnr_db")
    th = theory.ber_qam_esn0(4, theory.osnr_to_esn0_db(osnr, 32e9))
    ok = res.column("ber.bit_errors") >= 50
    ratio = res.column("ber.ber")[ok] / th[ok]
    ctx.check("Coherent receiver within ~0.7 dB of OSNR theory", bool(np.all((ratio > 0.8) & (ratio < 4))),
              f"BER / theory = {np.round(ratio, 2).tolist()}")
    fine = np.linspace(osnr.min(), osnr.max(), 60)
    fig, ax = plt.subplots(figsize=(7, 5))
    viz.plot_ber({"DP-QPSK + DSP (sim)": (osnr, res.column("ber.ber"))},
                 {"theory": (fine, theory.ber_qam_esn0(4, theory.osnr_to_esn0_db(fine, 32e9)))}, ax=ax,
                 xlabel="OSNR (0.1 nm) [dB]", title="32 GBd DP-QPSK")
    ctx.save(fig, "coherent_osnr")

    section("Kerr nonlinearity: optimum launch power (10 x 80 km, split-step)")
    powers = [-4.0, -1.0, 2.0, 5.0, 8.0] if not ctx.quick else [-4.0, 2.0, 8.0]
    evm = []
    for p in powers:
        r = cs.Simulation(coherent(launch_dbm=p, spans=10, nonlinear=True), batch_size=4, seed=4).run(n_batches=1)
        evm.append(r["evm.mer_db"])
        print(f"  launch {p:+5.1f} dBm  MER {r['evm.mer_db']:5.2f} dB  OSNR {r['link.osnr_db']:5.1f} dB")
    best = powers[int(np.argmax(evm))]
    ctx.check("MER peaks at an intermediate launch power (ASE vs Kerr)", powers[0] < best < powers[-1],
              f"optimum {best:+.0f} dBm")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(powers, evm, "o-")
    ax.set_xlabel("Launch power [dBm]")
    ax.set_ylabel("MER [dB]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Linear (ASE) regime left, nonlinear (SPM) regime right")
    ctx.save(fig, "launch_power")


if __name__ == "__main__":
    _common.run_standalone("optical", run, __doc__)
