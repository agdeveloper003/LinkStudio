"""Generate the data, code excerpts and reference tables used by the LinkStudio manual.

Every figure of the manual that shows a measured quantity is drawn from a data file
written here by running the real simulator blocks, so the plots are reproducible
results of the code, not illustrations. Code excerpts are extracted from the source
with ``inspect`` so that file names and line numbers always match the code base.

Usage (from the repository root)::

    python manual/tools/generate.py              # everything
    python manual/tools/generate.py --only ber_awgn coding
    python manual/tools/generate.py --list

Outputs go to ``manual/generated/``: ``data/*.dat`` (whitespace separated, one header
row), ``snippets/*.py`` plus ``snippets.tex`` (excerpt metadata) and the tables
``blocks.tex`` and ``scenarios.tex``.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
OUT = ROOT / "manual" / "generated"
DATA = OUT / "data"
SNIP = OUT / "snippets"
sys.path.insert(0, str(BACKEND))

import linkstudio as cs  # noqa: E402
from linkstudio.phy import theory  # noqa: E402

FIGURES: dict[str, callable] = {}


def figure(fn):
    FIGURES[fn.__name__.removeprefix("fig_")] = fn
    return fn


def save(name: str, columns: dict[str, np.ndarray]) -> None:
    """Write equal-length columns as a pgfplots table (NaN for missing points)."""
    keys = list(columns)
    arr = np.column_stack([np.asarray(columns[k], dtype=float) for k in keys])
    np.savetxt(DATA / f"{name}.dat", arr, header=" ".join(keys), comments="", fmt="%.6g")


def save_traces(name: str, traces: np.ndarray, x: np.ndarray) -> None:
    """Several curves in one file separated by empty lines (pgfplots jumps between them)."""
    with open(DATA / f"{name}.dat", "w") as fh:
        fh.write("x y\n")
        for tr in traces:
            for xi, yi in zip(x, tr):
                fh.write(f"{xi:.5g} {yi:.5g}\n")
            fh.write("\n")


def sweep_ber(graph: cs.Graph, param: str, values, metric: str = "ber", errors: int = 200,
              max_batches: int = 25, batch_size: int = 32, min_errors: int = 10, meter: str = "ber"):
    stop = cs.StopCriteria.errors(f"{meter}.bit_errors", errors, max_batches=max_batches)
    res = cs.Sweep(graph, {param: list(values)}, batch_size=batch_size, seed=11, stop=stop).run()
    ber = res.column(f"{meter}.{metric}").astype(float)
    ber[res.column(f"{meter}.bit_errors") < min_errors] = np.nan
    return ber


def capture(graph: cs.Graph, keys, batch: int = 4, seed: int = 0):
    return cs.Simulation(graph, batch_size=batch, seed=seed).run(n_batches=1, capture=keys).captures


# =============================================================== modulation
def uncoded(scheme: str, order: int, n_bits: int = 6000) -> cs.Graph:
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme=scheme, order=order)
    g.create("awgn", "awgn", snr_db=0.0, mode="ebn0")
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


@figure
def fig_ber_awgn():
    ebn0 = np.arange(0, 15.0, 1.0)
    fine = np.linspace(0, 16, 81)
    cols = {"ebn0": ebn0}
    for label, scheme, order in [("bpsk", "psk", 2), ("psk8", "psk", 8), ("qam16", "qam", 16), ("qam64", "qam", 64)]:
        cols[label] = sweep_ber(uncoded(scheme, order), "awgn.snr_db", ebn0)
    save("ber_awgn_sim", cols)
    save("ber_awgn_theory", {"ebn0": fine, "bpsk": theory.ber_bpsk(fine), "psk8": theory.ber_psk(8, fine),
                             "qam16": theory.ber_qam(16, fine), "qam64": theory.ber_qam(64, fine),
                             "shannon_limit": np.full_like(fine, theory.ebn0_min_db(1e-9))})


@figure
def fig_constellations():
    def run(extra):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=4000)
        g.create("map", "mapper", scheme="qam", order=16)
        chain = ["src", "map"]
        for i, (t, p) in enumerate(extra):
            g.create(f"imp{i}", t, **p)
            chain.append(f"imp{i}")
        g.create("awgn", "awgn", snr_db=26.0, mode="esn0")
        g.chain(*chain, "awgn")
        return capture(g, ["awgn.out"], batch=1)["awgn.out"].data[0, :800]
    cases = {
        "awgn": [],
        "pn": [("phase_noise", {"linewidth_hz": 150.0})],
        "iq": [("iq_imbalance", {"amplitude_db": 1.5, "phase_deg": 8.0})],
        "pa": [("power_amplifier", {"model": "saleh", "input_backoff_db": 4.0})],
    }
    for name, extra in cases.items():
        z = run(extra)
        save(f"const_{name}", {"i": z.real, "q": z.imag})


@figure
def fig_pulses():
    from linkstudio.phy.dsp.filters import rc_taps, rrc_taps
    sps, span = 16, 8
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    rc = rc_taps(sps, span, 0.25)
    rrc = rrc_taps(sps, span, 0.25)
    save("pulses_time", {"t": t, "rc": rc / rc.max(), "rrc": rrc / rrc.max()})
    nfft = 8192
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sps))
    cols = {"f": f}
    for b in (0.0, 0.25, 0.5, 1.0):
        h = rc_taps(sps, 32, b)
        H = np.abs(np.fft.fftshift(np.fft.fft(h, nfft)))
        cols[f"b{int(b * 100)}"] = H / H[nfft // 2]
    keep = np.abs(f) <= 1.2
    save("pulses_freq", {k: v[keep] for k, v in cols.items()})


@figure
def fig_eye():
    from linkstudio.phy.dsp.analysis import eye_traces
    sps = 16
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=400)
    g.create("map", "mapper", scheme="pam", order=2)
    g.create("tx", "pulse_shaper", sps=sps, rolloff=0.35, span=10)
    g.create("awgn", "awgn", snr_db=30.0, mode="esn0")
    g.create("mf", "matched_filter", decimate=False)
    g.chain("src", "map", "tx", "awgn", "mf")
    caps = capture(g, ["tx.out", "mf.out"], batch=1)
    x = np.arange(2 * sps + 1) / sps - 1.0
    traces = {}
    for key in ("tx.out", "mf.out"):
        sig = caps[key]
        start = int(round(sig.meta["delay"])) - sps + 20 * sps  # traces centred on symbol instants
        traces[key] = eye_traces(sig.data.real, sps, offset=start, symbols_per_trace=2, max_traces=90)
    # The matched filter output is at nominal amplitude (+-1); give the TX waveform the same RMS.
    tx = traces["tx.out"] * np.sqrt(np.mean(traces["mf.out"] ** 2) / np.mean(traces["tx.out"] ** 2))
    save_traces("eye_tx", tx, x)
    save_traces("eye_mf", traces["mf.out"], x)


# ============================================================== impairments
@figure
def fig_pa_curves():
    r = np.linspace(1e-3, 2.0, 200)
    p_in = float(np.mean(r ** 2))
    bo = -10 * np.log10(p_in)  # makes the block's input scaling exactly 1
    sig = cs.Signal(r[None, :].astype(np.complex128), cs.Kind.SYMBOLS)
    cols = {"r": r}
    from linkstudio.core.registry import create_block
    for name, params in [("rapp1", dict(model="rapp", smoothness=1)), ("rapp2", dict(model="rapp", smoothness=2)),
                         ("rapp5", dict(model="rapp", smoothness=5)), ("clip", dict(model="clip")),
                         ("saleh", dict(model="saleh"))]:
        blk = create_block("power_amplifier", input_backoff_db=bo, normalize_output=False, **params)
        y = blk.process({"in": sig}, None)["out"].data[0]
        cols[name] = np.abs(y)
        if name == "saleh":
            cols["saleh_pm_deg"] = np.rad2deg(np.angle(y))
    save("pa_curves", cols)


# =============================================================== fading
@figure
def fig_jakes():
    from scipy.special import j0
    from linkstudio.phy.channel.fading import sum_of_sinusoids
    rng = np.random.default_rng(5)
    fd = 0.01
    h = sum_of_sinusoids(rng, 400, 3000, fd, 16)
    lags = np.arange(0, 301, 3)
    ac = [np.mean(h[:, k:] * np.conj(h[:, :h.shape[1] - k])).real for k in lags]
    save("jakes_acf", {"lag": lags * fd, "sim": ac, "theory": j0(2 * np.pi * fd * lags)})
    one = sum_of_sinusoids(np.random.default_rng(9), 1, 2000, 0.002, 16)[0]
    save("jakes_envelope", {"t": np.arange(2000) * 0.002, "env_db": 20 * np.log10(np.abs(one))})


def siso_rayleigh() -> cs.Graph:
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("ch", "flat_fading", variation="fast")
    g.create("awgn", "awgn", snr_db=0.0)
    g.create("eq", "one_tap_equalizer")
    g.create("demap", "demapper", method="maxlog")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "ch", "awgn", "eq", "demap", "ber.rx")
    g.connect("ch.csi", "eq.csi")
    g.connect("src", "ber.tx")
    return g


def mimo(kind: str, nr: int = 2, method: str = "mmse", order: int = 2, variation: str = "fast",
         n_bits: int = 4000) -> cs.Graph:
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme="psk", order=order)
    chain = ["src", "map"]
    if kind == "smux":
        g.create("dmx", "stream_demux", n_streams=2)
        chain.append("dmx")
    elif kind == "alamouti":
        g.create("stbc", "alamouti_encoder")
        chain.append("stbc")
    g.create("ch", "mimo_channel", n_rx=nr, variation=variation)
    g.create("awgn", "awgn", snr_db=0.0)
    chain += ["ch", "awgn"]
    if kind == "alamouti":
        g.create("det", "alamouti_decoder")
        chain.append("det")
    else:
        g.create("det", "mimo_detector", method=method)
        g.create("mux", "stream_mux")
        chain += ["det", "mux"]
    g.connect("ch.csi", "det.csi")
    g.create("demap", "demapper", method="hard" if method == "ml" else "maxlog")
    g.create("ber", "ber_meter")
    g.chain(*chain, "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


@figure
def fig_diversity():
    ebn0 = np.arange(0, 22.0, 2.0)
    fine = np.linspace(0, 22, 89)
    siso = sweep_ber(siso_rayleigh(), "awgn.snr_db", ebn0)
    simo = sweep_ber(mimo("simo"), "awgn.snr_db", ebn0)
    # Block fading: errors are correlated within a frame, so many short frames and many errors
    # are needed for a stable estimate.
    ala = sweep_ber(mimo("alamouti", nr=1, variation="block", n_bits=100), "awgn.snr_db", ebn0,
                    errors=3000, batch_size=512, max_batches=600)
    save("diversity_sim", {"ebn0": ebn0, "siso": siso, "mrc2": simo, "alamouti": ala})
    save("diversity_theory", {"ebn0": fine, "awgn": theory.ber_bpsk(fine), "siso": theory.ber_bpsk_rayleigh(fine),
                              "mrc2": theory.ber_bpsk_mrc(2, fine),
                              "alamouti": theory.ber_bpsk_mrc(2, fine - 10 * np.log10(2))})


@figure
def fig_mimo_smux():
    ebn0 = np.arange(0, 22.0, 2.0)
    cols = {"ebn0": ebn0}
    for m in ("zf", "mmse", "ml"):
        cols[m] = sweep_ber(mimo("smux", method=m, order=4), "awgn.snr_db", ebn0)
    save("mimo_smux", cols)


# ================================================================= coding
@figure
def fig_coding():
    ebn0 = np.arange(0, 8.01, 0.5)

    def link(enc, dec, n_bits, soft=True):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=n_bits)
        g.create("enc", enc[0], **enc[1])
        g.create("map", "mapper", scheme="psk", order=4)
        g.create("awgn", "awgn", snr_db=0.0)
        g.create("demap", "demapper", method="maxlog" if soft else "hard")
        g.create("dec", dec[0], **dec[1])
        g.create("ber", "ber_meter")
        g.chain("src", "enc", "map", "awgn", "demap", "dec", "ber.rx")
        g.connect("src", "ber.tx")
        return g

    cols = {"ebn0": ebn0}
    cols["hamming"] = sweep_ber(link(("hamming_encoder", {}), ("hamming_decoder", {}), 4000, soft=False),
                                "awgn.snr_db", ebn0, batch_size=16, max_batches=20)
    cols["conv"] = sweep_ber(link(("conv_encoder", {}), ("viterbi_decoder", {}), 2000),
                             "awgn.snr_db", ebn0, batch_size=16, max_batches=20)
    cols["ldpc"] = sweep_ber(link(("ldpc_encoder", {}), ("ldpc_decoder", {}), 3840),
                             "awgn.snr_db", ebn0, batch_size=16, max_batches=20)
    cols["polar"] = sweep_ber(link(("polar_encoder", {}), ("polar_decoder", {}), 1280),
                              "awgn.snr_db", ebn0, batch_size=16, max_batches=20)
    save("coding_sim", cols)
    fine = np.linspace(0, 10, 81)
    save("coding_theory", {"ebn0": fine, "uncoded": theory.ber_bpsk(fine),
                           "hamming": theory.ber_hamming_hard(3, fine)})


@figure
def fig_ldpc_base():
    from linkstudio.phy.coding.ldpc import QCLDPCCode
    code = QCLDPCCode(8, 8, 48, 3, 1)
    z = code.z
    H = code.H.tocoo()
    mb, nb = code.m // z, code.n // z
    shift = -np.ones((mb, nb), dtype=int)
    for r, c in zip(H.row, H.col):
        if r % z == 0:
            shift[r // z, c // z] = (c - r) % z
    with open(OUT / "ldpc_base.tex", "w") as fh:
        fh.write("% Base matrix of the default QC-LDPC code (k_blocks=8, m_blocks=8, z=48), from QCLDPCCode.H\n")
        for i in range(mb):
            for j in range(nb):
                s = shift[i, j]
                if s >= 0:
                    style = "infoblk" if j < nb - mb else "parblk"
                    fh.write(f"\\node[{style}] at ({j * 0.62:.2f},{-i * 0.62:.2f}) {{\\tiny {s}}};\n")
                else:
                    fh.write(f"\\node[zeroblk] at ({j * 0.62:.2f},{-i * 0.62:.2f}) {{}};\n")


@figure
def fig_polar_reliability():
    from linkstudio.phy.coding.polar import frozen_mask, reliability
    n_log2, k = 6, 32
    z = reliability(n_log2, 0.0)
    frozen = frozen_mask(n_log2, k, 0.0)
    idx = np.arange(z.size)
    save("polar_info", {"i": idx[~frozen], "z": z[~frozen]})
    save("polar_frozen", {"i": idx[frozen], "z": z[frozen]})


# ============================================================ synchronization
@figure
def fig_gardner_scurve():
    # Same cubic Lagrange interpolator as sync.cubic_interpolate, vectorized over many instants.
    from linkstudio.phy.dsp.filters import cubic_resample as cubic_interpolate
    cols = {}
    taus = np.linspace(-0.5, 0.5, 41)
    cols["tau"] = taus
    for beta in (0.2, 0.35, 0.6):
        sps = 8
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=8000)
        g.create("map", "mapper", scheme="psk", order=4)
        g.create("tx", "pulse_shaper", sps=sps, rolloff=beta, span=12)
        g.create("mf", "matched_filter", decimate=False)
        g.chain("src", "map", "tx", "mf")
        sig = capture(g, ["mf.out"], batch=4)["mf.out"]
        x, start = sig.data, sig.meta["delay"]
        m = np.arange(2, 3900)
        s = []
        for tau in taus:
            t = np.broadcast_to(start + (m + tau) * sps, (x.shape[0], m.size))
            y, mid, prev = (cubic_interpolate(x, t - d) for d in (0.0, sps / 2, sps))
            s.append(np.mean(np.real(np.conj(mid) * (y - prev))))
        cols[f"b{int(beta * 100)}"] = s
    save("gardner_scurve", cols)


@figure
def fig_sync_constellations():
    sc = cs.load_scenario(BACKEND / "scenarios" / "wireless" / "synchronized_receiver.yaml")
    caps = cs.Simulation(sc.graph, batch_size=1, seed=4).run(n_batches=1, capture=["mf.out", "tr.out", "fs.out",
                                                                                   "cr.out"]).captures
    mf = caps["mf.out"]
    raw = mf.data[0, int(mf.meta["delay"])::mf.meta["sps"]][100:700]
    save("sync_raw", {"i": raw.real, "q": raw.imag})
    for key in ("tr", "fs", "cr"):
        z = caps[f"{key}.out"].data[0, 100:700]
        save(f"sync_{key}", {"i": z.real, "q": z.imag})


# ================================================================== OFDM
@figure
def fig_ofdm_chest():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("ofdm", "ofdm_modulator", n_fft=128, cp_len=16, n_active=100, pilot_spacing=6)
    g.create("ch", "multipath", preset="EVA")
    g.create("awgn", "awgn", snr_db=15.0, mode="esn0")
    g.create("demod", "ofdm_demodulator")
    g.create("ls", "ofdm_channel_estimator", method="ls")
    g.create("ideal", "ofdm_channel_estimator", method="perfect")
    g.chain("src", "map", "ofdm", "ch", "awgn", "demod", "ls")
    g.connect("demod", "ideal.in")
    g.connect("ch.csi", "ideal.csi")
    caps = capture(g, ["ls.out", "ideal.out", "ofdm.out"], batch=1, seed=3)
    lay = caps["ofdm.out"].meta["ofdm"]
    act = lay.active_freqs
    H_ls = caps["ls.out"].data[0, 0, lay.bins(act)]
    H_true = caps["ideal.out"].data[0, 0, lay.bins(act)]
    save("ofdm_chest", {"k": act, "true_db": 20 * np.log10(np.abs(H_true)), "ls_db": 20 * np.log10(np.abs(H_ls)),
                        "true_ph": np.unwrap(np.angle(H_true)), "ls_ph": np.unwrap(np.angle(H_ls))})
    pil = lay.pilot_freqs
    Hp = caps["ls.out"].data[0, 0, lay.bins(pil)]
    save("ofdm_pilots", {"k": pil, "db": 20 * np.log10(np.abs(Hp))})


@figure
def fig_papr():
    cols = {}
    grid = np.linspace(4, 13, 46)
    cols["papr"] = grid
    for n_fft, n_act in ((64, 52), (256, 200), (1024, 800)):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=2 * n_act * 40)
        g.create("map", "mapper", scheme="psk", order=4)
        g.create("ofdm", "ofdm_modulator", n_fft=n_fft, cp_len=0, n_active=n_act, pilot_spacing=0)
        g.chain("src", "map", "ofdm")
        x = capture(g, ["ofdm.out"], batch=64)["ofdm.out"].data.reshape(-1, n_fft)
        p = np.abs(x) ** 2
        papr = 10 * np.log10(p.max(axis=1) / p.mean(axis=1))
        cols[f"sim{n_fft}"] = [np.mean(papr > g_) for g_ in grid]
        cols[f"th{n_fft}"] = 1 - (1 - np.exp(-10 ** (grid / 10))) ** n_act
    save("papr_ccdf", cols)


# ================================================================ wireline
@figure
def fig_cable():
    from linkstudio.phy.wireline import cable_response
    f = np.logspace(4, 7.5, 200)
    cols = {"f_mhz": f / 1e6}
    for km in (1, 2, 3):
        cols[f"dsl{km}km"] = -20 * np.log10(np.abs(cable_response(f, "dsl_0.4mm", km * 1000)))
    cols["cat5e100m"] = -20 * np.log10(np.abs(cable_response(f, "cat5e", 100)))
    save("cable_loss", cols)
    fs, n = 17.664e6, 4096
    fg = np.fft.fftfreq(n, 1 / fs)
    H = cable_response(fg, "dsl_0.4mm", 2000)
    h_min = np.fft.ifft(H).real
    h_zero = np.fft.fftshift(np.fft.ifft(np.abs(H)).real)
    t = np.arange(n) / fs * 1e6
    keep = t < 40
    tz = (np.arange(n) - n // 2) / fs * 1e6
    keepz = np.abs(tz) < 20
    save("cable_impulse_min", {"t_us": t[keep], "h": h_min[keep] / h_min.max()})
    save("cable_impulse_zero", {"t_us": tz[keepz] + 20, "h": h_zero[keepz] / h_zero.max()})


@figure
def fig_ctle():
    from linkstudio.core.registry import create_block
    from linkstudio.phy.wireline import cable_response
    f = np.linspace(1e7, 2e10, 300)
    # Parameters chosen so that the combined response is flat within ~2 dB up to 14 GHz
    # (Nyquist frequency of a 28 GBd link).
    ch = cable_response(f, "backplane", 0.8)
    ctle = create_block("ctle", zero_hz=3e9, pole1_hz=2e10, pole2_hz=3e10).response(f)
    save("ctle", {"f_ghz": f / 1e9, "channel_db": 20 * np.log10(np.abs(ch)),
                  "ctle_db": 20 * np.log10(np.abs(ctle)), "total_db": 20 * np.log10(np.abs(ch * ctle))})


@figure
def fig_dmt_bits():
    from linkstudio.core.registry import create_block
    cols = {}
    rates = []
    for km in (1, 2, 3):
        m = create_block("dmt_modulator", line_length_m=km * 1000, snr_db=50.0)
        cols.setdefault("tone", m.tones)
        cols[f"bits{km}km"] = m.bits
        rates.append(m.metrics()["data_rate_mbps"])
    save("dmt_bits", cols)
    (OUT / "dmt_rates.tex").write_text(
        "".join(f"\\def\\dmtrate{n}{{{r:.1f}}}\n" for n, r in zip(("A", "B", "C"), rates)))


@figure
def fig_linecode_psd():
    from linkstudio.phy.dsp.analysis import psd
    sps = 16
    cols = {}
    for code in ("nrz", "manchester", "ami", "mlt3"):
        chips = 2 if code == "manchester" else 1
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=8192)
        g.create("lc", "line_encoder", code=code, symbol_rate=float(chips))
        g.create("ps", "pulse_shaper", shape="rect", sps=sps)
        g.chain("src", "lc", "ps")
        x = capture(g, ["ps.out"], batch=16)["ps.out"].data.real
        f, p = psd(x, fs=sps * chips, nperseg=1024)
        sel = (f >= 0) & (f <= 3)
        fi = np.linspace(0, 3, 151)
        cols.setdefault("f", fi)
        cols[code] = 10 * np.log10(np.maximum(np.interp(fi, f[sel], p[sel]), 1e-6))
    ref = max(np.max(v) for k, v in cols.items() if k != "f")
    save("linecode_psd", {k: (v if k == "f" else np.maximum(v - ref, -40)) for k, v in cols.items()})


# ================================================================= optical
@figure
def fig_fiber_dispersion():
    from linkstudio.phy.optical import beta2, propagate
    T0, fs, n = 5e-12, 1e12, 2048
    t = (np.arange(n) - n / 2) / fs
    e0 = np.exp(-t ** 2 / (2 * T0 ** 2)).astype(np.complex128)[None, :]
    b2 = abs(beta2(17.0, 1550.0))
    LD = T0 ** 2 / b2 / 1e3  # km
    cols = {"t_ps": t * 1e12}
    for k in (0, 1, 2, 4):
        e = propagate(e0, fs, k * LD, 0.0, 17.0, 1550.0, False, 0.0, 1.0)
        cols[f"z{k}"] = np.abs(e[0]) ** 2
    keep = np.abs(t) < 60e-12
    save("fiber_pulses", {k: v[keep] for k, v in cols.items()})

    def rms(p):
        p = p / p.sum()
        mu = np.sum(t * p)
        return np.sqrt(np.sum((t - mu) ** 2 * p))
    zs = np.linspace(0, 5, 11)
    w0 = rms(np.abs(e0[0]) ** 2)
    ratio = [rms(np.abs(propagate(e0, fs, z * LD, 0.0, 17.0, 1550.0, False, 0.0, 1.0)[0]) ** 2) / w0 for z in zs]
    save("fiber_broadening", {"z": zs, "sim": ratio, "theory": np.sqrt(1 + zs ** 2)})
    (OUT / "fiber_ld.tex").write_text(f"\\def\\fiberLD{{{LD:.3f}}}\n\\def\\fiberbeta{{{b2 * 1e27:.2f}}}\n")


@figure
def fig_spm():
    from linkstudio.phy.optical import propagate
    T0, fs, n = 5e-12, 2e12, 4096
    t = (np.arange(n) - n / 2) / fs
    e0 = np.exp(-t ** 2 / (2 * T0 ** 2)).astype(np.complex128)
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs)) * T0  # normalized (nu - nu0) T0
    cols = {"f": f}
    for label, phi in (("p0", 0.0), ("p15", 1.5 * np.pi), ("p35", 3.5 * np.pi)):
        p0 = phi / 1.3  # phi_max = gamma P0 L with L = 1 km, gamma = 1.3 /W/km
        e = propagate((np.sqrt(p0) * e0)[None, :] if p0 else e0[None, :], fs, 1.0, 0.0, 0.0, 1550.0,
                      True, 1.3, 50.0)
        S = np.abs(np.fft.fftshift(np.fft.fft(e[0]))) ** 2
        cols[label] = S / S.max()
    keep = np.abs(f) < 1.6
    save("spm_spectra", {k: v[keep] for k, v in cols.items()})


@figure
def fig_soliton():
    from linkstudio.phy.optical import beta2, propagate
    T0, fs, n = 5e-12, 4e12, 2048
    gamma = 1.3
    t = (np.arange(n) - n / 2) / fs
    b2 = abs(beta2(17.0, 1550.0))
    P0 = b2 / (gamma * 1e-3 * T0 ** 2)
    LD = T0 ** 2 / b2 / 1e3
    L = 4 * LD
    e0 = (np.sqrt(P0) / np.cosh(t / T0)).astype(np.complex128)[None, :]
    rows = {"h": [], "err": [], "steps_per_km": []}
    for spk in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128):
        e = propagate(e0, fs, L, 0.0, 17.0, 1550.0, True, gamma, spk)[0]
        ref = e0[0]
        th = np.angle(np.vdot(ref, e))  # remove the (convention dependent) soliton phase
        err = np.linalg.norm(e - ref * np.exp(1j * th)) / np.linalg.norm(ref)
        steps = max(1, int(np.ceil(L * spk)))
        rows["h"].append(L / steps)
        rows["err"].append(err)
        rows["steps_per_km"].append(spk)
    save("soliton_convergence", rows)
    e = propagate(e0, fs, L, 0.0, 17.0, 1550.0, False, gamma, 1.0)[0]
    en = propagate(e0, fs, L, 0.0, 17.0, 1550.0, True, gamma, 64.0)[0]
    keep = np.abs(t) < 40e-12
    save("soliton_profiles", {"t_ps": t[keep] * 1e12, "input": np.abs(e0[0][keep]) ** 2 / P0,
                              "linear": np.abs(e[keep]) ** 2 / P0, "nlse": np.abs(en[keep]) ** 2 / P0})
    (OUT / "soliton.tex").write_text(f"\\def\\solitonPzero{{{P0:.3f}}}\n\\def\\solitonL{{{L:.2f}}}\n")


@figure
def fig_osnr_spans():
    spans = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40])
    sim = []
    for nsp in spans:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=512)
        g.create("map", "mapper", scheme="psk", order=4, symbol_rate=32e9)
        g.create("tx", "pulse_shaper", sps=2)
        g.create("mod", "iq_modulator", power_dbm=0.0, linewidth_hz=0.0)
        g.create("link", "optical_link", n_spans=int(nsp), span_km=80.0, noise_figure_db=5.0)
        g.chain("src", "map", "tx", "mod", "link")
        sim.append(cs.Simulation(g, batch_size=1).run(n_batches=1)["link.osnr_db"])
    th = 58.0 + 0.0 - 16.0 - 5.0 - 10 * np.log10(spans)
    save("osnr_spans", {"spans": spans, "sim": sim, "rule58": th})


@figure
def fig_ook():
    from linkstudio.phy.optical import dbm_to_w
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=4000)
    g.create("lc", "line_encoder", symbol_rate=10e9)
    g.create("tx", "intensity_modulator", power_dbm=-22.0, extinction_ratio_db=10.0)
    g.create("pd", "photodiode", thermal_noise_pa_rthz=20, shot_noise=False, dark_current_na=0)
    g.create("agc", "agc")
    g.create("ld", "line_decoder")
    g.create("ber", "ber_meter")
    g.chain("src", "lc", "tx", "pd", "agc", "ld", "ber.rx")
    g.connect("src", "ber.tx")
    p = np.arange(-27.0, -18.9, 1.0)
    sim = sweep_ber(g, "tx.power_dbm", p, max_batches=30)
    fine = np.linspace(-28, -18, 81)
    sigma = 20e-12 * np.sqrt(10e9 / 2)
    th = [float(theory.q_to_ber(theory.ook_q_factor(dbm_to_w(x), 10.0, 0.8, sigma))) for x in fine]
    save("ook_sim", {"p": p, "ber": sim})
    save("ook_theory", {"p": fine, "ber": th})


@figure
def fig_coherent_dsp():
    sc = cs.load_scenario(BACKEND / "scenarios" / "optical" / "coherent_dp_qpsk.yaml")
    keys = ["rx.out", "fo.out", "eq.out", "cpr.out"]
    caps = cs.Simulation(sc.graph, batch_size=1, seed=2).run(n_batches=1, capture=keys).captures
    # 2000 symbols per polarization at 2 samples/symbol; skip the equalizer convergence.
    rx = caps["rx.out"].data[0, 0, 1600:3200:2]
    fo = caps["fo.out"].data[0, 0, 1600:3200:2]
    eq = caps["eq.out"].data[0, 0, 1200:2000]
    cpr = caps["cpr.out"].data[0, 0, 1200:2000]
    for name, z in (("rx", rx), ("fo", fo), ("eq", eq), ("cpr", cpr)):
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        save(f"coh_{name}", {"i": z.real, "q": z.imag})


@figure
def fig_fso():
    from scipy.special import gamma as G, kv
    from linkstudio.core.registry import create_block
    x = cs.Signal(np.ones((60000, 1), dtype=np.complex128), cs.Kind.SAMPLES)
    hgrid = np.linspace(0.02, 3.0, 60)
    cols = {"h": hgrid}
    for name, params in (("ln", dict(model="lognormal", log_intensity_variance=0.1)),
                         ("gg", dict(model="gamma_gamma", alpha=4.0, beta=2.0))):
        blk = create_block("fso_turbulence", **params)
        blk.reset(1, "fso")
        h = np.abs(blk.process({"in": x}, None)["out"].data[:, 0]) ** 2
        hist, edges = np.histogram(h, bins=60, range=(0, 3.02), density=True)
        cols[f"{name}_hist"] = np.interp(hgrid, 0.5 * (edges[1:] + edges[:-1]), hist)
    s2 = 0.1
    cols["ln_pdf"] = np.exp(-(np.log(hgrid) + s2 / 2) ** 2 / (2 * s2)) / (hgrid * np.sqrt(2 * np.pi * s2))
    a, b = 4.0, 2.0
    cols["gg_pdf"] = (2 * (a * b) ** ((a + b) / 2) / (G(a) * G(b)) * hgrid ** ((a + b) / 2 - 1)
                      * kv(a - b, 2 * np.sqrt(a * b * hgrid)))
    save("fso_pdf", cols)


# ================================================================ statistics
@figure
def fig_wilson():
    from linkstudio.phy.metrics import wilson_interval
    k = np.unique(np.round(np.logspace(0, 3.3, 40)).astype(int))
    p = 1e-4
    lo, hi = zip(*(wilson_interval(int(e), int(e / p)) for e in k))
    save("wilson", {"errors": k, "lo": np.array(lo) / p, "hi": np.array(hi) / p})


# ================================================================== excerpts
SNIPPETS: list[tuple[str, str, str | None, str | None]] = [
    # (name, dotted object, first line regex, last line regex)
    ("block_reset", "linkstudio.core.block.Block.reset", None, None),
    ("graph_toposort", "linkstudio.core.graph.Graph._toposort", None, None),
    ("graph_execute", "linkstudio.core.graph.Graph.execute", r"for name in self.order", r"values\[f\"\{name\}"),
    ("stop_check", "linkstudio.core.simulation.StopCriteria.check", None, None),
    ("sweep_point", "linkstudio.core.sweep.Sweep.run", r"for i, point in enumerate", r"res = sim.run"),
    ("awgn_noise_var", "linkstudio.phy.channel.awgn.AWGN.noise_variance", None, None),
    ("awgn_process", "linkstudio.phy.channel.awgn.AWGN.process", None, None),
    ("wilson", "linkstudio.phy.metrics.wilson_interval", None, None),
    ("demap_soft", "linkstudio.phy.dsp.constellations.Constellation.demap_soft", r"k = self.bits_per_symbol", None),
    ("qam_build", "linkstudio.phy.dsp.constellations.qam", r"side = int", None),
    ("rrc_taps", "linkstudio.phy.dsp.filters.rrc_taps", None, None),
    ("mf_normalize", "linkstudio.phy.waveform.pulse_shaping.MatchedFilter.process", r"y = fir_filter", r"noise = \{"),
    ("pulse_shaper", "linkstudio.phy.waveform.pulse_shaping.PulseShaper.process", None, None),
    ("phase_noise", "linkstudio.phy.channel.impairments.PhaseNoise.process", None, None),
    ("iq_imbalance", "linkstudio.phy.channel.impairments.IQImbalance.process", None, None),
    ("power_amplifier", "linkstudio.phy.channel.impairments.PowerAmplifier.process", r"p_in = ", r"y = np.minimum"),
    ("quantizer", "linkstudio.phy.channel.impairments.Quantizer.process", r"levels = ", None),
    ("sos", "linkstudio.phy.channel.fading.sum_of_sinusoids", None, None),
    ("flat_fading", "linkstudio.phy.channel.fading.FlatFading.process", r"if p.variation", r"h = np.sqrt\(k"),
    ("linear_eq", "linkstudio.phy.receiver.LinearEqualizer.process", r"H = np.zeros", r"new_nv = "),
    ("lms_loop", "linkstudio.phy.receiver.LMSEqualizer.process", r"for t in range", r"w \+= mu"),
    ("loop_gains", "linkstudio.phy.sync.loop_gains", None, None),
    ("cubic", "linkstudio.phy.sync.cubic_interpolate", r"n = x.shape", None),
    ("om_estimate", "linkstudio.phy.sync.TimingRecovery.process", r"if p.initial_estimate", r"tau = np.mod"),
    ("gardner_loop", "linkstudio.phy.sync.TimingRecovery.process", r"for m in range\(1", r"prev = y"),
    ("frame_sync", "linkstudio.phy.sync.FrameSync.process", r"if self.params.correct_cfo:", r"out = data\[rows"),
    ("dd_pll", "linkstudio.phy.sync.CarrierRecovery.process", r"for n in range", r"out\[:, n\]"),
    ("ofdm_mod", "linkstudio.phy.waveform.ofdm.OFDMModulator.process", r"n_sym = ", r"frame = with_cp"),
    ("ofdm_sync", "linkstudio.phy.waveform.ofdm.OFDMSync.process", r"prod = ", r"d = np.maximum"),
    ("ls_slope", "linkstudio.phy.waveform.ofdm.OFDMChannelEstimator.process", r"hp = grid.data", r"H\[\.\.\., lay.bins"),
    ("mimo_linear", "linkstudio.phy.mimo.MIMODetector.process", r"gram = ", r"nv = \(1 - bias"),
    ("alamouti_dec", "linkstudio.phy.mimo.AlamoutiDecoder.process", r"r1, r2 = ", r"s2 = np.sum"),
    ("crc_matrix", "linkstudio.phy.coding.crc.crc_parity_matrix", r"r = poly", None),
    ("lfsr", "linkstudio.phy.dsp.lfsr.lfsr_sequence", r"\"\"\"$", None),
    ("viterbi", "linkstudio.phy.coding.convolutional.Trellis.decode", r"S = self.n_states", None),
    ("ldpc_encode", "linkstudio.phy.coding.ldpc.QCLDPCCode.encode", r"s = ", None),
    ("ldpc_minsum", "linkstudio.phy.coding.ldpc.QCLDPCCode.decode", r"if method == \"min_sum\"", r"out = np.prod"),
    ("polar_rel", "linkstudio.phy.coding.polar.reliability", r"def rec", None),
    ("polar_encode", "linkstudio.phy.coding.polar.polar_encode", r"x = u.copy", None),
    ("polar_sc", "linkstudio.phy.coding.polar.sc_decode", None, None),
    ("depuncture", "linkstudio.phy.coding.puncture.Depuncturer.process", r"m = llr.shape", None),
    ("cable", "linkstudio.phy.wireline.cable_response", r"pa, pb = ", None),
    ("xtalk", "linkstudio.phy.wireline.crosstalk_psd_gain", r"fa = ", None),
    ("echo_nlms", "linkstudio.phy.wireline.EchoCanceller.process", r"for n in range", r"w \+= mu"),
    ("dfe_loop", "linkstudio.phy.wireline.DFEEqualizer.process", r"for n in range", r"past = np.concatenate"),
    ("waterfill", "linkstudio.phy.wireline.waterfill_bits", r"gamma = ", None),
    ("dmt_hermitian", "linkstudio.phy.wireline.DMTModulator.process", r"X = np.zeros", r"out = np.concatenate"),
    ("jitter", "linkstudio.phy.wireline.Jitter.process", r"n_sym = ", r"t = n\["),
    ("wiener", "linkstudio.phy.optical._wiener_phase", None, None),
    ("im_mod", "linkstudio.phy.optical.IntensityModulator.process", r"drive = ", r"field = "),
    ("iq_mzm", "linkstudio.phy.optical.IQModulator.process", r"if p.model", r"field = d /"),
    ("disp_op", "linkstudio.phy.optical.dispersion_operator", None, None),
    ("beta2", "linkstudio.phy.optical.beta2", None, None),
    ("ssfm", "linkstudio.phy.optical.propagate", None, None),
    ("pmd", "linkstudio.phy.optical.apply_pmd", None, None),
    ("ase", "linkstudio.phy.optical.ase_psd", None, None),
    ("add_ase", "linkstudio.phy.optical._add_ase", None, None),
    ("osnr", "linkstudio.phy.optical.osnr_db", None, None),
    ("link_loop", "linkstudio.phy.optical.OpticalLink.process", r"g = 10", r"ase = ase"),
    ("photodiode", "linkstudio.phy.optical.Photodiode.process", r"M = p.apd_gain", r"current = "),
    ("coherent_rx", "linkstudio.phy.optical.CoherentReceiver.process", r"walk = ", r"out = scale"),
    ("cd_comp", "linkstudio.phy.optical.CDCompensation.process", r"H = np.conj", r"y = np.fft"),
    ("fo_est", "linkstudio.phy.optical.FOCompensation.process", r"nfft = ", r"rot = "),
    ("butterfly", "linkstudio.phy.optical.AdaptiveMIMOEqualizer.process", r"for k in range\(n_sym", r"W \+= mu"),
    ("bps", "linkstudio.phy.optical.PhaseRecovery.process", r"if p.method == \"viterbi", r"phi = np.unwrap"),
    ("wdm_demux", "linkstudio.phy.optical.WDMDemultiplexer.process", r"shift = ", r"y = np.fft"),
    ("fso", "linkstudio.phy.optical.FSOTurbulence.process", r"if p.model", r"gain = "),
]


def _resolve(dotted: str):
    import importlib
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(dotted)


def generate_snippets() -> None:
    SNIP.mkdir(parents=True, exist_ok=True)
    meta = ["% Generated by manual/tools/generate.py: \\snipdef{name}{file}{first line}{last line}\n"]
    for name, dotted, start, end in SNIPPETS:
        obj = inspect.unwrap(_resolve(dotted))
        lines, first = inspect.getsourcelines(obj)
        a, b = 0, len(lines) - 1
        if start is not None:
            a = next(i for i, l in enumerate(lines) if re.search(start, l))
            if start == r"\"\"\"$":  # skip past the end of the docstring
                a = next(i for i, l in enumerate(lines) if i > 1 and l.strip().endswith('"""')) + 1
        if end is not None:
            b = next(i for i, l in enumerate(lines) if i >= a and re.search(end, l))
        chunk = lines[a:b + 1]
        indent = min(len(l) - len(l.lstrip()) for l in chunk if l.strip())
        (SNIP / f"{name}.py").write_text("".join(l[indent:] if l.strip() else "\n" for l in chunk), encoding="utf-8")
        path = Path(inspect.getsourcefile(obj)).resolve().relative_to(BACKEND).as_posix()
        meta.append(f"\\snipdef{{{name}}}{{{path}}}{{{first + a}}}{{{first + b}}}\n")
    (OUT / "snippets.tex").write_text("".join(meta), encoding="utf-8")


# ==================================================================== tables
def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def doc_to_tex(s: str) -> str:
    """Escape a docstring, rendering ``literal`` spans in monospace."""
    parts = re.split(r"``(.+?)``", s)
    return "".join(f"\\texttt{{{tex_escape(p)}}}" if i % 2 else tex_escape(p) for i, p in enumerate(parts))


def _default(prop: dict) -> str:
    if "default" not in prop:
        return "required"
    v = prop["default"]
    if v is None:
        return "auto"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list) and len(v) > 6:
        return "[" + ", ".join(str(x) for x in v[:5]) + ", ...]"
    return str(v)


def generate_block_table() -> None:
    from linkstudio.core.registry import block_catalogue
    cat = block_catalogue()
    order = ["source", "modulation", "waveform", "channel", "sync", "receiver", "mimo", "coding",
             "wireline", "optical", "metrics"]
    by_cat: dict[str, list[str]] = {}
    for name, info in cat.items():
        by_cat.setdefault(info["category"], []).append(name)
    out = ["% Generated by manual/tools/generate.py from linkstudio.core.registry.block_catalogue()\n"]
    total = 0
    for c in order + sorted(set(by_cat) - set(order)):
        if c not in by_cat:
            continue
        out.append(f"\\blockcategory{{{tex_escape(c)}}}\n")
        for name in sorted(by_cat[c]):
            info = cat[name]
            total += 1
            doc = info["doc"].split("\n\n")[0].replace("\n", " ")
            ins = ", ".join(f"{tex_escape(p)}:{tex_escape(v['kinds'])}" for p, v in info["inputs"].items()) or "--"
            outs = ", ".join(f"{tex_escape(p)}:{tex_escape(v['kinds'])}" for p, v in info["outputs"].items()) or "--"
            props = info["params"].get("properties", {})
            params = "; ".join(f"\\texttt{{{tex_escape(k)}}}={tex_escape(_default(v))}" for k, v in props.items())
            out.append(f"\\blockentry{{{tex_escape(name)}}}{{{doc_to_tex(doc)}}}{{{ins}}}{{{outs}}}"
                       f"{{{params or '--'}}}\n")
    (OUT / "blocks.tex").write_text("".join(out), encoding="utf-8")
    (OUT / "count_blocks.tex").write_text(f"\\def\\numblocks{{{total}}}\n", encoding="utf-8")


def generate_scenario_table() -> None:
    import yaml
    rows = []
    for path in sorted((BACKEND / "scenarios").rglob("*.yaml")):
        if "subsystems" in path.parts:
            continue
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        rel = path.relative_to(BACKEND / "scenarios").as_posix()
        mode = "sweep" if spec.get("sweep") else "simulate"
        rows.append((spec.get("category", "Examples (CLI)"), rel, spec.get("name", rel),
                     len(spec.get("blocks", {})), mode))
    order = ["Tutorials", "Wireless", "Coding", "Wireline", "Optical", "Examples (CLI)"]
    rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99, r[1]))
    out = ["% Generated by manual/tools/generate.py from backend/scenarios\n"]
    for cat_, rel, name, nb, mode in rows:
        out.append(f"{tex_escape(cat_)} & \\file{{{rel}}} & {tex_escape(name)} & {nb} & {mode}\\\\\n")
    (OUT / "scenarios.tex").write_text("".join(out), encoding="utf-8")
    (OUT / "count_scenarios.tex").write_text(f"\\def\\numscenarios{{{len(rows)}}}\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="Figures to regenerate.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-figures", action="store_true", help="Only excerpts and tables.")
    args = ap.parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return
    DATA.mkdir(parents=True, exist_ok=True)
    generate_snippets()
    generate_block_table()
    generate_scenario_table()
    print("excerpts and tables written")
    if args.no_figures:
        return
    failed = []
    for name in args.only or FIGURES:
        t = time.perf_counter()
        try:
            FIGURES[name]()
        except Exception as exc:  # keep generating the other figures
            failed.append(name)
            print(f"  {name:<22} FAILED: {exc!r}", flush=True)
            continue
        print(f"  {name:<22} {time.perf_counter() - t:6.1f} s", flush=True)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
