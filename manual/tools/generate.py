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
    """OSNR exercise of the course: 75 km spans, NF 5 dB, 0 dBm, B = R_s = 64 GHz."""
    spans = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60])
    sim, sim_gain = [], []
    for nsp in spans:
        for gain_db, out in ((None, sim), (14.8, sim_gain)):
            g = cs.Graph()
            g.create("src", "random_bits", n_bits=512)
            g.create("map", "mapper", scheme="psk", order=4, symbol_rate=64e9)
            g.create("tx", "pulse_shaper", sps=2)
            g.create("mod", "iq_modulator", power_dbm=0.0, linewidth_hz=0.0)
            g.create("link", "optical_link", n_spans=int(nsp), span_km=75.0, attenuation_db_km=0.2,
                     noise_figure_db=5.0, amplifier_gain_db=gain_db)
            g.chain("src", "map", "tx", "mod", "link")
            out.append(cs.Simulation(g, batch_size=1).run(n_batches=1)["link.osnr_rs_db"])
    fine = np.unique(np.round(np.geomspace(1, 60, 80)).astype(int))
    save("osnr_spans", {"spans": spans, "sim": sim, "sim_gain": sim_gain})
    save("osnr_spans_theory", {"spans": fine,
                               "transparent": [theory.osnr_link_db(0.0, int(m), 15.0, 5.0, 64e9) for m in fine],
                               "gain": [theory.osnr_link_db(0.0, int(m), 15.0, 5.0, 64e9, gain_db=14.8) for m in fine]})
    with open(OUT / "optical_osnr.tex", "w") as fh:
        fh.write(f"\\def\\osnrfirst{{{theory.osnr_link_db(0.0, 1, 15.0, 5.0, 64e9):.1f}}}\n")
        fh.write(f"\\def\\osnrforty{{{theory.osnr_link_db(0.0, 40, 15.0, 5.0, 64e9):.1f}}}\n")
        fh.write(f"\\def\\osnrdistributed{{{theory.osnr_distributed_db(0.0, 0.2, 1000.0, 5.0, 100e9):.1f}}}\n")


@figure
def fig_osnr_span_length():
    """1000 km link, 0 dBm, NF 5 dB, B = 100 GHz: OSNR against the span length."""
    lengths = np.array([10, 20, 25, 40, 50, 100, 125, 200])
    sim = []
    for ls in lengths:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=512)
        g.create("map", "mapper", scheme="psk", order=4, symbol_rate=100e9)
        g.create("tx", "pulse_shaper", sps=2)
        g.create("mod", "iq_modulator", power_dbm=0.0, linewidth_hz=0.0)
        g.create("link", "optical_link", n_spans=int(1000 // ls), span_km=float(ls), noise_figure_db=5.0)
        g.chain("src", "map", "tx", "mod", "link")
        sim.append(cs.Simulation(g, batch_size=1).run(n_batches=1)["link.osnr_rs_db"])
    ls = np.linspace(2, 200, 100)
    th = [theory.osnr_link_db(0.0, 1, 0.2 * x, 5.0, 100e9) - 10 * np.log10(1000 / x) for x in ls]
    save("osnr_span_length", {"ls": lengths, "sim": sim})
    save("osnr_span_length_theory", {"ls": ls, "osnr": th})


@figure
def fig_pam_imdd():
    """PAM-2 and PAM-4 IM/DD at 10 GBd into a PIN + TIA receiver against the course formula."""
    from linkstudio.phy.optical import dbm_to_w
    sigma = 20e-12 * np.sqrt(40e9 / 2)  # 20 pA/sqrt(Hz) over fs / 2 = 20 GHz
    cols_sim, cols_th = {}, {}
    fine = np.linspace(-30, -10, 101)
    cols_th["p"] = fine
    for order, grid in ((2, np.arange(-25.0, -18.9, 1.0)), (4, np.arange(-18.0, -11.9, 1.0))):
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=8000)
        g.create("map", "mapper", scheme="pam", order=order, symbol_rate=10e9)
        g.create("tx", "pulse_shaper", shape="rect", sps=4)
        g.create("mod", "intensity_modulator", power_dbm=-20.0, extinction_ratio_db=10.0)
        g.create("pd", "photodiode", thermal_noise_pa_rthz=20.0, shot_noise=False, dark_current_na=0.0)
        g.create("rx", "matched_filter", shape="none")
        g.create("agc", "agc")
        g.create("demap", "demapper", method="hard")
        g.create("ber", "ber_meter")
        g.chain("src", "map", "tx", "mod", "pd", "rx", "agc", "demap", "ber.rx")
        g.connect("src", "ber.tx")
        cols_sim[f"p{order}"] = grid
        cols_sim[f"ber{order}"] = sweep_ber(g, "mod.power_dbm", grid, max_batches=40, batch_size=16)
        cols_th[f"ber{order}"] = theory.pam_imdd_ber(order, dbm_to_w(fine), 10.0, 0.8, sigma)
    save("pam_imdd_sim", cols_sim)
    save("pam_imdd_theory", cols_th)


@figure
def fig_snr_receivers():
    """Electrical SNR of PIN + TIA and APD + TIA receivers and the quantum limit (10 Gb/s, B_eq = 10 GHz)."""
    from linkstudio.core import Kind, Signal
    from linkstudio.core.registry import create_block
    p_dbm = np.linspace(-40, 0, 81)
    p = 1e-3 * 10 ** (p_dbm / 10)
    beq = 10e9
    save("snr_receivers", {"p": p_dbm,
                           "quantum": 10 * np.log10(theory.snr_quantum(p, 1 / (2 * beq))),
                           "pin": 10 * np.log10(theory.snr_photodiode(p, 1.0, 20e-12, beq)),
                           "pin_tia": 10 * np.log10(theory.snr_photodiode(p, 1.0, 20e-12, beq, shot=False)),
                           "apd": 10 * np.log10(theory.snr_photodiode(p, 1.0, 20e-12, beq, 10.0, 7.0))})
    sims = {"p": np.arange(-35.0, 0.1, 5.0)}
    for name, params in (("pin", {}), ("apd", {"apd_gain": 10.0, "apd_excess_noise": "power_law",
                                               "apd_x": np.log(7) / np.log(10)})):
        vals = []
        for x in sims["p"]:
            pw = 1e-3 * 10 ** (x / 10)
            sig = Signal(np.full((8, 1 << 14), np.sqrt(pw), dtype=complex), Kind.SAMPLES, fs=2 * beq,
                         meta={"optical": True, "wavelength_nm": 1550.0})
            blk = create_block("photodiode", responsivity=1.0, thermal_noise_pa_rthz=20.0, dark_current_na=0.0,
                               **params)
            blk.reset(3, "pd")
            i = blk.process({"in": sig}, None)["out"].data
            vals.append(10 * np.log10(np.mean(i) ** 2 / np.var(i)))
        sims[name] = vals
    save("snr_receivers_sim", sims)


@figure
def fig_vcsel_li():
    """Static L-I curves of the VCSEL with self-heating at four ambient temperatures (Shi et al., Fig. 1)."""
    from linkstudio.core.registry import create_block
    blk = create_block("vcsel")
    i = np.linspace(0, 30e-3, 121)
    cols = {"i_ma": i * 1e3}
    for t0 in (25, 50, 65, 80):
        p, t = blk.light_current(i, ambient_c=float(t0))
        cols[f"p{t0}"] = p * 1e3
        cols[f"t{t0}"] = t
    save("vcsel_li", cols)
    from linkstudio.phy.lasers import threshold_current
    leak, temp = blk.operating_point()
    with open(OUT / "optical_vcsel.tex", "w") as fh:
        fh.write(f"\\def\\vcselith{{{threshold_current(blk.re) * 1e3:.2f}}}\n")
        fh.write(f"\\def\\vcseltemp{{{temp:.0f}}}\n")
        fh.write(f"\\def\\vcselleak{{{leak * 1e3:.2f}}}\n")
        h = blk.re
        fh.write(f"\\def\\vcselhzero{{{h.h0:.3f}}}\\def\\vcselhone{{{h.h1:.3f}}}"
                 f"\\def\\vcselhtwo{{{h.h2:.3f}}}\\def\\vcselhthree{{{h.h3:.3f}}}\\def\\vcselhd{{{h.diffusion:.2f}}}\n")


@figure
def fig_vcsel_transient():
    """Square current between 0.5 and 5 mA: secondary pulse from spatial hole burning (Shi et al., Fig. 2)."""
    from linkstudio.core import Kind, Signal
    from linkstudio.core.registry import create_block
    fs = 1e12
    n = 10000  # 10 ns, two 5 ns periods
    drive = np.where((np.arange(n) // 2500) % 2 == 0, 1.0, -1.0)[None, :]
    sig = Signal(drive, Kind.SAMPLES, fs=fs, meta={})
    cols = {"t_ns": np.arange(n)[::5] / fs * 1e9}
    zoom = {"t_ns": np.arange(2400, 3400) / fs * 1e9}
    for name, shb in (("shb", True), ("uniform", False)):
        blk = create_block("vcsel", bias_ma=2.75, modulation_ma=2.25, spatial_hole_burning=shb, thermal=False,
                           rin_db_hz=None, parasitic_ghz=None)
        blk.reset(0, "vcsel")
        p = np.abs(blk.process({"in": sig}, None)["out"].data[0]) ** 2 * 1e3
        cols[name] = p[::5]
        zoom[name] = p[2400:3400]
    save("vcsel_transient", cols)
    save("vcsel_turnoff", zoom)


@figure
def fig_laser_response():
    """Small-signal modulation response of the VCSEL at three bias currents, with simulated points."""
    from linkstudio.core.registry import create_block
    from linkstudio.phy.lasers import simulate, small_signal_response
    blk = create_block("vcsel", thermal=False, parasitic_ghz=None)
    f = np.linspace(0.05e9, 40e9, 400)
    cols = {"f": f / 1e9}
    for ib in (2, 4, 8):
        cols[f"i{ib}"] = 20 * np.log10(np.abs(small_signal_response(blk.re, ib * 1e-3, f)))
    save("laser_response", cols)
    fs, n = 1.6e12, 1 << 14
    t = np.arange(n) / fs
    pts = {"f": [], "h": []}
    bias = 4e-3
    p0 = None
    for fk in (2e9, 6e9, 10e9, 14e9, 18e9, 22e9, 28e9, 34e9):
        fk = round(fk * n / fs) * fs / n
        p, _ = simulate(blk.re, bias + 0.05e-3 * np.sin(2 * np.pi * fk * t)[None, :], fs, preroll=n // 2)
        amp = 2 * np.abs(np.mean(p[0] * np.exp(-2j * np.pi * fk * t)))
        if p0 is None:  # normalize with the static slope dP/dI
            from linkstudio.phy.lasers import static_power
            p0 = (static_power(blk.re, bias + 1e-6)[0] - static_power(blk.re, bias - 1e-6)[0]) / 2e-6 * 0.05e-3
        pts["f"].append(fk / 1e9)
        pts["h"].append(20 * np.log10(amp / p0))
    save("laser_response_sim", pts)


@figure
def fig_dml_chirp():
    """Power and instantaneous frequency of a directly modulated DFB laser (10 Gb/s pattern)."""
    from linkstudio.core import Kind, Signal
    from linkstudio.core.registry import create_block
    fs, sps = 640e9, 64
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    drive = np.repeat(2.0 * bits - 1, sps)[None, :]
    blk = create_block("dml", bias_ma=40.0, modulation_ma=25.0, parasitic_ghz=25.0)
    blk.reset(0, "dml")
    y = blk.process({"in": Signal(drive, Kind.SAMPLES, fs=fs, meta={})}, None)["out"].data[0]
    p = np.abs(y) ** 2
    freq = np.gradient(np.unwrap(np.angle(y))) * fs / (2 * np.pi)
    t = np.arange(y.size) / fs * 1e12
    keep = slice(0, None, 2)
    save("dml_chirp", {"t_ps": t[keep], "p_mw": p[keep] * 1e3, "f_ghz": freq[keep] / 1e9,
                       "drive": drive[0][keep]})


@figure
def fig_cd_limit():
    """Eye-opening penalty of chirp-free IM/DD against L / L_max for three bit rates (noise-free)."""
    from linkstudio.core import Kind, Signal
    from linkstudio.core.registry import create_block
    from linkstudio.phy.dsp.filters import rc_taps
    rng = np.random.default_rng(1)
    sps = 16
    cols = {"x": np.linspace(0.0, 2.0, 21)}
    for name, rb, order in (("ook10", 10e9, 2), ("ook25", 25e9, 2), ("pam50", 50e9, 4)):
        nb = int(np.log2(order))
        rs = rb / nb
        lmax = theory.cd_limit_km(16.0, rb, nb)
        levels = rng.integers(0, order, 2048)
        a = 2 * levels / (order - 1) - 1
        up = np.zeros(a.size * sps)
        up[::sps] = a
        taps = rc_taps(sps, 16, 0.5)
        drive = np.convolve(up, taps / taps.max(), mode="same")[None, :]
        sig = Signal(drive, Kind.SAMPLES, fs=rs * sps, meta={})
        tx = create_block("intensity_modulator", power_dbm=0.0, extinction_ratio_db=15.0)
        tx.reset(0, "tx")
        e = tx.process({"in": sig}, None)["out"]
        def opening(p: np.ndarray) -> float:
            """Smallest vertical eye opening at the best sampling phase."""
            best = 0.0
            for off in range(sps):
                s = p[off::sps][64:-64]
                lv = levels[64:-64][: s.size]
                best = max(best, min(np.min(s[lv == k + 1]) - np.max(s[lv == k]) for k in range(order - 1)))
            return best

        ref = opening(np.abs(e.data[0]) ** 2)
        pen = []
        for x in cols["x"]:
            fib = create_block("fiber", length_km=float(x * lmax), dispersion_ps_nm_km=16.0, attenuation_db_km=0.0)
            op = opening(np.abs(fib.process({"in": e}, None)["out"].data[0]) ** 2)
            pen.append(-10 * np.log10(op / ref) if op > 0 else np.nan)
        cols[name] = pen
    save("cd_limit", cols)


@figure
def fig_raman():
    """Signal power along a 100 km span with a counter-propagating Raman pump (and the link metrics)."""
    from linkstudio.phy.optical import RamanGain
    z = np.linspace(0, 100e3, 201)
    alpha = 0.2 / (10 * np.log10(np.e)) / 1e3
    cols = {"z": z / 1e3, "passive": -0.2 * z / 1e3}
    for pump in (300, 600):
        r = RamanGain(pump * 1e-3, 0.4, 0.25, 100e3)
        cols[f"p{pump}"] = 10 * np.log10(np.e) * (np.array([r.integral(0.0, zi) for zi in z]) - alpha * z)
    save("raman_profile", cols)
    pumps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800])
    rows = {"pump": pumps, "osnr": [], "nf": [], "onoff": []}
    for pump in pumps:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=512)
        g.create("map", "mapper", scheme="psk", order=4, symbol_rate=32e9)
        g.create("tx", "pulse_shaper", sps=2)
        g.create("mod", "iq_modulator", power_dbm=0.0, linewidth_hz=0.0)
        g.create("link", "optical_link", n_spans=10, span_km=100.0, raman_pump_mw=float(pump))
        g.chain("src", "map", "tx", "mod", "link")
        m = cs.Simulation(g, batch_size=1).run(n_batches=1).metrics
        rows["osnr"].append(m["link.osnr_db"])
        rows["nf"].append(m.get("link.equivalent_nf_db", 5.0))
        rows["onoff"].append(m.get("link.raman_on_off_gain_db", 0.0))
    save("raman_link", rows)


@figure
def fig_coherent_ber():
    """Back-to-back PM-QPSK and PM-16QAM with ASE noise loading against the constellation SNR."""
    cols = {}
    for order, losses in ((4, np.arange(37.0, 44.1, 1.0)), (16, np.arange(31.0, 38.1, 1.0))):  # SNR = 48.9 dB - loss
        snr, ber = [], []
        for loss in losses:
            g = cs.Graph()
            g.create("src", "random_bits", n_bits=16000 if order == 16 else 8000)
            g.create("map", "mapper", scheme="psk" if order == 4 else "qam", order=order, symbol_rate=32e9)
            g.create("dmx", "stream_demux", n_streams=2)
            g.create("tx", "pulse_shaper", sps=4, rolloff=0.1, span=32)
            g.create("mod", "iq_modulator", power_dbm=0.0, linewidth_hz=0.0)
            g.create("att", "optical_attenuator", loss_db=float(loss))
            g.create("amp", "optical_amplifier", mode="gain", gain_db=30.0, noise_figure_db=5.0)
            g.create("rx", "coherent_receiver", lo_linewidth_hz=0.0, shot_noise=False, thermal_noise_pa_rthz=0.0)
            g.create("mf", "matched_filter")
            g.create("agc", "agc", remove_dc=False)
            g.create("mux", "stream_mux")
            g.create("demap", "demapper", method="hard")
            g.create("ber", "ber_meter")
            g.chain("src", "map", "dmx", "tx", "mod", "att", "amp", "rx", "mf", "agc", "mux", "demap", "ber.rx")
            g.connect("src", "ber.tx")
            res = cs.Simulation(g, batch_size=8, seed=7).run(
                stop=cs.StopCriteria.errors("ber.bit_errors", 300, max_batches=30))
            snr.append(res["amp.osnr_rs_db"])
            ber.append(res["ber.ber"] if res["ber.bit_errors"] >= 10 else np.nan)
        cols[f"snr{order}"], cols[f"ber{order}"] = snr, ber
    save("coherent_ber_sim", cols)
    fine = np.linspace(4, 22, 91)
    save("coherent_ber_theory", {"snr": fine,
                                 "bpsk": theory.ber_pm_qam(2, fine), "qpsk": theory.ber_pm_qam(4, fine),
                                 "qam8": theory.ber_pm_qam(8, fine), "qam16": theory.ber_pm_qam(16, fine),
                                 "qam64": theory.ber_pm_qam(64, fine),
                                 "qpsk_exact": theory.ber_qam_esn0(4, fine), "qam16_exact": theory.ber_qam_esn0(16, fine)})


@figure
def fig_gn_launch():
    """Received SNR of the centre channel against the launch power: split-step simulation and GN model."""
    from linkstudio.phy.optical import ase_psd, beta2
    sc = cs.load_scenario(BACKEND / "scenarios" / "optical" / "wdm_nonlinear.yaml")
    g = sc.graph
    powers = np.arange(-4.0, 6.1, 1.0)
    sim = []
    for pw in powers:
        g.set_param("mod.power_dbm", float(pw))
        res = cs.Simulation(g, batch_size=2, seed=1).run(n_batches=1)
        sim.append(res["evm.mer_db"])
    rs, nch, df = 32e9, 5, 50e9
    eta = theory.gn_eta(1.3, 0.2, beta2(17.0, 1550.0) * 1e27, rs, 80.0, nch, df)
    p_ase = 2 * ase_psd(10 ** 1.6, 5.0, 1550.0) * rs
    fine = np.linspace(-6, 8, 141)
    pw = 1e-3 * 10 ** (fine / 10)
    save("gn_launch_sim", {"p": powers, "mer": sim})
    save("gn_launch_theory", {"p": fine, "gsnr": theory.gsnr_db(pw, 10, p_ase, eta),
                              "osnr": 10 * np.log10(pw / (10 * p_ase)),
                              "nli": 10 * np.log10(pw / (10 * eta * pw ** 3))})
    with open(OUT / "optical_gn.tex", "w") as fh:
        fh.write(f"\\def\\gnpopt{{{10 * np.log10(theory.gn_optimum_power(p_ase, eta) / 1e-3):.1f}}}\n")
        fh.write(f"\\def\\gneta{{{eta:.0f}}}\n")


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


@figure
def fig_pic_loop():
    from linkstudio.phy.wireline import cable_response
    f = np.geomspace(10e3, 2e6, 120)
    save("pic_loss", {"f_khz": f / 1e3, "model": -20 * np.log10(np.abs(cable_response(f, "pic_26awg", 1828.8))),
                      "fit": -20 * np.log10(np.abs(cable_response(f, "dsl_0.4mm", 1828.8)))})
    # ITU-T G.996.1 Table 2, loop Mid-C at 70 F
    save("pic_g9961", {"f_khz": [20, 40, 100, 200, 260, 300, 400, 500, 600, 780, 1100],
                       "table": [13.3, 16.2, 20.0, 23.4, 25.4, 26.8, 30.1, 33.2, 36.3, 41.3, 49.1]})


@figure
def fig_pmd_dgd():
    from linkstudio.phy.optical import pmd_transfer
    dw = 2 * np.pi * 1e8
    t = pmd_transfer(np.random.default_rng(4), 20000, np.array([-dw, 0.0, dw]), 1.0e-12, 10)
    td = (t[:, 2] - t[:, 0]) / (2 * dw)
    m = 1j * np.einsum("bij,bkj->bik", td, t[:, 1].conj())
    ev = np.linalg.eigvalsh((m + np.conj(np.transpose(m, (0, 2, 1)))) / 2)
    x = (ev[:, 1] - ev[:, 0]) / 1.0e-12
    hist, edges = np.histogram(x, bins=40, range=(0, 3), density=True)
    c = 0.5 * (edges[1:] + edges[:-1])
    save("pmd_dgd", {"x": c, "hist": hist, "maxwell": 32 * c ** 2 / np.pi ** 2 * np.exp(-4 * c ** 2 / np.pi)})


@figure
def fig_pathloss():
    from linkstudio.core.registry import create_block
    from linkstudio.phy.channel.propagation import free_space_loss_db, tr38901_path_loss_db
    d = np.geomspace(10, 1000, 60)
    cols = {"d": d, "fs": free_space_loss_db(np.hypot(d, 10 - 1.5), 3.5)}
    for sc, hbs in (("uma", 25.0), ("umi", 10.0)):
        for name, los in (("los", True), ("nlos", False)):
            cols[f"{sc}_{name}"] = tr38901_path_loss_db(sc, np.full(d.size, los), d, 3.5, hbs, 1.5)[0]
    sim = []
    for dist in d:
        blk = create_block("path_loss", model="umi", frequency_ghz=3.5, distance_m=float(dist))
        blk.reset(5, "pl")
        sim.append(float(np.mean(blk.draw_loss_db(4000)[0])))
    cols["umi_sim"] = np.where(np.arange(d.size) % 4 == 0, sim, np.nan)
    save("pathloss", cols)


@figure
def fig_mmf_profiles():
    from linkstudio.phy.multimode import C_LIGHT, optimum_alpha, overfilled_rms, profile_contrast
    from linkstudio.core.registry import create_block

    def fiber(**params):
        blk = create_block("multimode_fiber", model="mode_groups", length_m=1000.0, **params)
        blk.setup()
        return blk

    delta, n1g = profile_contrast(0.2, 1.467), 1.482
    per_km_ps = n1g / C_LIGHT * 1e3 * 1e12  # relative delay -> ps/km
    fine = np.linspace(1.90, 2.10, 201)
    save("mmf_rms_theory", {"alpha": fine, "rms": [overfilled_rms(a, delta) * per_km_ps for a in fine]})
    coarse = np.round(np.arange(1.90, 2.1001, 0.01), 3)
    save("mmf_rms_sim", {"alpha": coarse, "rms": [fiber(alpha=float(a)).modal_rms_s() * 1e12 for a in coarse]})
    a_opt = optimum_alpha(delta)
    for name, blk in (("opt", fiber(alpha=a_opt)), ("parabolic", fiber(alpha=2.0)),
                      ("om4", fiber(fiber_type="OM4")), ("om3", fiber(fiber_type="OM3"))):
        w, d = blk.groups
        g = len(w)
        save(f"mmf_delays_{name}", {"x": (np.arange(1, g + 1) - 0.5) / g, "d": d * 1e3 * 1e12})
    with open(OUT / "mmf.tex", "w") as fh:
        fh.write(f"\\newcommand{{\\mmfalphaopt}}{{{a_opt:.4f}}}\n")
        fh.write(f"\\newcommand{{\\mmfalphaomfour}}{{{fiber(fiber_type='OM4').profile_alpha(850):.4f}}}\n")
        fh.write(f"\\newcommand{{\\mmfalphaomthree}}{{{fiber(fiber_type='OM3').profile_alpha(850):.4f}}}\n")
        fh.write(f"\\newcommand{{\\mmfdelta}}{{{delta:.4f}}}\n")
        fh.write(f"\\newcommand{{\\mmfgroups}}{{{len(fiber(fiber_type='OM4').groups[0])}}}\n")
        fh.write(f"\\newcommand{{\\mmfsigmamin}}{{{overfilled_rms(a_opt, delta) * per_km_ps:.1f}}}\n")


@figure
def fig_mmf_launch():
    from scipy.stats import poisson
    from linkstudio.phy.multimode import gaussian_launch_weights
    w0 = np.sqrt(25e-6 * 850e-9 / (np.pi * 0.2)) * 1e6
    p = np.arange(12)
    offset = gaussian_launch_weights(19, w0, 1.5 * w0, w0)[:12]
    centre = gaussian_launch_weights(19, 4.0, 0.0, w0)[:12]
    t = ((4.0 ** 2 - w0 ** 2) / (4.0 ** 2 + w0 ** 2)) ** 2
    save("mmf_launch", {"p": p, "offset_sim": offset, "offset_th": poisson.pmf(p, 2.25),
                        "centre_sim": centre, "centre_th": np.where(p % 2 == 0, (1 - t) * t ** (p // 2), 0.0)})


@figure
def fig_mmf_eyes():
    from linkstudio.phy.dsp.analysis import eye_traces
    sc = cs.load_scenario(BACKEND / "scenarios" / "optical" / "datacenter_vcsel_mmf.yaml")
    g = sc.graph
    sps = 16
    g.set_param("tx.sps", sps)
    g.set_param("laser.rin_db_hz", None)
    x = np.arange(2 * sps + 1) / sps - 1.0
    for name, fiber_type, length in (("om4_100", "OM4", 100.0), ("om4_300", "OM4", 300.0), ("om3_300", "OM3", 300.0)):
        g.set_param("fib.fiber_type", fiber_type)
        g.set_param("fib.length_m", length)
        sig = capture(g, ["pd.out"], batch=1)["pd.out"]
        start = int(round(sig.meta["delay"])) - sps + 20 * sps
        tr = eye_traces(sig.data.real, sps, offset=start, symbols_per_trace=2, max_traces=80)
        save_traces(f"eye_{name}", tr * 1e3, x)  # mA


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
    ("parallel_merge", "linkstudio.core.parallel.run_batches", r"while reason is None and next_merge", r"reason = done"),
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
    ("wiener", "linkstudio.phy.optical.wiener_phase", r"if linewidth", None),
    ("im_mod", "linkstudio.phy.optical.IntensityModulator.process", r"er = 10", r"field = "),
    ("iq_mzm", "linkstudio.phy.optical.IQModulator.process", r"if p.model", r"field = d /"),
    ("disp_op", "linkstudio.phy.optical.dispersion_operator", None, None),
    ("beta2", "linkstudio.phy.optical.beta2", None, None),
    ("ssfm", "linkstudio.phy.optical.propagate", None, None),
    ("pmd_transfer", "linkstudio.phy.optical.pmd_transfer", r"t = np.broadcast_to", None),
    ("ase", "linkstudio.phy.optical.ase_psd", None, None),
    ("add_ase", "linkstudio.phy.optical._add_ase", None, None),
    ("osnr", "linkstudio.phy.optical.osnr_db", None, None),
    ("link_loop", "linkstudio.phy.optical.OpticalLink.process", r"for _ in range", r"power = power"),
    ("photodiode", "linkstudio.phy.optical.Photodiode.process", r"M, F = ", r"current = M"),
    ("coherent_rx", "linkstudio.phy.optical.CoherentReceiver.process", r"walk = ", r"noise = np.sqrt"),
    ("cd_comp", "linkstudio.phy.optical.CDCompensation.process", r"H = np.conj", r"y = np.fft"),
    ("fo_est", "linkstudio.phy.optical.FOCompensation.process", r"nfft = ", r"rot = "),
    ("butterfly", "linkstudio.phy.optical.AdaptiveMIMOEqualizer.process", r"for k in range\(n_sym", r"W \+= mu"),
    ("bps", "linkstudio.phy.optical.PhaseRecovery.process", r"if p.method == \"viterbi", r"phi = np.unwrap"),
    ("wdm_demux", "linkstudio.phy.optical.WDMDemultiplexer.process", r"shift = ", r"y = np.fft"),
    ("fso", "linkstudio.phy.optical.FSOTurbulence.draw", r"s2, a, b = ", None),
    ("fso_params", "linkstudio.phy.optical.gamma_gamma_parameters", r"s125 = ", None),
    ("rin", "linkstudio.phy.optical.intensity_noise", r"if rin_db_hz", None),
    ("mmf_delay", "linkstudio.phy.multimode.group_delay", r"x = np.asarray", None),
    ("mmf_rms", "linkstudio.phy.multimode.overfilled_rms", r"q = alpha", None),
    ("mmf_chromatic", "linkstudio.phy.multimode.chromatic_rms_ps_per_km", r"d1 = ", None),
    ("mmf_groups", "linkstudio.phy.multimode.MultimodeFiber._weights_delays", r"g = max", None),
    ("mmf_launch", "linkstudio.phy.multimode.gaussian_launch_weights", r"xi = np.linspace", None),
    ("mmf_response", "linkstudio.phy.multimode.MultimodeFiber.response", r"p = self.params", None),
    ("mmf_default_alpha", "linkstudio.phy.multimode.MultimodeFiber.profile_alpha", r"p = self.params", None),
    ("laser_rhs", "linkstudio.phy.lasers._rhs", r"s = np.exp", None),
    ("laser_steady", "linkstudio.phy.lasers._carriers", r"g = re.g0", None),
    ("laser_small_signal", "linkstudio.phy.lasers.small_signal_response", r"n0, n1, s = ", None),
    ("laser_overlap", "linkstudio.phy.lasers.overlap_coefficients", r"R, Rm = ", None),
    ("vcsel_thermal", "linkstudio.phy.lasers.thermal_operating_point", r"i = np.atleast_1d", None),
    ("vcsel_vi", "linkstudio.phy.lasers.ThermalModel.voltage", r"i, t = ", None),
    ("laser_block", "linkstudio.phy.lasers._LaserBlock.process", r"current = p.bias_ma", r"phase = phase"),
    ("mzm_field", "linkstudio.phy.optical.mzm_field", r"d = 0.0", None),
    ("mzm_block", "linkstudio.phy.optical.MachZehnderModulator.process", r"bias, swing = ", r"field = e_in"),
    ("raman", "linkstudio.phy.optical.RamanGain.ase_psd", r"alpha = ", None),
    ("amp_sat", "linkstudio.phy.optical.saturated_gain", r"if p_in_w", None),
    ("link_gn", "linkstudio.phy.optical.OpticalLink.process", r"wdm = x.meta", r"\"gn_optimum_launch_dbm\""),
    ("gn_eta", "linkstudio.phy.theory.gn_eta", r"alpha = ", None),
    ("osnr_link", "linkstudio.phy.theory.osnr_link_db", r"a = 10", None),
    ("pam_ser", "linkstudio.phy.theory.pam_imdd_ser", r"er = 10", None),
    ("coh_isnr", "linkstudio.phy.optical.CoherentReceiver.process", r"if p.hybrid_phase_error_deg", r"var \+= sig_power"),
    ("delay_kernel", "linkstudio.phy.channel.fading.delay_kernel", r"whole = ", None),
    ("tr38901", "linkstudio.phy.channel.propagation.tr38901_path_loss_db", r"d2d = ", None),
    ("dielectric", "linkstudio.phy.wireline.dielectric_response", r"if loss_db_m_hz == 0", None),
    ("pic", "linkstudio.phy.wireline.pic_response", r"r, ind, g, c = ", None),
    ("fso_channel", "linkstudio.phy.optical.FSOChannel.process", r"atm = ", r"out = x.data"),
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
        obj = getattr(obj, "py_func", obj)  # numba-compiled functions keep the Python source
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
