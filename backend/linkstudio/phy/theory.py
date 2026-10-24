"""Closed-form performance references used to validate the simulator.

All functions take Eb/N0 or Es/N0 in **dB** and accept scalars or arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import erfc


def db2lin(x_db: np.ndarray | float) -> np.ndarray:
    return 10 ** (np.asarray(x_db, dtype=float) / 10)


def qfunc(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail probability Q(x)."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def ber_bpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """BPSK (and Gray QPSK) over AWGN: Q(sqrt(2 Eb/N0))."""
    return qfunc(np.sqrt(2 * db2lin(ebn0_db)))


def ber_pam(order: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Exact BER of Gray-labelled M-PAM over AWGN (Cho & Yoon, 2002)."""
    g = db2lin(ebn0_db)
    L = order
    k_bits = int(np.log2(L))
    total = np.zeros_like(g)
    for k in range(1, k_bits + 1):
        pk = np.zeros_like(g)
        for i in range(int((1 - 2.0 ** -k) * L)):
            w = (-1) ** int(i * 2 ** (k - 1) / L) * (2 ** (k - 1) - int(i * 2 ** (k - 1) / L + 0.5))
            pk += w * erfc((2 * i + 1) * np.sqrt(3 * k_bits * g / (L ** 2 - 1)))
        total += pk / L
    return total / k_bits


def ber_qam(order: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Exact BER of square Gray M-QAM over AWGN (product of two sqrt(M)-PAM)."""
    side = int(round(np.sqrt(order)))
    return ber_pam(side, ebn0_db)


def ser_psk(order: int, esn0_db: np.ndarray | float) -> np.ndarray:
    """Exact symbol error rate of M-PSK over AWGN (Craig's integral)."""
    g = np.atleast_1d(db2lin(esn0_db))
    s2 = np.sin(np.pi / order) ** 2
    out = np.array([integrate.quad(lambda th, gi=gi: np.exp(-gi * s2 / np.sin(th) ** 2),
                                   0, (order - 1) * np.pi / order)[0] / np.pi for gi in g])
    return out if np.ndim(esn0_db) else out[0]


def ber_psk(order: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """BER of Gray M-PSK over AWGN: exact for M = 2, 4; SER/k approximation otherwise."""
    if order in (2, 4):
        return ber_bpsk(ebn0_db)
    k = np.log2(order)
    esn0_db = np.asarray(ebn0_db, dtype=float) + 10 * np.log10(k)
    return ser_psk(order, esn0_db) / k


def ber_bpsk_rayleigh(ebn0_db: np.ndarray | float) -> np.ndarray:
    """BPSK over flat Rayleigh fading with coherent detection: 0.5 (1 - sqrt(g / (1 + g)))."""
    g = db2lin(ebn0_db)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def ber_bpsk_mrc(branches: int, ebn0_db_per_branch: np.ndarray | float) -> np.ndarray:
    """BPSK with L-branch maximal-ratio combining over i.i.d. Rayleigh fading (Proakis)."""
    from scipy.special import comb
    g = db2lin(ebn0_db_per_branch)
    mu = np.sqrt(g / (1 + g))
    p = (1 - mu) / 2
    total = sum(comb(branches - 1 + l, l) * ((1 + mu) / 2) ** l for l in range(branches))
    return p ** branches * total


def ber_rayleigh(awgn_ber, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Average any AWGN BER curve ``awgn_ber(ebn0_db)`` over Rayleigh fading (numerical)."""
    def one(ebn0: float) -> float:
        gbar = db2lin(ebn0)
        f = lambda g: awgn_ber(10 * np.log10(max(g, 1e-30))) * np.exp(-g / gbar) / gbar
        return integrate.quad(f, 0, 50 * gbar, limit=200)[0]
    vals = np.array([one(e) for e in np.atleast_1d(ebn0_db)])
    return vals if np.ndim(ebn0_db) else vals[0]


def ber_hamming_hard(m: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Approximate post-decoding BER of a Hamming(2^m-1) code, BPSK, hard decisions."""
    n = 2 ** m - 1
    k = n - m
    p = ber_bpsk(np.asarray(ebn0_db, dtype=float) + 10 * np.log10(k / n))
    from scipy.special import comb
    total = np.zeros_like(p)
    for j in range(2, n + 1):
        total += (j + 1) * comb(n, j) * p ** j * (1 - p) ** (n - j)
    return total / n


def q_to_ber(q: np.ndarray | float) -> np.ndarray:
    """BER of a binary decision with Q factor q: 0.5 erfc(q / sqrt(2))."""
    return 0.5 * erfc(np.asarray(q, dtype=float) / np.sqrt(2))


def ook_q_factor(p_avg_w: float, extinction_ratio_db: float, responsivity: float,
                 sigma0: float, sigma1: float | None = None) -> float:
    """Q factor of an IM/DD OOK receiver: (I1 - I0) / (sigma1 + sigma0)."""
    er = 10 ** (extinction_ratio_db / 10)
    p1, p0 = 2 * p_avg_w * er / (er + 1), 2 * p_avg_w / (er + 1)
    s1 = sigma0 if sigma1 is None else sigma1
    return responsivity * (p1 - p0) / (sigma0 + s1)


def osnr_to_esn0_db(osnr_db: np.ndarray | float, symbol_rate: float, polarizations: int = 2,
                    ref_bandwidth: float = 12.5e9) -> np.ndarray:
    """Es/N0 per polarization from the OSNR in the reference bandwidth (0.1 nm)."""
    return np.asarray(osnr_db, dtype=float) + 10 * np.log10(2 * ref_bandwidth / (polarizations * symbol_rate))


def ber_qam_esn0(order: int, esn0_db: np.ndarray | float) -> np.ndarray:
    """Square Gray M-QAM BER (QPSK for M = 4) as a function of Es/N0."""
    return ber_qam(order, np.asarray(esn0_db, dtype=float) - 10 * np.log10(np.log2(order)))


def gaussian_pulse_broadening(t0_s: float, beta2_s2_m: float, length_m: float) -> float:
    """RMS-width broadening factor T1/T0 of an unchirped Gaussian pulse by dispersion."""
    return float(np.sqrt(1 + (beta2_s2_m * length_m / t0_s ** 2) ** 2))


def shannon_capacity(snr_db: np.ndarray | float) -> np.ndarray:
    """AWGN capacity in bit/s/Hz."""
    return np.log2(1 + db2lin(snr_db))


def ebn0_min_db(spectral_efficiency: float) -> float:
    """Minimum Eb/N0 for reliable communication at the given bit/s/Hz (Shannon)."""
    eta = spectral_efficiency
    return float(10 * np.log10((2 ** eta - 1) / eta))
