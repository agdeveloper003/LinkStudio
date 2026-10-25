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


# ------------------------------------------------------------------- optical
H_PLANCK = 6.62607015e-34
Q_E = 1.602176634e-19
C_LIGHT = 299_792_458.0


def photon_energy(wavelength_nm: float) -> float:
    """h f (J)."""
    return H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)


def responsivity(quantum_efficiency: float, wavelength_nm: float) -> float:
    """Photodiode responsivity R = q eta / (h f) (A/W)."""
    return Q_E * quantum_efficiency / photon_energy(wavelength_nm)


def pam_imdd_ser(order: int, p_avg_w: np.ndarray | float, extinction_ratio_db: float, resp: float,
                 sigma: float) -> np.ndarray:
    """Symbol error rate of IM/DD PAM-M with signal-independent Gaussian noise of std ``sigma`` (A).

    The M power levels are equally spaced between P_1 and P_M = ER P_1, so the level spacing
    is dP = 2 P_avg (ER - 1) / ((ER + 1)(M - 1)); with mid-way thresholds
    SER = (M - 1)/M erfc(R dP / (2 sqrt(2) sigma)).
    """
    er = 10 ** (extinction_ratio_db / 10)
    dp = 2 * np.asarray(p_avg_w, dtype=float) * (er - 1) / ((er + 1) * (order - 1))
    return (order - 1) / order * erfc(resp * dp / (2 * np.sqrt(2) * sigma))


def pam_imdd_ber(order: int, p_avg_w: np.ndarray | float, extinction_ratio_db: float, resp: float,
                 sigma: float) -> np.ndarray:
    """Gray-labelled IM/DD PAM-M: BER ~ SER / log2 M (one bit error per symbol error)."""
    return pam_imdd_ser(order, p_avg_w, extinction_ratio_db, resp, sigma) / np.log2(order)


def ber_quantum_limit(p_avg_w: np.ndarray | float, bit_rate: float, wavelength_nm: float = 1550.0) -> np.ndarray:
    """Quantum limit of OOK with an ideal photon counter: BER = exp(-N_1) / 2.

    A "1" carries N_1 = 2 P_avg / (h f R_B) photons on average (P_0 = 0); it is misread only
    when no photon arrives, with Poisson probability exp(-N_1).
    """
    return 0.5 * np.exp(-2 * np.asarray(p_avg_w, dtype=float) / (photon_energy(wavelength_nm) * bit_rate))


def snr_photodiode(p_w: np.ndarray | float, resp: float, irnd: float, b_eq: float, apd_gain: float = 1.0,
                   excess_noise: float = 1.0, shot: bool = True) -> np.ndarray:
    """Electrical SNR (R M P)^2 / (IRND^2 B_eq + 2 q B_eq M^2 F R P) of a PIN or APD + TIA receiver."""
    p = np.asarray(p_w, dtype=float)
    noise = irnd ** 2 * b_eq + (2 * Q_E * b_eq * apd_gain ** 2 * excess_noise * resp * p if shot else 0.0)
    return (resp * apd_gain * p) ** 2 / noise


def snr_quantum(p_w: np.ndarray | float, interval_s: float, wavelength_nm: float = 1550.0) -> np.ndarray:
    """Quantum-limited SNR of photon counting: the mean photon number P dT / (h f)."""
    return np.asarray(p_w, dtype=float) * interval_s / photon_energy(wavelength_nm)


def cd_limit_km(dispersion_ps_nm_km: float, bit_rate: float, bits_per_symbol: int = 1) -> float:
    """Dispersion limit of chirp-free IM/DD at 1550 nm: L = n_bit^2 / (10 |D| R_B^2) (R_B in Tb/s).

    From |delta tau| = |D| L delta lambda <= T_s / 2 with the spectral width
    delta f = 2/3 of the symbol rate (heuristic of the course notes).
    """
    return bits_per_symbol ** 2 / (10 * abs(dispersion_ps_nm_km) * (bit_rate / 1e12) ** 2)


def ase_power(gain_lin: float, nf_db: float, bandwidth: float, wavelength_nm: float = 1550.0) -> float:
    """ASE power of one amplifier in both polarizations: P_ASE = 2 N0 B = h f (G - 1) F B (n_sp = F / 2)."""
    return photon_energy(wavelength_nm) * (gain_lin - 1) * 10 ** (nf_db / 10) * bandwidth


def osnr_link_db(p_tx_dbm: float, n_spans: int, span_loss_db: float, nf_db: float, bandwidth: float,
                 gain_db: float | None = None, wavelength_nm: float = 1550.0) -> float:
    """OSNR after ``n_spans`` identical spans (loss A, amplifier gain G, noise figure F).

    P_ASE = h f B F (G - 1) (1 - (A G)^M) / (1 - A G), P_Rx = P_Tx (A G)^M; with transparent
    spans (G A = 1, ``gain_db`` None) P_ASE = M h f B F (G - 1).
    """
    a = 10 ** (-span_loss_db / 10)
    g = 1 / a if gain_db is None else 10 ** (gain_db / 10)
    ag = a * g
    acc = n_spans if abs(ag - 1) < 1e-12 else (1 - ag ** n_spans) / (1 - ag)
    p_rx = 1e-3 * 10 ** (p_tx_dbm / 10) * ag ** n_spans
    return float(10 * np.log10(p_rx / (ase_power(g, nf_db, bandwidth, wavelength_nm) * acc)))


def osnr_distributed_db(p_tx_dbm: float, attenuation_db_km: float, length_km: float, nf_db: float,
                        bandwidth: float, wavelength_nm: float = 1550.0) -> float:
    """Limit of ever shorter transparent spans: OSNR = P_Tx / (h f B F alpha_Np L) (alpha_Np: power, 1/km)."""
    alpha_np = attenuation_db_km * np.log(10) / 10
    p = 1e-3 * 10 ** (p_tx_dbm / 10)
    return float(10 * np.log10(p / (photon_energy(wavelength_nm) * bandwidth * 10 ** (nf_db / 10)
                                    * alpha_np * length_km)))


def gn_eta(gamma_w_km: float, attenuation_db_km: float, beta2_ps2_km: float, symbol_rate: float, span_km: float,
           n_channels: int = 1, spacing_hz: float | None = None) -> float:
    """NLI coefficient eta of one span of the incoherent GN model (P_NLI = eta P_ch^3, 1/W^2).

    Poggiolini's closed form for a Nyquist-like WDM comb of N_ch channels at symbol rate R_s
    and spacing delta f, dual polarization, NLI integrated over R_s:

        eta = 8/27 gamma^2 L_eff^2 asinh(pi^2/2 |beta2| L_eff,a R_s^2 N_ch^(2 R_s / delta f))
              / (pi |beta2| L_eff,a R_s^2)

    with alpha the field loss coefficient, L_eff = (1 - exp(-2 alpha L)) / (2 alpha) and
    L_eff,a = 1 / (2 alpha). For long spans L_eff -> L_eff,a and this is the course formula
    4/(27 pi) gamma^2 / (alpha |beta2| R_s^2) asinh(pi^2 |beta2| R_s^2 N_ch^(2 R_s/df) / (4 alpha)).
    """
    alpha = attenuation_db_km / (20 * np.log10(np.e)) / 1e3  # field, 1/m
    gamma = gamma_w_km * 1e-3
    b2 = abs(beta2_ps2_km) * 1e-27  # s^2/m
    leff = (1 - np.exp(-2 * alpha * span_km * 1e3)) / (2 * alpha)
    la = 1 / (2 * alpha)
    df = spacing_hz or symbol_rate
    arg = np.pi ** 2 / 2 * b2 * la * symbol_rate ** 2 * n_channels ** (2 * symbol_rate / df)
    return float(8 / 27 * gamma ** 2 * leff ** 2 * np.arcsinh(arg) / (np.pi * b2 * la * symbol_rate ** 2))


def gn_optimum_power(p_ase_span: float, eta: float) -> float:
    """Launch power per channel that maximizes the GSNR: P_opt = (P_ASE / (2 eta))^(1/3) (W).

    At P_opt the NLI power is half the ASE power (the "3 dB rule").
    """
    return float((p_ase_span / (2 * eta)) ** (1 / 3))


def gsnr_db(p_ch_w: np.ndarray | float, n_spans: int, p_ase_span: float, eta: float,
            isnr_db: float | None = None) -> np.ndarray:
    """GSNR = P / (M P_ASE + M eta P^3), optionally combined with the transceiver ISNR (1/SNR = 1/GSNR + 1/ISNR)."""
    p = np.asarray(p_ch_w, dtype=float)
    inv = n_spans * (p_ase_span + eta * p ** 3) / p
    if isnr_db is not None:
        inv = inv + 10 ** (-isnr_db / 10)
    return 10 * np.log10(1 / inv)


def ber_pm_qam(order: int, snr_db: np.ndarray | float) -> np.ndarray:
    """Course approximations of the BER of PM-QAM versus the constellation SNR E_s/N_0 per polarization.

    BPSK: erfc(sqrt(SNR))/2; QPSK: erfc(sqrt(SNR/2))/2; square M = L^2:
    (L - 1)/(L log2 L) erfc(sqrt(3 SNR / (2 (L^2 - 1)))); 8-QAM: 2/3 erfc(sqrt(3 SNR/14));
    32-QAM: 13/40 erfc(sqrt(SNR/20)). Accurate between BER 3e-2 and 1e-3 and below.
    """
    s = db2lin(snr_db)
    if order == 2:
        return 0.5 * erfc(np.sqrt(s))
    if order == 8:
        return 2 / 3 * erfc(np.sqrt(3 * s / 14))
    if order == 32:
        return 13 / 40 * erfc(np.sqrt(s / 20))
    L = int(round(np.sqrt(order)))
    if L * L != order:
        raise ValueError(f"No PM-QAM approximation for order {order}.")
    return (L - 1) / (L * np.log2(L)) * erfc(np.sqrt(3 * s / (2 * (L * L - 1))))


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
