def ase_psd(gain_lin: float, nf_db: float, wavelength_nm: float) -> float:
    """ASE power spectral density per polarization (W/Hz): S_ASE = N_0 = n_sp h f (G - 1), n_sp = F / 2.

    Each of the four field quadratures carries white noise of density N_0 / 2, so the ASE
    power in a bandwidth B is P_ASE = 2 N_0 B = h f (G - 1) F B.
    """
    if gain_lin <= 1:
        return 0.0
    nu = C_LIGHT / (wavelength_nm * 1e-9)
    return 10 ** (nf_db / 10) / 2 * H_PLANCK * nu * (gain_lin - 1)
