def ase_psd(gain_lin: float, nf_db: float, wavelength_nm: float) -> float:
    """ASE power spectral density per polarization (W/Hz) of an amplifier."""
    if gain_lin <= 1:
        return 0.0
    nu = C_LIGHT / (wavelength_nm * 1e-9)
    nsp = max(10 ** (nf_db / 10) * gain_lin / (2 * (gain_lin - 1)), 0.5)
    return nsp * H_PLANCK * nu * (gain_lin - 1)
