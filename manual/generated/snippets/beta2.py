def beta2(dispersion_ps_nm_km: float, wavelength_nm: float) -> float:
    """Group-velocity dispersion beta2 in s^2/m."""
    D = dispersion_ps_nm_km * 1e-6  # s/m^2
    lam = wavelength_nm * 1e-9
    return -D * lam ** 2 / (2 * np.pi * C_LIGHT)
