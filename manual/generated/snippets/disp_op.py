def dispersion_operator(n: int, fs: float, cd_ps_nm: float, wavelength_nm: float) -> np.ndarray:
    """exp(j beta2 L / 2 w^2) for an accumulated dispersion D*L in ps/nm."""
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
    b2l = beta2(cd_ps_nm, wavelength_nm) * 1e3  # D*L in ps/nm == (ps/nm/km) * km
    return np.exp(1j * b2l / 2 * w ** 2)
