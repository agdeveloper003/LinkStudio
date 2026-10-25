def dispersion_operator(n: int, fs: float, cd_ps_nm: float, wavelength_nm: float,
                        slope_ps_nm2: float = 0.0) -> np.ndarray:
    """All-pass dispersion filter for an accumulated D*L (ps/nm) and slope S*L (ps/nm^2).

    exp(j (beta2 L w^2 / 2 - beta3 L w^3 / 6)) on the NumPy frequency grid. The envelope
    follows Agrawal's convention (carrier exp(-j w0 t)), so a positive NumPy frequency is a
    lower optical frequency; hence the minus sign of the odd beta3 term.
    """
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
    b2l = beta2(cd_ps_nm, wavelength_nm) * 1e3  # D*L in ps/nm == (ps/nm/km) * km
    b3l = beta3(cd_ps_nm, slope_ps_nm2, wavelength_nm) * 1e3
    return np.exp(1j * (b2l / 2 * w ** 2 - b3l / 6 * w ** 3))
