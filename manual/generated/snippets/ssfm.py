def propagate(e: np.ndarray, fs: float, length_km: float, att_db_km: float, d_ps_nm_km: float,
              wavelength_nm: float, nonlinear: bool, gamma: float, steps_per_km: float) -> np.ndarray:
    """Split-step Fourier solution of the NLSE (Manakov equation for dual polarization)."""
    n = e.shape[-1]
    alpha = att_db_km / (10 * np.log10(np.e)) / 1e3  # 1/m
    if not nonlinear or gamma == 0:
        lin = dispersion_operator(n, fs, d_ps_nm_km * length_km, wavelength_nm) * np.exp(-alpha * length_km * 1e3 / 2)
        return np.fft.ifft(np.fft.fft(e, axis=-1) * lin, axis=-1)
    steps = max(1, int(np.ceil(length_km * steps_per_km)))
    dz = length_km * 1e3 / steps
    half = dispersion_operator(n, fs, d_ps_nm_km * length_km / steps / 2, wavelength_nm) * np.exp(-alpha * dz / 4)
    # Effective length around the step midpoint (where the nonlinear phase is applied).
    leff = 2 * np.sinh(alpha * dz / 2) / alpha if alpha > 0 else dz
    g = gamma * 1e-3 * (8 / 9 if e.ndim == 3 else 1.0)  # 1/(W km) -> 1/(W m)
    for _ in range(steps):
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
        p = np.sum(np.abs(e) ** 2, axis=1, keepdims=True) if e.ndim == 3 else np.abs(e) ** 2
        e = e * np.exp(1j * g * p * leff)
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half, axis=-1)
    return e
