def propagate(e: np.ndarray, fs: float, length_km: float, att_db_km: float, d_ps_nm_km: float,
              wavelength_nm: float, nonlinear: bool, gamma: float, steps_per_km: float,
              slope_ps_nm2_km: float = 0.0, nonlinear_phase_rad: float | None = None,
              gain: RamanGain | None = None) -> np.ndarray:
    """Split-step Fourier solution of the NLSE (Manakov equation for dual polarization).

    The step is at most 1 / ``steps_per_km`` km. With ``nonlinear_phase_rad`` it is also
    limited so that the peak nonlinear phase per step, gamma P_peak L_eff, stays below that
    value (nonlinear-phase-rotation criterion of Sinkin et al. 2003). ``gain`` adds a
    distributed (Raman) power gain g(z) to the attenuation.
    """
    n = e.shape[-1]
    alpha = att_db_km / (10 * np.log10(np.e)) / 1e3  # 1/m
    total = length_km * 1e3

    def amp(z1: float, z2: float) -> float:
        """Field factor exp(int (g - alpha) dz / 2) over [z1, z2]."""
        return float(np.exp(((gain.integral(z1, z2) if gain else 0.0) - alpha * (z2 - z1)) / 2))

    if not nonlinear or gamma == 0:
        lin = dispersion_operator(n, fs, d_ps_nm_km * length_km, wavelength_nm, slope_ps_nm2_km * length_km)
        return np.fft.ifft(np.fft.fft(e, axis=-1) * lin * amp(0.0, total), axis=-1)
    max_dz = total / max(1, int(np.ceil(length_km * steps_per_km)))
    g = gamma * 1e-3 * (8 / 9 if e.ndim == 3 else 1.0)  # 1/(W km) -> 1/(W m)
    z = 0.0
    while z < total * (1 - 1e-12):
        dz = min(max_dz, total - z)
        if nonlinear_phase_rad:
            p_peak = float(np.max(np.sum(np.abs(e) ** 2, axis=1) if e.ndim == 3 else np.abs(e) ** 2))
            dz = min(dz, max(nonlinear_phase_rad / max(g * p_peak, 1e-30), 1.0))
        half = dispersion_operator(n, fs, d_ps_nm_km * dz / 2e3, wavelength_nm, slope_ps_nm2_km * dz / 2e3)
        # Effective length around the step midpoint (where the nonlinear phase is applied),
        # for the local net loss alpha - g.
        a = alpha - (gain.coefficient(z + dz / 2) if gain else 0.0)
        leff = 2 * np.sinh(a * dz / 2) / a if abs(a) > 1e-12 else dz
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half * amp(z, z + dz / 2), axis=-1)
        p = np.sum(np.abs(e) ** 2, axis=1, keepdims=True) if e.ndim == 3 else np.abs(e) ** 2
        e = e * np.exp(1j * g * p * leff)
        e = np.fft.ifft(np.fft.fft(e, axis=-1) * half * amp(z + dz / 2, z + dz), axis=-1)
        z += dz
    return e
