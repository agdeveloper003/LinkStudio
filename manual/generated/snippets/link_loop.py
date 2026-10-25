g = 10 ** (p.attenuation_db_km * p.span_km / 10)
s = ase_psd(g, p.noise_figure_db, lam)
for _ in range(p.n_spans):
    e = propagate(e, fs, p.span_km, p.attenuation_db_km, p.dispersion_ps_nm_km, lam,
                  p.nonlinear, p.gamma_w_km, p.steps_per_km, p.dispersion_slope_ps_nm2_km,
                  p.nonlinear_phase_rad)
    if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
        e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.span_km), p.pmd_sections)
    e = _add_ase(self.rng, e * np.sqrt(g), s, fs)
    ase = ase + s  # the span loss and the amplifier gain cancel
