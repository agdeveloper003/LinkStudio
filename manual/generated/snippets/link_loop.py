for _ in range(p.n_spans):
    e = propagate(e * np.sqrt(a_pre), fs, p.span_km, att, d, lam, p.nonlinear, gamma, p.steps_per_km,
                  p.dispersion_slope_ps_nm2_km, p.nonlinear_phase_rad, gain=raman)
    if e.ndim == 3 and (p.polarization_rotation or p.pmd_ps_sqrt_km):
        e = apply_pmd(self.rng, e, fs, p.pmd_ps_sqrt_km * 1e-12 * np.sqrt(p.span_km), p.pmd_sections)
    if s_r:
        e = _add_ase(self.rng, e, s_r, fs)
    e = _add_ase(self.rng, e * np.sqrt(g), s, fs)
    ase = (ase * a_pre * a_f * g_r + s_r) * g + s
    power = power * a_pre * a_f * g_r * g
