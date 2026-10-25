M, F = p.apd_gain, self.excess_noise()
i_primary = self.responsivity_aw(x.meta.get("wavelength_nm", 1550.0)) * power + p.dark_current_na * 1e-9
var = (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw * np.ones_like(power)
if p.shot_noise:
    var = var + 2 * Q_E * M ** 2 * F * i_primary * bw
current = M * i_primary + np.sqrt(var) * self.rng.standard_normal(power.shape)
