p = self.params
if p.alpha is not None:
    return p.alpha
a0 = optimum_alpha(self.delta, p.profile_dispersion)
target = GAUSS_BW_RMS / (self.ofl_bw * 1e6) / (p.group_index / C_LIGHT * 1e3)  # relative RMS over 1 km
return brentq(lambda a: self._rms(self._weights_delays(a, wavelength_nm, "overfilled")) - target,
              a0, a0 + 1.0, xtol=1e-12)
