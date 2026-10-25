g = max(1, int(round(np.sqrt(mode_count(self.radius_um, self.na, wavelength_nm, alpha)))))
mu = np.arange(1, g + 1)
delays = group_delay(((mu - 0.5) / g) ** 2, alpha, self.delta, p.profile_dispersion)
if launch == "overfilled":
    w = (2 * mu - 1).astype(float)  # group mu holds 2 mu - 1 of the g^2 modes
elif launch == "gaussian":
    w0 = np.sqrt(self.radius_um * 1e-6 * wavelength_nm * 1e-9 / (np.pi * self.na)) * 1e6
    w = gaussian_launch_weights(g, p.launch_waist_um, p.launch_offset_um, w0)
else:
    if not p.group_weights:
        raise BlockError("Custom launch needs group_weights.")
    w = np.zeros(g)
    k = min(g, len(p.group_weights))
    w[:k] = np.asarray(p.group_weights[:k], dtype=float)
return w, delays
