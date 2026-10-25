alpha = att_db_km / (10 * np.log10(np.e)) / 1e3
z = np.linspace(0.0, self.length, n_points)
net = np.array([self.integral(zi, self.length) for zi in z]) - alpha * (self.length - z)
nu = C_LIGHT / (wavelength_nm * 1e-9)
return float(RAMAN_NSP * H_PLANCK * nu * np.trapezoid(self.coefficient(z) * np.exp(net), z))
