atm = kim_attenuation_db_km(p.visibility_km, lam) * p.distance_m / 1e3 if p.visibility_km else 0.0
loss = 10 ** (-(atm + geometric_loss_db(p.distance_m, p.divergence_mrad, p.rx_aperture_m)) / 10)
h = np.ones(B)
s2r = rytov_variance(p.cn2, lam, p.distance_m)
if p.turbulence != "none" and s2r > 0:
    a, b = gamma_gamma_parameters(s2r)
    if p.turbulence == "gamma_gamma":
        h = self.rng.gamma(a, 1 / a, B) * self.rng.gamma(b, 1 / b, B)
    else:
        s2 = np.log1p(1 / a + 1 / b + 1 / (a * b))
        h = np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), B))
if p.pointing_jitter_m > 0:
    wz = p.distance_m * p.divergence_mrad * 1e-3 / 2
    v = np.sqrt(np.pi / 2) * (p.rx_aperture_m / 2) / wz
    wzeq2 = wz ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2))
    r = p.pointing_jitter_m * np.sqrt(-2 * np.log(1 - self.rng.random(B)))
    hp = np.exp(-2 * r ** 2 / wzeq2)
    self._hp.extend(hp.tolist())
    h = h * hp
shape = (B,) + (1,) * (x.data.ndim - 1)
out = x.data * np.sqrt(loss) * np.sqrt(h).reshape(shape)
