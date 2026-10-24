if p.model == "lognormal":
    s2 = p.log_intensity_variance
    h = np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), B))
else:
    h = self.rng.gamma(p.alpha, 1 / p.alpha, B) * self.rng.gamma(p.beta, 1 / p.beta, B)
shape = (B,) + (1,) * (x.data.ndim - 1)
gain = np.sqrt(h).reshape(shape)
