s2, a, b = self.model_parameters()
if self.params.model == "lognormal":
    return np.exp(self.rng.normal(-s2 / 2, np.sqrt(s2), n))
return self.rng.gamma(a, 1 / a, n) * self.rng.gamma(b, 1 / b, n)
