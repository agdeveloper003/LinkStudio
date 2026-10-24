if p.variation == "block":
    h = complex_normal(self.rng, (B, 1))
elif p.variation == "fast":
    h = complex_normal(self.rng, (B, n))
else:
    if not x.fs:
        raise BlockError("Jakes fading needs the sample rate (fs) of the signal.")
    h = sum_of_sinusoids(self.rng, B, n, p.doppler_hz / x.fs, p.n_sinusoids)
if p.model == "rician":
    k = 10 ** (p.k_factor_db / 10)
    los = np.exp(1j * self.rng.uniform(0, 2 * np.pi, (B, 1)))
    h = np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * h
