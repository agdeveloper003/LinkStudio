walk = _wiener_phase(self.rng, B, n + 1, p.lo_linewidth_hz, fs)
bridge = walk[:, :n] - np.arange(n) / n * walk[:, n:n + 1]
fo = np.round(p.frequency_offset_hz * n / fs) * fs / n
phase = bridge + 2 * np.pi * fo * np.arange(n) / fs
lo = np.exp(-1j * (phase[:, None, :] if e.ndim == 3 else phase))
p_lo = dbm_to_w(p.lo_power_dbm)
scale = p.responsivity * np.sqrt(p_lo)
bw = fs / 2
var = 2 * (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw
if p.shot_noise:
    var += 2 * Q_E * p.responsivity * p_lo * bw
noise = np.sqrt(var / 2) * (self.rng.standard_normal(e.shape) + 1j * self.rng.standard_normal(e.shape))
out = scale * e * lo + noise
