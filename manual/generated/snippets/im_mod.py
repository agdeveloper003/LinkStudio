drive = np.real(x.data)
peak = np.max(np.abs(drive)) or 1.0
er = 10 ** (p.extinction_ratio_db / 10)
m = (er - 1) / (er + 1)
p_avg = dbm_to_w(p.power_dbm)
power = p_avg * (1 + m * np.clip(drive / peak, -1, 1))
if p.rin_db_hz is not None:
    power = power * (1 + np.sqrt(10 ** (p.rin_db_hz / 10) * fs / 2) * self.rng.standard_normal(power.shape))
power = np.maximum(power, 0.0)
phase = _wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
if p.chirp_alpha:
    phase = phase + p.chirp_alpha / 2 * np.log(np.maximum(power, 1e-12 * p_avg) / p_avg)
field = np.sqrt(power) * np.exp(1j * phase)
