current = p.bias_ma * 1e-3 + p.modulation_ma * 1e-3 * drive
if p.parasitic_ghz:
    current = apply_frequency_response(current, self.parasitic, fs, pad=256)
leak, temp = self.operating_point()
preroll = int(round(3 * self.re.tau_n * fs))
power, phase = simulate(self.re, np.maximum(current - leak, 0.0), fs, preroll=preroll)
p_avg = float(np.mean(power))
power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, fs), 0.0)
phase = phase + wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
