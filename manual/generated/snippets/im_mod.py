drive = np.real(x.data)
peak = np.max(np.abs(drive)) or 1.0
drive = np.clip(drive / peak, -1, 1)
delay = x.meta.get("delay", 0.0)
if p.resonance_ghz is not None:
    def h(f: np.ndarray) -> np.ndarray:
        return laser_response(f, p.resonance_ghz * 1e9, p.damping_per_ns * 1e9,
                              p.parasitic_ghz * 1e9 if p.parasitic_ghz else None)
    drive = apply_frequency_response(drive, h, fs, pad=256)
    delay += dc_group_delay(h, fs) * fs
er = 10 ** (p.extinction_ratio_db / 10)
m = (er - 1) / (er + 1)
p_avg = dbm_to_w(p.power_dbm)
power = p_avg * (1 + m * drive)
power = np.maximum(intensity_noise(self.rng, power, p_avg, p.rin_db_hz, fs), 0.0)
phase = _wiener_phase(self.rng, *power.shape, p.linewidth_hz, fs)
if p.chirp_alpha:
    phase = phase + p.chirp_alpha / 2 * np.log(np.maximum(power, 1e-12 * p_avg) / p_avg)
if p.adiabatic_chirp_ghz_mw:
    kappa = p.adiabatic_chirp_ghz_mw * 1e9 / 1e-3  # Hz per W
    phase = phase + 2 * np.pi * kappa * np.cumsum(power - p_avg, axis=-1) / fs
field = np.sqrt(power) * np.exp(1j * phase)
