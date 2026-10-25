walk = wiener_phase(self.rng, B, n + 1, p.lo_linewidth_hz, fs)
bridge = walk[:, :n] - np.arange(n) / n * walk[:, n:n + 1]
fo = np.round(p.frequency_offset_hz * n / fs) * fs / n
phase = bridge + 2 * np.pi * fo * np.arange(n) / fs
lo = np.exp(-1j * (phase[:, None, :] if e.ndim == 3 else phase))
npol = e.shape[1] if e.ndim == 3 else 1
p_lo = dbm_to_w(p.lo_power_dbm) / npol  # a polarization splitter feeds one hybrid per polarization
scale = p.responsivity * np.sqrt(p_lo)
bw = fs / 2
var = 2 * (p.thermal_noise_pa_rthz * 1e-12) ** 2 * bw
if p.shot_noise:
    var += 2 * Q_E * p.responsivity * p_lo * bw
beat = scale * e * lo
if p.hybrid_phase_error_deg:  # the Q output sees the LO at 90 + error degrees
    err = np.deg2rad(p.hybrid_phase_error_deg)
    beat = beat.real + 1j * (beat.imag * np.cos(err) - beat.real * np.sin(err))
sig_power = scale ** 2 * (x.meta.get("power") or 1e-3) / npol
if p.internal_snr_db is not None:  # P_N,int = chi P_sig with chi = 1 / ISNR
    var += sig_power * 10 ** (-p.internal_snr_db / 10)
noise = np.sqrt(var / 2) * (self.rng.standard_normal(e.shape) + 1j * self.rng.standard_normal(e.shape))
