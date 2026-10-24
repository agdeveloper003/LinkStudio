n_sym = N // sps + 2
rj = self.rng.standard_normal((B, n_sym)) * self.params.rj_rms_ui
rj_t = np.stack([np.interp(n / sps, np.arange(n_sym), r) for r in rj])
sj = self.params.sj_amplitude_ui * np.sin(2 * np.pi * self.params.sj_frequency_hz * n / fs)
t = n[None, :] + sps * (rj_t + sj[None, :])
