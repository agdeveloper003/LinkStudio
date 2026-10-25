if p.hybrid_phase_error_deg:  # the Q output sees the LO at 90 + error degrees
    err = np.deg2rad(p.hybrid_phase_error_deg)
    beat = beat.real + 1j * (beat.imag * np.cos(err) - beat.real * np.sin(err))
sig_power = scale ** 2 * (x.meta.get("power") or 1e-3) / npol
if p.internal_snr_db is not None:  # P_N,int = chi P_sig with chi = 1 / ISNR
    var += sig_power * 10 ** (-p.internal_snr_db / 10)
