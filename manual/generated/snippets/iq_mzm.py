if p.model == "mzm":
    peak = max(np.max(np.abs(d.real)), np.max(np.abs(d.imag))) or 1.0
    arg = np.pi / 2 * p.drive_swing
    d = (np.sin(arg * d.real / peak) + 1j * np.sin(arg * d.imag / peak))
rms = np.sqrt(np.mean(np.abs(d) ** 2, axis=_pol_axes(d), keepdims=True))
p_tot = dbm_to_w(p.power_dbm)
field = d / np.maximum(rms, 1e-30) * np.sqrt(p_tot / npol)
