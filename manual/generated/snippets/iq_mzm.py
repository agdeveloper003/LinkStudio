if p.model == "mzm":
    peak = max(np.max(np.abs(d.real)), np.max(np.abs(d.imag))) or 1.0
    ei = mzm_field(p.drive_swing * d.real / peak, 1.0, p.extinction_ratio_db)
    eq = mzm_field(p.drive_swing * d.imag / peak, 1.0, p.extinction_ratio_db)
    d = ei + np.exp(1j * np.pi / 2 * (1 + p.quadrature_error_deg / 90)) * eq
elif p.quadrature_error_deg:
    d = d.real + np.exp(1j * np.pi / 2 * (1 + p.quadrature_error_deg / 90)) * d.imag
rms = np.sqrt(np.mean(np.abs(d) ** 2, axis=_pol_axes(d), keepdims=True))
p_tot = dbm_to_w(p.power_dbm)
field = d / np.maximum(rms, 1e-30) * np.sqrt(p_tot / npol)
