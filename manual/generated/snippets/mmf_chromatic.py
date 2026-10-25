d1 = s0 * wavelength_nm / 4 * (1 - (lambda0_nm / wavelength_nm) ** 4)
d2 = 0.7 * s0 * rms_width_nm
return rms_width_nm * float(np.hypot(d1, d2))
