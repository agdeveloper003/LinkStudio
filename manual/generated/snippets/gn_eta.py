alpha = attenuation_db_km / (20 * np.log10(np.e)) / 1e3  # field, 1/m
gamma = gamma_w_km * 1e-3
b2 = abs(beta2_ps2_km) * 1e-27  # s^2/m
leff = (1 - np.exp(-2 * alpha * span_km * 1e3)) / (2 * alpha)
la = 1 / (2 * alpha)
df = spacing_hz or symbol_rate
arg = np.pi ** 2 / 2 * b2 * la * symbol_rate ** 2 * n_channels ** (2 * symbol_rate / df)
return float(8 / 27 * gamma ** 2 * leff ** 2 * np.arcsinh(arg) / (np.pi * b2 * la * symbol_rate ** 2))
