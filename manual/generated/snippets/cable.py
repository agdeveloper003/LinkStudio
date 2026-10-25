pa, pb = CABLE_PRESETS[preset]
a = pa if a is None else a
b = pb if b is None else b
f_mhz = np.abs(f) / 1e6
km = length_m / 1000
neper = np.log(10) / 20
skin = a * np.sqrt(f_mhz) * km * neper
low = LOW_FREQUENCY_TERM.get(preset, 0.0) / np.sqrt(np.maximum(f_mhz, 1.0)) * km * neper
dielectric = dielectric_response(f, b * 1e-9, length_m, DIELECTRIC_CONSTANT.get(preset, 2.3))
return np.exp(-skin * (1 + 1j * np.sign(f)) - low) * dielectric
