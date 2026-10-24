pa, pb = CABLE_PRESETS[preset]
a = pa if a is None else a
b = pb if b is None else b
f_mhz = np.abs(f) / 1e6
km = length_m / 1000
neper = np.log(10) / 20
skin = a * np.sqrt(f_mhz) * km * neper
dielectric = b * f_mhz * km * neper
return np.exp(-skin * (1 + 1j * np.sign(f)) - dielectric)
