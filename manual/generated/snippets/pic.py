r, ind, g, c = pic_primary_constants(f, preset)
w = 2 * np.pi * np.maximum(np.abs(f), 1.0)  # the table starts at 1 Hz (DC behaves like 1 Hz)
zs, ys = r + 1j * w * ind, g + 1j * w * c
x = np.sqrt(zs * ys) * length_m
z0 = np.sqrt(zs / ys)
z = termination_ohm
h = 2 * z / (2 * np.cosh(x) * z + z0 * np.sinh(x) + np.sinh(x) / z0 * z ** 2)
_, l_hi, _, c_hi = pic_primary_constants(np.array([_PIC_F[-1]]), preset)
h = h * np.exp(1j * w * length_m * np.sqrt(l_hi[0] * c_hi[0]))
return np.where(f < 0, np.conj(h), h)
