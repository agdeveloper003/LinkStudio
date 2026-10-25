R, Rm = active_radius_um, mode_radius_um
r = np.linspace(0.0, R, 4001)
psi = 2 * (R / Rm) ** 2 * np.exp(-2 * r ** 2 / Rm ** 2)
b = j0(SIGMA1 * r / R)
norm = 2 / R ** 2
jn2 = j0(SIGMA1) ** 2
h0 = norm * np.trapezoid(psi * r, r)
h1 = norm * np.trapezoid(psi * b * r, r)
h3 = norm / jn2 * np.trapezoid(psi * b ** 2 * r, r)
return float(h0), float(h1), float(h1 / jn2), float(h3)
