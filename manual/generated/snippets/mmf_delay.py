x = np.asarray(u, dtype=float) ** (alpha / (alpha + 2))
c1 = (alpha - 2 - eps) / (alpha + 2)
c2 = (3 * alpha - 2 - 2 * eps) / (2 * (alpha + 2))
return delta * c1 * x + delta ** 2 * c2 * x ** 2
