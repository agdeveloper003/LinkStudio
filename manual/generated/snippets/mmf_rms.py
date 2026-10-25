q = alpha / (alpha + 2)
c1 = (alpha - 2 - eps) / (alpha + 2)
c2 = (3 * alpha - 2 - 2 * eps) / (2 * (alpha + 2))
m = [1 / (k * q + 1) for k in range(5)]
var = (c1 ** 2 * (m[2] - m[1] ** 2) + 2 * delta * c1 * c2 * (m[3] - m[1] * m[2])
       + delta ** 2 * c2 ** 2 * (m[4] - m[2] ** 2))
return delta * float(np.sqrt(var))
