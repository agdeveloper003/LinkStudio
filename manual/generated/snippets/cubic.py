n = x.shape[-1]
i = np.floor(t).astype(np.int64)
mu = t - i
rows = np.arange(x.shape[0])
xm1, x0, x1, x2 = (x[rows, np.clip(i + d, 0, n - 1)] for d in (-1, 0, 1, 2))
c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
c2 = xm1 / 2 - x0 + x1 / 2
c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
return ((c3 * mu + c2) * mu + c1) * mu + x0
