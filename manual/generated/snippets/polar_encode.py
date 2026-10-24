x = u.copy()
n = x.shape[-1]
half = 1
while half < n:
    x = x.reshape(*x.shape[:-1], -1, 2, half)
    x[..., 0, :] ^= x[..., 1, :]
    x = x.reshape(*x.shape[:-3], n)
    half *= 2
return x
