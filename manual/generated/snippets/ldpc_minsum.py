if method == "min_sum":
    sign = np.where(V < 0, -1.0, 1.0)
    sign[:, ~valid] = 1.0
    mag = np.abs(V)
    mag[:, ~valid] = np.inf
    i1 = np.argmin(mag, axis=-1)[..., None]
    min1 = np.take_along_axis(mag, i1, -1)
    np.put_along_axis(mag, i1, np.inf, -1)
    min2 = mag.min(axis=-1, keepdims=True)
    pos = np.arange(V.shape[-1])
    out_mag = np.where(pos == i1, min2, min1) * alpha
    out = np.prod(sign, axis=-1, keepdims=True) * sign * out_mag
