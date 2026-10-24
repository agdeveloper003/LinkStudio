for m in range(1, n_out):
    t = np.clip(start + (m + tau) * sps, 1, L - 3)
    y = cubic_interpolate(data, t)
    mid = cubic_interpolate(data, np.clip(t - sps / 2, 1, L - 3))
    err = np.real(np.conj(mid) * (y - prev))  # > 0 when sampling late
    integ += ki * err
    tau -= kp * err + integ
    out[:, m] = y
    prev = y
