for n in range(N):
    y = x.data[:, n] * np.exp(-1j * phi)
    d = const.points[const.nearest(y)]
    err = np.imag(y * np.conj(d)) / np.maximum(np.abs(d) ** 2, 1e-12)
    integ += ki * err
    phi += kp * err + integ
    out[:, n] = y
