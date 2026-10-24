for t in range(N + D):
    u = ypad[:, t:t + n][:, ::-1]  # [y[t], y[t-1], ...]
    z = np.sum(np.conj(w) * u, axis=1)
    k = t - D
    if k < 0:
        continue
    out[:, k] = z
    if k < p.n_train:
        target = ref.data[:, k]
    elif p.decision_directed:
        target = const.points[const.nearest(z)]
    else:
        continue
    e = target - z
    if k >= p.n_train:
        err_power += float(np.sum(np.abs(e) ** 2))
        err_count += B
    mu = p.step_size / (1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True)) if p.normalized \
        else p.step_size
    w += mu * u * np.conj(e)[:, None]
