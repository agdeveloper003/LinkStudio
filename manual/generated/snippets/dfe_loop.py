for n in range(N):
    u = pad[:, n:n + F][:, ::-1]
    z = np.sum(w * u, axis=1) - np.sum(bfb * past, axis=1)
    out[:, n] = z
    d = ref.data[:, n] if n < p.n_train else const.points[const.nearest(z)]
    e = d - z
    if n >= p.n_train:
        err += float(np.sum(np.abs(e) ** 2))
        count += B
    norm = 1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True) + np.sum(np.abs(past) ** 2, axis=1, keepdims=True)
    w += p.step_size * e[:, None] * np.conj(u) / norm
    if K:
        bfb -= p.step_size * e[:, None] * np.conj(past) / norm
        past = np.concatenate([d[:, None], past[:, :-1]], axis=1)
