for n in range(N):
    u = ref[:, n:n + T][:, ::-1]
    e = y[:, n] - np.sum(w * u, axis=1)
    out[:, n] = e
    mu = self.params.train_step_size if n < self.params.train_samples else self.params.step_size
    w += mu * e[:, None] * np.conj(u) / (1e-9 + np.sum(np.abs(u) ** 2, axis=1, keepdims=True))
