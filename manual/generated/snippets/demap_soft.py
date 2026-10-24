k = self.bits_per_symbol
flat = y.reshape(-1)
nv = np.broadcast_to(np.asarray(noise_var, dtype=np.float64), y.shape).reshape(-1)
llr = np.empty((flat.size, k))
zero = self.bit_labels == 0  # (M, k)
for s in range(0, flat.size, _CHUNK):
    metric = -np.abs(flat[s:s + _CHUNK, None] - self.points[None, :]) ** 2
    metric /= np.maximum(nv[s:s + _CHUNK, None], 1e-300)
    for i in range(k):
        m0, m1 = metric[:, zero[:, i]], metric[:, ~zero[:, i]]
        if method == "exact":
            llr[s:s + _CHUNK, i] = logsumexp(m0, axis=1) - logsumexp(m1, axis=1)
        else:
            llr[s:s + _CHUNK, i] = m0.max(axis=1) - m1.max(axis=1)
return llr.reshape(*y.shape[:-1], -1)
