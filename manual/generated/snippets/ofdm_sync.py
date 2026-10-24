prod = r[:, :n - N] * np.conj(r[:, N:])  # r[k] r*[k+N]
csum = np.concatenate([np.zeros((B, 1)), np.cumsum(prod, axis=1)], axis=1)
n_sym = (n - N - cp - D) // L + 1
theta = np.arange(D + 1)
gamma = np.zeros((B, D + 1), dtype=np.complex128)
for m in range(max(n_sym, 1)):
    start = theta + m * L
    gamma += csum[:, start + cp] - csum[:, start]
d = np.argmax(np.abs(gamma), axis=1)
eps = -np.angle(gamma[np.arange(B), d]) / (2 * np.pi) if self.params.correct_cfo else np.zeros(B)
backoff = self.params.backoff if self.params.backoff is not None else cp // 4
d = np.maximum(d - backoff, 0)
