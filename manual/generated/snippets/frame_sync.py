if self.params.correct_cfo:
    cand = np.arange(D + 1)[:, None] + np.arange(h)  # (D+1, h)
    P = np.sum(np.conj(data[:, cand]) * data[:, cand + h], axis=-1)  # (B, D+1)
    R = np.sum(np.abs(data[:, cand + h]) ** 2, axis=-1)
    d0 = np.argmax(np.abs(P) ** 2 / np.maximum(R, 1e-12) ** 2, axis=1)
    eps = np.angle(P[b_idx, d0]) / (2 * np.pi * h)  # cycles per symbol
    data = data * np.exp(-2j * np.pi * eps[:, None] * n_idx)

win = np.arange(D + 1)[:, None] + np.arange(L)  # (D+1, L)
corr = np.sum(data[:, win] * np.conj(pre), axis=-1)  # (B, D+1)
d = np.argmax(np.abs(corr), axis=1)

if self.params.correct_cfo:  # refine with the exact preamble position
    first = d[:, None] + np.arange(h)
    fine = np.angle(np.sum(np.conj(data[rows, first]) * data[rows, first + h], axis=-1)) / (2 * np.pi * h)
    data = data * np.exp(-2j * np.pi * fine[:, None] * n_idx)
    eps = eps + fine
pwin = d[:, None] + np.arange(L)
phase = np.angle(np.sum(data[rows, pwin] * np.conj(pre), axis=-1))

idx = d[:, None] + L + np.arange(n_pay)
out = data[rows, idx] * np.exp(-1j * phase)[:, None]
