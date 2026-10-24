X = np.zeros((B, n_sym, p.n_fft), dtype=np.complex128)
for b, idx in self.groups.items():
    sel = self.offsets[idx][:, None] + np.arange(b)  # (n_tones_b, b)
    X[..., self.tones[idx]] = self.consts[b].map(bits[..., sel].reshape(B, n_sym, -1))
X[..., p.n_fft - self.tones] = np.conj(X[..., self.tones])
t = np.fft.ifft(X, axis=-1, norm="ortho").real
out = np.concatenate([t[..., p.n_fft - p.cp_len:], t], axis=-1).reshape(B, -1)
