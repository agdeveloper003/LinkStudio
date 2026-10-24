H = np.zeros((B, Lc, n), dtype=np.complex128)
for j in range(n):
    H[:, j:j + L, j] = h
nv = x.meta.get("noise_var", 0.0)
lam = np.mean(nv) if self.params.method == "mmse" else 1e-9
A = np.conj(np.swapaxes(H, 1, 2)) @ H + lam * np.eye(n)
W = np.linalg.solve(A, np.conj(np.swapaxes(H, 1, 2)))  # (B, n, Lc): column d = taps for delay d
if self.params.delay is not None:
    if self.params.delay >= Lc:
        raise BlockError(f"Delay must be < n_taps + L - 1 = {Lc}.")
    d = np.full(B, self.params.delay)
else:
    gain = np.real(np.einsum("bij,bji->bi", H, W))  # (H W)[d, d]
    d = np.argmax(gain, axis=1)
rows = np.arange(B)
w = W[rows, :, d]  # (B, n)
bias = np.einsum("bij,bj->bi", H, w)[rows, d]
w = w / bias[:, None]
full = np.stack([np.convolve(x.data[b], w[b]) for b in range(B)])
idx = d[:, None] + np.arange(x.length)
y = np.take_along_axis(np.pad(full, ((0, 0), (0, Lc))), idx, axis=1)
new_nv = np.asarray(nv) * np.sum(np.abs(w) ** 2, axis=1)[:, None]
