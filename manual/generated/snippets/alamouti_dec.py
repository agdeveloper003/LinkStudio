r1, r2 = y[:, :, 0::2], y[:, :, 1::2]  # (B, Nr, n/2)
h1, h2 = H[:, :, 0:1], H[:, :, 1:2]  # (B, Nr, 1)
norm = np.sum(np.abs(h1) ** 2 + np.abs(h2) ** 2, axis=1)  # (B, 1)
s1 = np.sum(np.conj(h1) * r1 + h2 * np.conj(r2), axis=1) / norm
s2 = np.sum(np.conj(h2) * r1 - h1 * np.conj(r2), axis=1) / norm
