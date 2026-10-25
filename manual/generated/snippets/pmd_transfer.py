t = np.broadcast_to(_random_unitary(rng, batch)[:, None], (batch, w.size, 2, 2)).copy()
if mean_dgd_s <= 0 or n_sections == 0:
    return t
dtau = mean_dgd_s * np.sqrt(3 * np.pi / (8 * n_sections))
d = np.zeros((w.size, 2, 2), dtype=np.complex128)
d[:, 0, 0] = np.exp(1j * w * dtau / 2)
d[:, 1, 1] = np.exp(-1j * w * dtau / 2)
for _ in range(n_sections):
    u = _random_unitary(rng, batch)
    t = np.einsum("nij,bjk,bnkl->bnil", d, u, t)
return np.einsum("bij,bnjk->bnik", _random_unitary(rng, batch), t)
