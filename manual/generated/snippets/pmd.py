def apply_pmd(rng: np.random.Generator, e: np.ndarray, fs: float, dgd_s: float) -> np.ndarray:
    """Random polarization rotation plus first-order PMD (differential group delay)."""
    B, _, n = e.shape
    u1, u2 = _random_unitary(rng, B), _random_unitary(rng, B)
    e = np.einsum("bij,bjn->bin", u1, e)
    if dgd_s > 0:
        w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)
        E = np.fft.fft(e, axis=-1)
        E[:, 0] *= np.exp(1j * w * dgd_s / 2)
        E[:, 1] *= np.exp(-1j * w * dgd_s / 2)
        e = np.fft.ifft(E, axis=-1)
    return np.einsum("bij,bjn->bin", u2, e)
