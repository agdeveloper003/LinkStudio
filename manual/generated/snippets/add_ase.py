def _add_ase(rng: np.random.Generator, e: np.ndarray, psd: float, fs: float) -> np.ndarray:
    sigma = np.sqrt(psd * fs / 2)
    return e + sigma * (rng.standard_normal(e.shape) + 1j * rng.standard_normal(e.shape))
