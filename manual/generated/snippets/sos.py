def sum_of_sinusoids(rng: np.random.Generator, batch: int, n: int, doppler_norm: float,
                     n_sinusoids: int = 16) -> np.ndarray:
    """Time-varying Rayleigh process with Jakes (Clarke) Doppler spectrum.

    Random-phase sum-of-sinusoids model; ``doppler_norm`` is f_D / f_s.
    """
    t = np.arange(n)
    h = np.zeros((batch, n), dtype=np.complex128)
    alpha = rng.uniform(0, 2 * np.pi, (batch, n_sinusoids))
    phi = rng.uniform(0, 2 * np.pi, (batch, n_sinusoids))
    for i in range(n_sinusoids):
        h += np.exp(1j * (2 * np.pi * doppler_norm * np.cos(alpha[:, i, None]) * t + phi[:, i, None]))
    return h / np.sqrt(n_sinusoids)
