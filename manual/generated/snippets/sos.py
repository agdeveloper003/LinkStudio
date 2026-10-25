def sum_of_sinusoids(rng: np.random.Generator, batch: int, n: int, doppler_norm: float,
                     n_sinusoids: int = 16) -> np.ndarray:
    """Time-varying Rayleigh process with Jakes (Clarke) Doppler spectrum, E|h|^2 = 1.

    Zheng & Xiao (2003) sum-of-sinusoids generator: M sinusoids with arrival angles
    alpha_n = (2 pi n - pi + theta) / (4 M) and independent random phases. Unlike the
    original Jakes model it is wide-sense stationary, and its autocorrelation approaches
    J0(2 pi f_D tau) per realization with few sinusoids. ``doppler_norm`` is f_D / f_s.
    """
    m = n_sinusoids
    wd = 2 * np.pi * doppler_norm * np.arange(n)
    theta = rng.uniform(-np.pi, np.pi, (batch, 1))
    phi = rng.uniform(-np.pi, np.pi, (batch, 1))
    psi = rng.uniform(-np.pi, np.pi, (batch, m))
    alpha = (2 * np.pi * np.arange(1, m + 1) - np.pi + theta) / (4 * m)  # (batch, m)
    arg = wd[None, None, :] * np.cos(alpha)[:, :, None] + phi[:, :, None]
    xc = np.sum(np.cos(psi)[:, :, None] * np.cos(arg), axis=1)
    xs = np.sum(np.sin(psi)[:, :, None] * np.cos(arg), axis=1)
    return (xc + 1j * xs) * np.sqrt(2 / m)
