if linewidth <= 0:
    return np.zeros((batch, n))
return np.cumsum(rng.standard_normal((batch, n)) * np.sqrt(2 * np.pi * linewidth / fs), axis=-1)
