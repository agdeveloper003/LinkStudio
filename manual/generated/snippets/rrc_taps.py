def rrc_taps(sps: int, span: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine pulse, unit energy."""
    t = _time_axis(sps, span)
    b = rolloff
    if b == 0:
        return _unit_energy(np.sinc(t))
    h = np.empty_like(t)
    eps = 1e-9
    at_zero = np.abs(t) < eps
    at_sing = np.abs(np.abs(4 * b * t) - 1) < eps
    rest = ~(at_zero | at_sing)
    tt = t[rest]
    h[rest] = (np.sin(np.pi * tt * (1 - b)) + 4 * b * tt * np.cos(np.pi * tt * (1 + b))) / (
        np.pi * tt * (1 - (4 * b * tt) ** 2))
    h[at_zero] = 1 - b + 4 * b / np.pi
    h[at_sing] = b / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
    return _unit_energy(h)
