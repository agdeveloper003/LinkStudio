def rec(z: float, size: int) -> list[float]:
    if size == 1:
        return [z]
    return rec(2 * z - z * z, size // 2) + rec(z * z, size // 2)
return np.array(rec(float(np.exp(-10 ** (design_snr_db / 10))), 1 << n_log2))
