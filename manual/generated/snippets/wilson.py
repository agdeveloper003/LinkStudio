def wilson_interval(errors: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score confidence interval of an error probability."""
    if trials == 0:
        return float("nan"), float("nan")
    p = errors / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))
