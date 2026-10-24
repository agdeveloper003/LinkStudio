def loop_gains(bandwidth: float, damping: float, detector_gain: float = 1.0) -> tuple[float, float]:
    """Proportional and integral gains of a second-order loop.

    ``bandwidth`` is the noise bandwidth normalized to the update rate.
    """
    theta = bandwidth / (damping + 1 / (4 * damping))
    denom = 1 + 2 * damping * theta + theta ** 2
    return 4 * damping * theta / denom / detector_gain, 4 * theta ** 2 / denom / detector_gain
