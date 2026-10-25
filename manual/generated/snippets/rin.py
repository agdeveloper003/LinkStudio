if rin_db_hz is None:
    return power
return power + p_avg * np.sqrt(10 ** (rin_db_hz / 10) * fs / 2) * rng.standard_normal(power.shape)
