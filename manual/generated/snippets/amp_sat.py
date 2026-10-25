if p_in_w <= 0:
    return g0
return float(brentq(lambda g: g - g0 * np.exp(-(g - 1) * p_in_w / p_sat_w), 1.0, g0))
