er = 10 ** (extinction_ratio_db / 10)
dp = 2 * np.asarray(p_avg_w, dtype=float) * (er - 1) / ((er + 1) * (order - 1))
return (order - 1) / order * erfc(resp * dp / (2 * np.sqrt(2) * sigma))
