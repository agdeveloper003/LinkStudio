d = 0.0 if extinction_ratio_db is None else 10 ** (-extinction_ratio_db / 20)
theta = np.pi * v / (2 * v_pi)
if push_pull:
    return ((1 + d) / 2 * np.exp(1j * theta) - (1 - d) / 2 * np.exp(-1j * theta)) / 1j
return ((1 + d) / 2 * np.exp(2j * theta) - (1 - d) / 2) / 1j
