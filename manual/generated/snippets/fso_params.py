s125 = sigma_r2 ** 1.2  # sigma_R^(12/5)
alpha = 1 / np.expm1(0.49 * sigma_r2 / (1 + 1.11 * s125) ** (7 / 6))
beta = 1 / np.expm1(0.51 * sigma_r2 / (1 + 0.69 * s125) ** (5 / 6))
return float(alpha), float(beta)
