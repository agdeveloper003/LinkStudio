p_in = float(np.mean(np.abs(x.data) ** 2))
scale = np.sqrt(10 ** (-p.input_backoff_db / 10) / max(p_in, 1e-300))
u = x.data * scale
r, theta = np.abs(u), np.angle(u)
if p.model == "rapp":
    a = r / (1 + r ** (2 * p.smoothness)) ** (1 / (2 * p.smoothness))
    y = a * np.exp(1j * theta)
elif p.model == "saleh":
    # Divided by alpha_a so that the small-signal gain is 1, like the other models.
    a = r / (1 + p.saleh_beta_a * r ** 2)
    dphi = p.saleh_alpha_p * r ** 2 / (1 + p.saleh_beta_p * r ** 2)
    y = a * np.exp(1j * (theta + dphi))
else:
    y = np.minimum(r, 1.0) * np.exp(1j * theta)
