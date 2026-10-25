s = np.exp(y)
g = g0 / (1.0 + eps * s)
r0 = (n0 - n_tr) * h0 - n1 * h1
dn0 = i / Q_E - n0 / tau_n - g * r0 * s
dn1 = g * ((n0 - n_tr) * h2 - n1 * h3) * s - n1 * (1.0 + hd) / tau_n
dy = beta * n0 / (tau_n * s) + g * r0 - 1.0 / tau_p
dphi = 0.5 * alpha * g0 * (r0 - r_bias)
return dn0, dn1, dy, dphi
