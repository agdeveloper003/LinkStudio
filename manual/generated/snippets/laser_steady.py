g = re.g0 / (1 + re.epsilon * s)
k = g * s * re.h2 / (g * s * re.h3 + (1 + re.diffusion) / re.tau_n)  # N1 = k (N0 - N_T)
c = re.h0 - re.h1 * k
n0 = (s / re.tau_p + g * s * c * re.n_tr) / (re.beta / re.tau_n + g * s * c)
return n0, k * (n0 - re.n_tr), c * (n0 - re.n_tr)
