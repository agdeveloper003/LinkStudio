if loss_db_m_hz == 0:
    return np.ones_like(f, dtype=np.complex128)
m1, m2 = 4.0, 12.0
c0 = 299_792_458.0
tan_d = loss_db_m_hz * c0 / (20 / np.log(10) * np.pi * np.sqrt(eps_r))
d_eps = tan_d * eps_r * (m2 - m1) * np.log(10) / (np.pi / 2)

def log_term(w):
    return d_eps / (m2 - m1) * np.log10((10 ** m2 + 1j * w) / (10 ** m1 + 1j * w))

eps_inf = eps_r - float(np.real(log_term(2 * np.pi * 1e9)))
w = 2 * np.pi * np.asarray(f, dtype=float)
n = np.sqrt(eps_inf + log_term(w))
return np.exp(-1j * w * length_m * (n - np.sqrt(eps_inf)) / c0)
