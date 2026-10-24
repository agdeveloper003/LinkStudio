gram = Hh @ H
reg = n0 if self.params.method == "mmse" else 0.0
inv = np.linalg.inv(gram + reg * np.eye(nt))
s = (inv @ Hh @ yv)[..., 0]  # (B, n, Nt)
if self.params.method == "zf":
    nv = n0 * np.real(np.diagonal(inv, axis1=-2, axis2=-1))
else:
    bias = np.real(np.diagonal(inv @ gram, axis1=-2, axis2=-1))  # (W H)_kk
    s = s / bias
    nv = (1 - bias) / np.maximum(bias, 1e-12)
