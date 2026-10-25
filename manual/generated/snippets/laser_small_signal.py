n0, n1, s = (v[0] for v in steady_state(re, current))
A = jacobian(re, n0, n1, s)
b = np.array([1 / Q_E, 0.0, 0.0])
w = 2j * np.pi * np.asarray(f, dtype=float)
M = w[:, None, None] * np.eye(3) - A
x = np.linalg.solve(M, np.broadcast_to(b, (w.size, 3))[..., None])[..., 0]
dc = np.linalg.solve(-A, b)
return x[:, 2] / dc[2]
