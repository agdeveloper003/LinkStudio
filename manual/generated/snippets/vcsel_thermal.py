i = np.atleast_1d(np.asarray(current, dtype=float))
t = np.full(i.shape, th.ambient_c)
for _ in range(iterations):
    il = th.leakage_current(t)
    p = static_power(re, i - il)
    target = th.ambient_c + th.r_th * (th.voltage(i, t) * i - p)
    t_new = 0.5 * t + 0.5 * target
    if np.max(np.abs(t_new - t)) < 1e-6:
        t = t_new
        break
    t = t_new
il = th.leakage_current(t)
return t, il, static_power(re, i - il)
