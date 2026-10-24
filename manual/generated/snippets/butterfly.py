for k in range(n_sym):
    c = start + k * sps + D  # centre index in the padded signal
    u = pad[:, :, c - D:c + D + 1][:, :, ::-1]  # (B, P, T)
    y = np.einsum("bpqt,bqt->bp", W, u)
    out[:, :, k] = y
    if not use_lms:  # blind: phase-insensitive, leaves the carrier phase to phase_recovery
        e = y * (radius - np.abs(y) ** 2)
    elif k < p.n_train:
        e = refd[:, :, k] - y
    else:
        e = const.points[const.nearest(y)] - y if const is not None else refd[:, :, k] - y
        err += float(np.sum(np.abs(e) ** 2))
        count += e.size
    mu = p.train_step_size if use_lms and k < p.n_train else p.step_size
    W += mu * e[:, :, None, None] * np.conj(u)[:, None, :, :]
