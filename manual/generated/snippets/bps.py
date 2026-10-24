if p.method == "viterbi_viterbi":
    s = smooth(y ** 4)
    phi = (np.angle(s) - np.pi) / 4
else:
    if const is None:
        raise BlockError("BPS needs the constellation (mapper upstream).")
    phases = np.arange(p.n_test_phases) / p.n_test_phases * np.pi / 2 - np.pi / 4
    dist = np.empty((*y.shape, phases.size))
    for i, ph in enumerate(phases):
        r = y * np.exp(-1j * ph)
        dist[..., i] = np.abs(r - const.points[const.nearest(r)]) ** 2
    dist = np.moveaxis(smooth(np.moveaxis(dist, -1, 0)), 0, -1)
    phi = phases[np.argmin(dist, axis=-1)]
phi = np.unwrap(4 * phi, axis=-1) / 4
