if p.initial_estimate and sps >= 3:
    # Oerder & Meyr: phase of the symbol-rate spectral line of |x|^2.
    seg = slice(int(start), int(start) + p.estimate_symbols * sps)
    k = np.arange(seg.start, min(seg.stop, L))
    line = np.sum(np.abs(data[:, k]) ** 2 * np.exp(-2j * np.pi * k / sps), axis=1)
    eps = -np.angle(line) / (2 * np.pi)  # peak position modulo one symbol
    tau = np.mod(eps - start / sps + 0.5, 1.0) - 0.5
