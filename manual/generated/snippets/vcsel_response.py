h = resonance_hz ** 2 / (resonance_hz ** 2 - f ** 2 + 1j * f * damping_per_s / (2 * np.pi))
if parasitic_hz:
    h = h / (1 + 1j * f / parasitic_hz)
return h
