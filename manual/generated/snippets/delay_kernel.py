whole = int(np.floor(delay))
frac = delay - whole
if frac < 1e-9:
    return whole, np.ones(1)
k = np.arange(-half_width + 1, half_width + 1)
taps = np.sinc(k - frac) * (0.5 + 0.5 * np.cos(np.pi * (k - frac) / half_width))
first = whole + k[0]
if first < 0:
    taps, first = taps[-first:], 0
return first, taps / np.linalg.norm(taps)
