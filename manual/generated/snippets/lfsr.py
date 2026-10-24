deg, step = max(taps), min(taps)
s = np.empty(deg + n, dtype=np.uint8)
s[:deg] = 1 if state is None else state
pos = deg
# Every output depends on bits at least `step` positions back, so chunks of
# `step` bits can be produced with vector operations.
while pos < deg + n:
    m = min(step, deg + n - pos)
    acc = np.zeros(m, dtype=np.uint8)
    for t in taps:
        acc ^= s[pos - t:pos - t + m]
    s[pos:pos + m] = acc
    pos += m
return s[deg:].copy(), s[n:].copy()
