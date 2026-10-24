S = self.n_states
pm = np.full((B, S), -np.inf)
pm[:, 0] = 0.0
decisions = np.empty((T, B, S), dtype=bool)
for t in range(T):
    bm0 = 0.5 * llr[:, t, :] @ self.sign0.T  # (B, S) correlation metrics
    bm1 = 0.5 * llr[:, t, :] @ self.sign1.T
    c0 = pm[:, self.prev0] + bm0
    c1 = pm[:, self.prev1] + bm1
    d = c1 > c0
    decisions[t] = d
    pm = np.where(d, c1, c0)
    pm -= pm.max(axis=1, keepdims=True)
state = np.zeros(B, dtype=np.int64) if terminated else np.argmax(pm, axis=1)
bits = np.empty((B, T), dtype=np.uint8)
rows = np.arange(B)
for t in range(T - 1, -1, -1):
    bits[:, t] = self.input_bit[state]
    state = np.where(decisions[t, rows, state], self.prev1[state], self.prev0[state])
return bits[:, :T - (self.K - 1)] if terminated else bits
