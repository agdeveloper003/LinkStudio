def sc_decode(llr: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Successive-cancellation decoding. Returns the estimated u ``(B, N)``."""
    u_hat = np.zeros(llr.shape, dtype=np.uint8)

    def rec(L: np.ndarray, lo: int) -> np.ndarray:
        n = L.shape[-1]
        mask = frozen[lo:lo + n]
        if mask.all():
            return np.zeros(L.shape, dtype=np.uint8)
        if not mask.any():  # rate-1 node: direct hard decision
            x = (L < 0).astype(np.uint8)
            u_hat[:, lo:lo + n] = polar_encode(x)
            return x
        if n == 1:
            x = (L < 0).astype(np.uint8)
            u_hat[:, lo:lo + 1] = x
            return x
        h = n // 2
        L1, L2 = L[:, :h], L[:, h:]
        xa = rec(_f(L1, L2), lo)
        xb = rec(L2 + (1 - 2.0 * xa) * L1, lo + h)
        return np.concatenate([xa ^ xb, xb], axis=1)

    rec(llr, 0)
    return u_hat
