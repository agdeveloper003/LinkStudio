"""Linear-feedback shift-register sequences (PRBS, scramblers)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Primitive trinomials x^a + x^b + 1 used by standard PRBS generators (ITU-T O.150).
PRBS_TAPS: dict[int, tuple[int, int]] = {
    7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14), 20: (20, 3), 23: (23, 18), 31: (31, 28),
}


def lfsr_sequence(taps: Sequence[int], n: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` bits of ``s[k] = XOR_{t in taps} s[k - t]``.

    Args:
        taps: Exponents of the feedback polynomial (without the constant term),
            e.g. ``(7, 4)`` for ``x^7 + x^4 + 1``.
        n: Number of output bits.
        state: The previous ``max(taps)`` bits, oldest first (default: all ones).

    Returns:
        ``(bits, new_state)`` so that generation can continue seamlessly.
    """
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


def int_to_state(seed: int, deg: int) -> np.ndarray:
    """Register state from an integer seed (MSB = oldest bit)."""
    if seed <= 0 or seed >= 1 << deg:
        raise ValueError(f"Seed must be in [1, 2^{deg} - 1].")
    return ((seed >> np.arange(deg - 1, -1, -1)) & 1).astype(np.uint8)
