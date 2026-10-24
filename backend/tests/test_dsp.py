"""Pure DSP functions: constellations, filters, LFSR."""

import numpy as np
import pytest

from linkstudio.phy.dsp import constellations as C
from linkstudio.phy.dsp.bits import bits_to_ints, ints_to_bits
from linkstudio.phy.dsp.filters import design_pulse, fir_filter, rrc_taps
from linkstudio.phy.dsp.lfsr import PRBS_TAPS, lfsr_sequence


def test_bits_ints_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (3, 60), dtype=np.uint8)
    assert np.array_equal(ints_to_bits(bits_to_ints(bits, 6), 6), bits)


@pytest.mark.parametrize("const", [C.psk(2), C.psk(4), C.psk(8), C.qam(16), C.qam(64), C.pam(4)])
def test_unit_energy_and_gray(const):
    assert np.isclose(np.mean(np.abs(const.points) ** 2), 1.0)
    # Gray: every point has its nearest neighbour(s) at Hamming distance 1.
    d = np.abs(const.points[:, None] - const.points[None, :])
    np.fill_diagonal(d, np.inf)
    for i in range(const.order):
        nearest = np.isclose(d[i], d[i].min())
        hd = np.sum(const.bit_labels[nearest] != const.bit_labels[i], axis=1)
        assert hd.min() == 1


@pytest.mark.parametrize("const", [C.psk(8), C.qam(16), C.qam(256)])
def test_noiseless_demapping(const):
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (2, const.bits_per_symbol * 100), dtype=np.uint8)
    y = const.map(bits)
    assert np.array_equal(const.demap_hard(y), bits)
    llr = const.demap_soft(y, 0.01, "exact")
    assert np.array_equal((llr < 0).astype(np.uint8), bits)


def test_maxlog_close_to_exact_at_high_snr():
    const = C.qam(16)
    rng = np.random.default_rng(2)
    y = const.map(rng.integers(0, 2, (1, 400), dtype=np.uint8)) + 0.05 * rng.standard_normal((1, 100))
    a = const.demap_soft(y, 0.005, "exact")
    b = const.demap_soft(y, 0.005, "maxlog")
    assert np.allclose(a, b, rtol=0.05, atol=0.5)


def test_rrc_is_nyquist_after_matched_filter():
    sps, span = 8, 12
    h = rrc_taps(sps, span, 0.3)
    assert np.isclose(np.sum(h ** 2), 1.0)
    g = np.convolve(h, h)
    c = len(g) // 2
    samples = g[c % sps::sps]
    peak_idx = c // sps
    isi = np.delete(samples, peak_idx)
    assert np.isclose(samples[peak_idx], 1.0, atol=1e-3)
    assert np.max(np.abs(isi)) < 5e-3


@pytest.mark.parametrize("shape", ["rrc", "rc", "gaussian", "rect"])
def test_pulses_unit_energy(shape):
    assert np.isclose(np.sum(design_pulse(shape, 4, 6) ** 2), 1.0)


def test_fir_filter_batched():
    x = np.random.default_rng(0).standard_normal((3, 50))
    taps = np.array([1.0, 0.5, 0.25])
    y = fir_filter(x, taps)
    assert np.allclose(y[1], np.convolve(x[1], taps))


@pytest.mark.parametrize("order", [7, 9, 11])
def test_prbs_period(order):
    period = 2 ** order - 1
    seq, _ = lfsr_sequence(PRBS_TAPS[order], 2 * period)
    assert np.array_equal(seq[:period], seq[period:])
    assert seq[:period].sum() == 2 ** (order - 1)  # maximal-length balance property


def test_lfsr_continuation():
    full, _ = lfsr_sequence((7, 4), 300)
    a, state = lfsr_sequence((7, 4), 120)
    b, _ = lfsr_sequence((7, 4), 180, state)
    assert np.array_equal(np.concatenate([a, b]), full)
