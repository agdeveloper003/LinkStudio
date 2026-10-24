"""Channel coding blocks."""

import numpy as np
import pytest

import linkstudio as cs
from linkstudio.phy.coding.block_codes import HammingCode
from linkstudio.phy.coding.convolutional import Trellis
from helpers import ber_at


def _run_once(graph, batch=4):
    return cs.Simulation(graph, batch_size=batch, seed=0).run(n_batches=1, capture="*")


def test_crc_detects_errors():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=500)
    g.create("enc", "crc_encoder", polynomial="crc24a")
    g.create("dec", "crc_decoder", polynomial="crc24a")
    g.chain("src", "enc", "dec")
    res = _run_once(g, batch=16)
    assert res.captures["enc.out"].length == 524
    assert res.captures["dec.valid"].data.all()
    assert np.array_equal(res.captures["dec.out"].data, res.captures["src.out"].data)
    assert np.isclose(res.captures["enc.out"].meta["code_rate"], 500 / 524)

    from linkstudio.phy.coding.crc import CRC_POLYNOMIALS, crc_parity
    bits = res.captures["enc.out"].data.copy()
    bits[:, 17] ^= 1
    parity = crc_parity(bits[:, :-24], CRC_POLYNOMIALS["crc24a"])
    assert not np.any(np.all(parity == bits[:, -24:], axis=1))


def test_scrambler_is_involution():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=300)
    g.create("scr", "scrambler")
    g.create("des", "scrambler")
    g.chain("src", "scr", "des")
    res = _run_once(g)
    assert not np.array_equal(res.captures["scr.out"].data, res.captures["src.out"].data)
    assert np.array_equal(res.captures["des.out"].data, res.captures["src.out"].data)


@pytest.mark.parametrize("kind", ["block", "random"])
def test_interleaver_inverse(kind):
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=320)
    g.create("il", "interleaver", kind=kind, rows=16)
    g.create("dil", "interleaver", kind=kind, rows=16, inverse=True)
    g.chain("src", "il", "dil")
    res = _run_once(g)
    assert np.array_equal(res.captures["dil.out"].data, res.captures["src.out"].data)


@pytest.mark.parametrize("m", [3, 4, 5])
def test_hamming_corrects_single_errors(m):
    code = HammingCode(m)
    rng = np.random.default_rng(m)
    u = rng.integers(0, 2, (5, code.k * 20), dtype=np.uint8)
    c = code.encode(u)
    assert np.all(((c.reshape(5, -1, code.n).astype(int) @ code.H.T) & 1) == 0)
    err = c.reshape(5, -1, code.n).copy()
    pos = rng.integers(0, code.n, err.shape[:2])
    np.put_along_axis(err, pos[..., None], 1 - np.take_along_axis(err, pos[..., None], -1), -1)
    assert np.array_equal(code.decode(err.reshape(5, -1)), u)


@pytest.mark.parametrize("gens,K", [((7, 5), 3), ((171, 133), 7), ((133, 171, 165), 7)])
def test_viterbi_noiseless(gens, K):
    t = Trellis(gens, K)
    u = np.random.default_rng(K).integers(0, 2, (4, 200), dtype=np.uint8)
    c = t.encode(u)
    assert c.shape[-1] == (200 + K - 1) * len(gens)
    assert np.array_equal(t.decode(1.0 - 2.0 * c), u)


def test_viterbi_corrects_errors():
    t = Trellis((171, 133), 7)
    u = np.random.default_rng(3).integers(0, 2, (8, 500), dtype=np.uint8)
    c = t.encode(u).astype(np.float64)
    flips = np.random.default_rng(4).random(c.shape) < 0.02
    llr = (1 - 2 * c) * np.where(flips, -1, 1)
    assert np.mean(t.decode(llr) != u) < 1e-3


def test_repetition_soft():
    g = cs.Graph()
    g.create("src", "random_bits", n_bits=1000)
    g.create("enc", "repetition_encoder", n=3)
    g.create("map", "mapper", scheme="psk", order=2)
    g.create("awgn", "awgn", snr_db=2.0, signal_power=1.0)
    g.create("demap", "demapper", method="exact")
    g.create("dec", "repetition_decoder", n=3)
    g.create("ber", "ber_meter")
    g.chain("src", "enc", "map", "awgn", "demap", "dec", "ber.rx")
    g.connect("src", "ber.tx")
    # Soft repetition decoding gives no coding gain over uncoded BPSK at equal Eb/N0.
    from linkstudio.phy.theory import ber_bpsk
    assert np.isclose(ber_at(g), ber_bpsk(2.0), rtol=0.2)


def test_conv_code_gain():
    def link(coded: bool) -> cs.Graph:
        g = cs.Graph()
        g.create("src", "random_bits", n_bits=1000)
        g.create("map", "mapper", scheme="psk", order=2)
        g.create("awgn", "awgn", snr_db=4.0, signal_power=1.0)
        g.create("demap", "demapper", method="maxlog")
        g.create("ber", "ber_meter")
        if coded:
            g.create("enc", "conv_encoder")
            g.create("dec", "viterbi_decoder")
            g.chain("src", "enc", "map", "awgn", "demap", "dec", "ber.rx")
        else:
            g.chain("src", "map", "awgn", "demap", "ber.rx")
        g.connect("src", "ber.tx")
        return g

    uncoded = ber_at(link(False))
    coded = cs.Simulation(link(True), batch_size=64, seed=1).run(n_batches=10)["ber.ber"]
    assert uncoded > 1e-2
    assert coded < 1e-4
