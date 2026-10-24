"""Shared builders for the test-suite."""

from __future__ import annotations

from typing import Any

import linkstudio as cs


def uncoded_link(scheme: str = "psk", order: int = 4, snr_db: float = 6.0, n_bits: int = 2400,
                 mode: str = "ebn0", demap: str = "hard", **awgn: Any) -> cs.Graph:
    """src -> mapper -> AWGN -> demapper -> BER meter."""
    g = cs.Graph("uncoded")
    g.create("src", "random_bits", n_bits=n_bits)
    g.create("map", "mapper", scheme=scheme, order=order)
    g.create("awgn", "awgn", snr_db=snr_db, mode=mode, signal_power=1.0, **awgn)
    g.create("demap", "demapper", method=demap)
    g.create("ber", "ber_meter")
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    return g


def ber_at(graph: cs.Graph, errors: int = 300, batch_size: int = 64, max_batches: int = 400,
           seed: int = 7) -> float:
    stop = cs.StopCriteria.errors("ber.bit_errors", errors, max_batches=max_batches)
    return cs.Simulation(graph, batch_size=batch_size, seed=seed).run(stop=stop)["ber.ber"]
