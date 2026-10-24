"""LinkStudio - a modular, block-based communication system simulator.

Quick start::

    import linkstudio as cs

    g = cs.Graph("qpsk")
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("awgn", "awgn", snr_db=6.0)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")

    result = cs.Simulation(g, batch_size=64, seed=1).run(n_batches=20)
    print(result["ber.ber"])
"""

import linkstudio.phy  # noqa: F401  (registers the built-in PHY blocks)
from linkstudio.core import *  # noqa: F401,F403
from linkstudio.core import __all__ as _core_all

__version__ = "0.1.0"
__all__ = list(_core_all) + ["__version__"]
