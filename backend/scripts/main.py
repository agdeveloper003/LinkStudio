"""LinkStudio PHY engine - guided tour and self-check.

Runs every demo (modulation, pulse shaping, channels, coding, OFDM, scenarios),
prints tables, saves figures to ``backend/output`` and ends with a pass/fail summary
comparing the simulator against communication theory.

    python scripts/main.py             # full run
    python scripts/main.py --quick     # faster, noisier
    python scripts/main.py --only coding ofdm
    python scripts/main.py --show      # also open the figures
"""

from __future__ import annotations

import time

import _common
from _common import DemoContext

import matplotlib.pyplot as plt

import linkstudio as cs
import demo_channels
import demo_coding
import demo_mimo
import demo_modulation
import demo_ofdm
import demo_optical
import demo_pulse_shaping
import demo_scenarios
import demo_sync
import demo_wireline

DEMOS = {
    "modulation": demo_modulation.run,
    "pulse_shaping": demo_pulse_shaping.run,
    "channels": demo_channels.run,
    "coding": demo_coding.run,
    "ofdm": demo_ofdm.run,
    "sync": demo_sync.run,
    "mimo": demo_mimo.run,
    "wireline": demo_wireline.run,
    "optical": demo_optical.run,
    "scenarios": demo_scenarios.run,
}


def quick_start() -> None:
    """The smallest complete example: build, run, read a metric."""
    _common.banner(f"LinkStudio {cs.__version__} - quick start")
    g = cs.Graph("hello-qpsk")
    g.create("src", "random_bits", n_bits=2000)
    g.create("map", "mapper", scheme="psk", order=4)
    g.create("awgn", "awgn", mode="ebn0", snr_db=6.0)
    g.create("demap", "demapper", method="hard")
    g.create("ber", "ber_meter")
    g.chain("src", "map", "awgn", "demap", "ber.rx")
    g.connect("src", "ber.tx")
    print(g.describe())
    print()
    print(cs.Simulation(g, batch_size=64, seed=1).run(n_batches=20).summary())
    print(f"\n{len(cs.available_blocks())} registered blocks: {', '.join(cs.available_blocks())}")


def main() -> None:
    parser = _common.make_parser(__doc__)
    parser.add_argument("--only", nargs="+", choices=list(DEMOS), help="Run only these demos.")
    args = parser.parse_args()

    start = time.perf_counter()
    quick_start()
    checks = []
    for name in args.only or DEMOS:
        ctx = DemoContext(out_dir=args.out, quick=args.quick, show=args.show, demo=name)
        t0 = time.perf_counter()
        DEMOS[name](ctx)
        print(f"\n  ({name} took {time.perf_counter() - t0:.1f} s)")
        checks.extend(ctx.checks)

    failed = _common.print_checks(checks)
    print(f"\nTotal time {time.perf_counter() - start:.1f} s. Figures and CSV files in {args.out}")
    if args.show:
        plt.show()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
