"""Shared helpers for the demo scripts: import path, command-line flags, checks and figures.

Every demo module exposes ``run(ctx: DemoContext)`` so it can be executed on its own
(``python scripts/demo_coding.py``) or as part of ``python scripts/main.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import matplotlib  # noqa: E402

if "--show" not in sys.argv:
    matplotlib.use("Agg")  # render to files only, no windows

import matplotlib.pyplot as plt  # noqa: E402

SCENARIO_DIR = BACKEND_DIR / "scenarios"
DEFAULT_OUT = BACKEND_DIR / "output"


@dataclass
class Check:
    demo: str
    name: str
    passed: bool
    detail: str


@dataclass
class DemoContext:
    """State shared by the demos of one run."""

    out_dir: Path = DEFAULT_OUT
    quick: bool = False
    show: bool = False
    demo: str = ""
    checks: list[Check] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record and print a pass/fail check."""
        self.checks.append(Check(self.demo, name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        return bool(passed)

    def save(self, fig: plt.Figure, name: str) -> Path:
        """Save a figure as ``<out_dir>/<demo>_<name>.png``."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.demo}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        if not self.show:
            plt.close(fig)
        print(f"  figure -> {path.relative_to(BACKEND_DIR) if path.is_relative_to(BACKEND_DIR) else path}")
        return path


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def section(title: str) -> None:
    print(f"\n--- {title}")


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="Fewer points / errors (faster, noisier).")
    parser.add_argument("--show", action="store_true", help="Open the figures in windows at the end.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder for figures and CSV files.")
    return parser


def print_checks(checks: list[Check]) -> int:
    """Print a summary table and return the number of failed checks."""
    failed = [c for c in checks if not c.passed]
    banner(f"Summary: {len(checks) - len(failed)}/{len(checks)} checks passed")
    width = max((len(f"{c.demo}: {c.name}") for c in checks), default=10)
    for c in checks:
        label = f"{c.demo}: {c.name}"
        print(f"  {'PASS' if c.passed else 'FAIL'}  {label:<{width}}  {c.detail}")
    return len(failed)


def run_standalone(name: str, run: Callable[[DemoContext], None], description: str) -> None:
    """Entry point used by every demo's ``if __name__ == '__main__'`` block."""
    args = make_parser(description).parse_args()
    ctx = DemoContext(out_dir=args.out, quick=args.quick, show=args.show, demo=name)
    start = time.perf_counter()
    run(ctx)
    failed = print_checks(ctx.checks)
    print(f"\nDone in {time.perf_counter() - start:.1f} s. Figures in {ctx.out_dir}")
    if ctx.show:
        plt.show()
    raise SystemExit(1 if failed else 0)
