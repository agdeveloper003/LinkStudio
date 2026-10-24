"""Runtime context handed to every block on each batch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunContext:
    """Information about the batch being processed.

    Attributes:
        batch_size: Number of independent frames simulated in parallel.
        batch_index: Index of the current batch since the last reset.
        seed: Master seed of the run.
    """

    batch_size: int
    batch_index: int = 0
    seed: int = 0
