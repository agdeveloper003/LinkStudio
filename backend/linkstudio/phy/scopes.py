"""Scope (display) block.

A scope is a sink with up to four inputs ``in1`` .. ``in4``: each taps the output
feeding it and the UI shows one of them (``channel``). The plot (``view``) is chosen
among the ones that suit the displayed signal; ``auto`` picks the natural one (a
constellation for symbols, a spectrum for samples, a histogram for LLRs...). The
scope does not change the simulation; the plot data are computed by the API from the
tapped signals (see :func:`linkstudio.api.probes.scope_probes`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block

N_INPUTS = 4
View = Literal["auto", "time", "spectrum", "constellation", "eye", "histogram", "grid", "csi"]


@register_block("scope", category="measurement", label="Scope")
class Scope(Block):
    """Plots a tapped signal; the available views depend on its kind.

    Bits and LLRs: time (and LLR histogram). Symbols: constellation, time, spectrum.
    Samples: spectrum, constellation, eye diagram, time. Resource grids: constellation
    and grid. Channel state: channel response.
    """

    class Params(BlockParams):
        channel: int = Field(1, ge=1, le=N_INPUTS, description="Input shown on the screen.")
        view: View = Field("auto", description="Plot of the shown signal; 'auto' picks the one that suits its kind.")

    inputs = {
        f"in{k}": Port(Kind.BITS | Kind.LLR | Kind.SYMBOLS | Kind.SAMPLES | Kind.GRID | Kind.CSI,
                       required=False, doc=f"Tapped signal {k}.")
        for k in range(1, N_INPUTS + 1)
    }
    outputs = {}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        return {}
