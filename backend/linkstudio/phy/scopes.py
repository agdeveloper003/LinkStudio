"""Scope (display) blocks.

A scope is a sink with up to four inputs ``in1`` .. ``in4``: each taps the output
feeding it and the UI shows one of them (``channel``). It does not change the
simulation; the plot data are computed by the API from the tapped signals (see
:func:`linkstudio.api.probes.scope_probes`).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from linkstudio.core import Block, BlockParams, Kind, Port, RunContext, Signal, register_block

N_INPUTS = 4


def _inputs(kinds: Kind) -> dict[str, Port]:
    return {f"in{k}": Port(kinds, required=False, doc=f"Tapped signal {k}.") for k in range(1, N_INPUTS + 1)}


class Scope(Block):
    """Base class of the scope blocks; ``view`` selects the plot."""

    view: ClassVar[str] = ""

    class Params(BlockParams):
        channel: int = Field(1, ge=1, le=N_INPUTS, description="Input shown on the screen.")

    inputs = _inputs(Kind.ANY)
    outputs = {}

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        return {}


@register_block("scope_time", category="measurement", subcategory="Scopes", label="Oscilloscope")
class TimeScope(Scope):
    """Waveform versus time (I and Q for complex signals) of the first frame."""

    view = "time"
    inputs = _inputs(Kind.BITS | Kind.LLR | Kind.SYMBOLS | Kind.SAMPLES)


@register_block("scope_spectrum", category="measurement", subcategory="Scopes", label="Spectrum Analyzer")
class SpectrumScope(Scope):
    """Power spectral density (Welch) in physical units when the sample rate is known."""

    view = "spectrum"
    inputs = _inputs(Kind.SYMBOLS | Kind.SAMPLES)


@register_block("scope_constellation", category="measurement", subcategory="Scopes", label="Constellation Diagram")
class ConstellationScope(Scope):
    """I/Q scatter plot, with the reference constellation when it is known."""

    view = "constellation"
    inputs = _inputs(Kind.SYMBOLS | Kind.SAMPLES | Kind.GRID)


@register_block("scope_eye", category="measurement", subcategory="Scopes", label="Eye Diagram")
class EyeScope(Scope):
    """Overlaid two-symbol traces of an oversampled waveform."""

    view = "eye"
    inputs = _inputs(Kind.SAMPLES)


@register_block("scope_histogram", category="measurement", subcategory="Scopes", label="LLR Histogram")
class HistogramScope(Scope):
    """Histogram of the log-likelihood ratios."""

    view = "histogram"
    inputs = _inputs(Kind.LLR)


@register_block("scope_grid", category="measurement", subcategory="Scopes", label="Resource Grid Viewer")
class GridScope(Scope):
    """Magnitude of the OFDM resource grid (subcarrier x symbol, dB)."""

    view = "grid"
    inputs = _inputs(Kind.GRID)


@register_block("scope_csi", category="measurement", subcategory="Scopes", label="Channel Response Viewer")
class CSIScope(Scope):
    """Channel gains, impulse or frequency response carried by a CSI signal."""

    view = "csi"
    inputs = _inputs(Kind.CSI)
