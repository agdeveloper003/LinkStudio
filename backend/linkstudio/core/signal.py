"""Signal container passed along the edges of a simulation graph.

Every edge carries a :class:`Signal`: a NumPy array plus a semantic *kind* (bits,
symbols, waveform samples, ...), an optional sample rate and a metadata dictionary
that travels downstream together with the data.

Array convention
----------------
Axis 0 is always the **batch** axis (independent frames simulated in parallel) and
the **last** axis is time / bit index. Most PHY blocks therefore see arrays of shape
``(batch, n)``. Multi-dimensional kinds (e.g. OFDM resource grids) add axes in
between: ``(batch, n_ofdm_symbols, n_fft)``.

Well-known metadata keys
------------------------
Blocks copy the metadata of their input and add/override entries. Keys used by the
built-in PHY library:

``bits_per_symbol``  bits carried by one constellation symbol (set by the mapper)
``code_rate``        cumulative code rate of the chain (set by encoders)
``constellation``    :class:`~linkstudio.phy.dsp.constellations.Constellation` used by the mapper
``n_symbols``        number of symbols per frame (set by mapper / pulse shaper)
``sps``              samples per data symbol of a waveform (used for Es/N0 scaling)
``power``            nominal mean power per sample (the AWGN SNR reference)
``delay``            known nominal processing delay in samples (pulse-shaping filters)
``pulse_taps``       transmit pulse taps (used by the matched filter)
``noise_var``        noise variance per complex sample / symbol (scalar or array)
``ofdm``             :class:`~linkstudio.phy.waveform.ofdm.OFDMLayout` of the waveform
``n_data_symbols``   number of data symbols carried by an OFDM frame
``csi_type``         form of channel state information: "gain", "cir" or "freq"
``unit``             physical unit of the samples ("√W" optical field, "A" photocurrent);
                     absent or None for normalized (dimensionless) signals
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Kind(enum.Flag):
    """Semantic type of the data carried by a :class:`Signal`.

    Kinds are flags so that ports can accept several of them (``Kind.BITS | Kind.LLR``).
    """

    BITS = 1  # hard bits {0, 1}, uint8
    LLR = 2  # log-likelihood ratios, float; convention LLR = log P(b=0) / P(b=1)
    SYMBOLS = 4  # complex baseband symbols at symbol rate
    SAMPLES = 8  # complex baseband waveform samples
    GRID = 16  # frequency-domain resource grid (OFDM)
    CSI = 32  # channel state information (gains, impulse responses, frequency responses)
    FLAGS = 64  # boolean flags, one per frame (e.g. CRC check result)
    ANY = 127

    def label(self) -> str:
        """Human readable name, e.g. ``"BITS|LLR"``."""
        if self is Kind.ANY:
            return "ANY"
        return "|".join(m.name for m in self if m.name) or "NONE"


_KEEP: Any = object()


@dataclass(frozen=True, eq=False)
class Signal:
    """Data flowing between blocks.

    Attributes:
        data: Array with the batch on axis 0 and time on the last axis.
        kind: Semantic kind of ``data``.
        fs: Sample (or symbol) rate in Hz, if meaningful.
        meta: Metadata propagated downstream (see module docstring).
    """

    data: np.ndarray
    kind: Kind
    fs: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def evolve(self, data: Any = _KEEP, *, kind: Any = _KEEP, fs: Any = _KEEP, **meta: Any) -> Signal:
        """Return a new signal derived from this one, merging ``meta`` into the metadata."""
        return Signal(
            data=self.data if data is _KEEP else data,
            kind=self.kind if kind is _KEEP else kind,
            fs=self.fs if fs is _KEEP else fs,
            meta={**self.meta, **meta},
        )

    @property
    def batch_size(self) -> int:
        return int(self.data.shape[0])

    @property
    def length(self) -> int:
        return int(self.data.shape[-1])

    def __repr__(self) -> str:
        fs = f", fs={self.fs:g}" if self.fs else ""
        return f"Signal({self.kind.label()}, shape={self.data.shape}, dtype={self.data.dtype}{fs})"
