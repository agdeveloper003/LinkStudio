"""Block base class: the unit of computation of a simulation graph.

A block declares

* ``Params``  - a Pydantic model describing its parameters (validated, serializable,
  and exportable as JSON Schema for a future UI),
* ``inputs`` / ``outputs`` - named, typed :class:`Port` objects,
* ``process`` - the function mapping one batch of input signals to output signals.

Minimal example::

    @register_block("gain", category="channel")
    class Gain(Block):
        class Params(BlockParams):
            gain_db: float = 0.0

        inputs = {"in": Port(Kind.SAMPLES | Kind.SYMBOLS)}
        outputs = {"out": Port.like("in")}

        def process(self, inputs, ctx):
            x = inputs["in"]
            return {"out": x.evolve(x.data * 10 ** (self.params.gain_db / 20))}
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict

from linkstudio.core.context import RunContext
from linkstudio.core.signal import Kind, Signal


class BlockError(RuntimeError):
    """Raised when a block is misconfigured or receives invalid data."""


@dataclass(frozen=True)
class Port:
    """Typed input or output of a block.

    Attributes:
        kinds: Accepted kinds (inputs) or produced kinds (outputs).
        required: Inputs only - whether the port must be connected.
        same_as: Outputs only - the output has the same kind as this input.
        doc: Short description.
    """

    kinds: Kind = Kind.ANY
    required: bool = True
    same_as: str | None = None
    doc: str = ""

    @classmethod
    def like(cls, input_name: str, doc: str = "") -> Port:
        """Output port whose kind mirrors the kind of input ``input_name``."""
        return cls(Kind.ANY, same_as=input_name, doc=doc)


class BlockParams(BaseModel):
    """Base class of every block parameter model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Block:
    """Base class of all simulation blocks."""

    type_name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    is_composite: ClassVar[bool] = False
    Params: ClassVar[type[BlockParams]] = BlockParams
    inputs: ClassVar[Mapping[str, Port]] = {}
    outputs: ClassVar[Mapping[str, Port]] = {}

    def __init__(self, **params: Any) -> None:
        self.name: str = self.type_name or type(self).__name__.lower()
        self.params: Any = self.Params(**params)
        self.rng: np.random.Generator = np.random.default_rng(0)
        self.setup()

    # ------------------------------------------------------------------ ports
    @property
    def input_ports(self) -> Mapping[str, Port]:
        return self.inputs

    @property
    def output_ports(self) -> Mapping[str, Port]:
        return self.outputs

    # -------------------------------------------------------------- lifecycle
    def setup(self) -> None:
        """Derive internal state (filter taps, tables, ...) from the parameters.

        Called at construction and after every parameter change.
        """

    def reset(self, seed: int, path: str) -> None:
        """Re-seed the block random generator and clear runtime state.

        The generator is derived from the master ``seed`` and the block ``path``
        (its hierarchical name), so every block gets an independent but
        reproducible random stream.
        """
        self.rng = np.random.default_rng([seed, zlib.crc32(path.encode())])
        self.__dict__["_recording"] = set()
        try:
            self.on_reset()
        finally:
            self.__dict__["_state_keys"] = tuple(sorted(self.__dict__.pop("_recording")))

    def __setattr__(self, name: str, value: Any) -> None:
        recording = self.__dict__.get("_recording")
        if recording is not None:
            recording.add(name)
        object.__setattr__(self, name, value)

    def on_reset(self) -> None:
        """Clear runtime state such as accumulators or filter memories."""

    # ------------------------------------------------------------- parameters
    def set_param(self, name: str, value: Any) -> None:
        """Change one parameter (validated) and rebuild derived state."""
        data = self.params.model_dump()
        if name not in data:
            raise BlockError(
                f"Block '{self.name}' ({self.type_name}) has no parameter '{name}'. "
                f"Available: {sorted(data)}"
            )
        data[name] = value
        self.params = self.Params.model_validate(data)
        self.setup()

    def get_param(self, name: str) -> Any:
        if name not in type(self.params).model_fields:
            raise BlockError(f"Block '{self.name}' ({self.type_name}) has no parameter '{name}'.")
        return getattr(self.params, name)

    # ------------------------------------------------------------- processing
    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        """Process one batch. Must return a signal for every declared output."""
        raise NotImplementedError

    def metrics(self) -> dict[str, float]:
        """Accumulated measurements (metric blocks override this)."""
        return {}

    #: The frequency response acts on a complex envelope (plot both signs of f).
    two_sided_response: ClassVar[bool] = False
    #: Without a model response, estimate H(f) from the simulated input and output
    #: (meaningful for random linear channels; misleading for noise sources).
    estimate_response: ClassVar[bool] = False

    def frequency_response(self, f: np.ndarray, x: Signal) -> np.ndarray | None:
        """Model transfer function H(f) at frequencies ``f`` (Hz), or None.

        ``x`` is the signal at the block input (sample rate and metadata such as the
        wavelength). Linear blocks override this so the UI can plot their response.
        """
        return None

    # ---------------------------------------------------------- parallel runs
    def export_state(self) -> dict[str, Any]:
        """Accumulators set by :meth:`on_reset`, plus ``_last*`` attributes.

        Parallel runs execute batches in worker processes and merge these states
        into one block with :meth:`merge_state` before reading :meth:`metrics`.
        """
        keys = set(self.__dict__.get("_state_keys", ())) | {k for k in self.__dict__ if k.startswith("_last")}
        return {k: self.__dict__[k] for k in keys if k in self.__dict__}

    def merge_state(self, state: Mapping[str, Any]) -> None:
        """Merge the state exported by another copy of this block.

        Numbers add up and lists are concatenated (counters and per-frame logs);
        ``_last*`` attributes and anything else take the incoming value.
        """
        for key, value in state.items():
            current = self.__dict__.get(key)
            if key.startswith("_last") or current is None:
                object.__setattr__(self, key, value)
            elif isinstance(current, bool) or not isinstance(current, (int, float, list)):
                object.__setattr__(self, key, value)
            elif isinstance(current, list):
                object.__setattr__(self, key, current + list(value))
            else:
                object.__setattr__(self, key, current + value)

    # ---------------------------------------------------------- serialization
    def to_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": self.type_name}
        params = self.params.model_dump(mode="json", exclude_defaults=True)
        if params:
            spec["params"] = params
        return spec

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}'>"
