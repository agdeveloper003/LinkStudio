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
        self.on_reset()

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

    # ---------------------------------------------------------- serialization
    def to_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": self.type_name}
        params = self.params.model_dump(mode="json", exclude_defaults=True)
        if params:
            spec["params"] = params
        return spec

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}'>"
