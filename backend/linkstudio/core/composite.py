"""Hierarchical blocks: a whole graph wrapped as a single block.

Subsystems make chains modular: build a ``Transmitter`` or ``Receiver`` once, then
reuse it in many scenarios (and later as a node of a network simulation).

Example::

    tx = Graph("tx")
    tx.create("src", "random_bits", n_bits=1000)
    tx.create("map", "mapper", scheme="qam", order=16)
    tx.chain("src", "map")
    transmitter = Subsystem(tx, outputs={"bits": "src.out", "symbols": "map.out"})
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from linkstudio.core.block import Block, Port
from linkstudio.core.context import RunContext
from linkstudio.core.graph import Endpoint, Graph
from linkstudio.core.signal import Signal


class Subsystem(Block):
    """A graph exposed as a block with user-defined input and output ports.

    Args:
        graph: The inner graph.
        inputs: Mapping ``exposed name -> inner input port`` (``"block.port"``).
        outputs: Mapping ``exposed name -> inner output port``.
    """

    type_name = "subsystem"
    category = "core"
    is_composite = True

    def __init__(self, graph: Graph, inputs: Mapping[str, str] | None = None,
                 outputs: Mapping[str, str] | None = None) -> None:
        self.graph = graph
        self.in_map: dict[str, Endpoint] = {n: graph.resolve(r, output=False) for n, r in (inputs or {}).items()}
        self.out_map: dict[str, Endpoint] = {n: graph.resolve(r, output=True) for n, r in (outputs or {}).items()}
        inner_in = {n: graph.blocks[ep.block].input_ports[ep.port] for n, ep in self.in_map.items()}
        graph.external_inputs = {ep: inner_in[n].kinds for n, ep in self.in_map.items()}
        kinds = graph.validate()
        self._inputs = {n: Port(p.kinds, p.required, doc=f"-> {self.in_map[n]}") for n, p in inner_in.items()}
        self._outputs = {n: Port(kinds[ep], doc=f"<- {ep}") for n, ep in self.out_map.items()}
        self.last_values: dict[str, Signal] = {}
        super().__init__()

    @property
    def input_ports(self) -> Mapping[str, Port]:
        return self._inputs

    @property
    def output_ports(self) -> Mapping[str, Port]:
        return self._outputs

    def reset(self, seed: int, path: str) -> None:
        self.last_values = {}
        self.graph.reset(seed, prefix=path + "/")

    def set_param(self, name: str, value: Any) -> None:
        self.graph.set_param(name, value)

    def get_param(self, name: str) -> Any:
        return self.graph.get_param(name)

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        external = {self.in_map[n]: sig for n, sig in inputs.items()}
        self.last_values = self.graph.execute(ctx, external)
        return {n: self.last_values[str(ep)] for n, ep in self.out_map.items()}

    def metrics(self) -> dict[str, float]:
        return self.graph.metrics()

    def to_spec(self) -> dict[str, Any]:
        from linkstudio.core.scenario import graph_to_dict

        return {
            "type": "subsystem",
            "graph": graph_to_dict(self.graph),
            "inputs": {n: str(ep) for n, ep in self.in_map.items()},
            "outputs": {n: str(ep) for n, ep in self.out_map.items()},
        }
