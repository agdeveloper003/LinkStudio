"""Directed acyclic graph of blocks and its executor.

Naming conventions
------------------
* ``"block.port"`` addresses a port (``"awgn.out"``). Omitting the port
  (``"awgn"``) selects the block's first declared port.
* ``"block.param"`` addresses a parameter (``"awgn.snr_db"``).
* ``"/"`` descends into subsystems: ``"rx/demap.method"``, ``"tx/map.out"``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from linkstudio.core.block import Block, BlockError
from linkstudio.core.context import RunContext
from linkstudio.core.signal import Kind, Signal


class GraphError(ValueError):
    """Raised for invalid graph structures (cycles, unconnected inputs, kind mismatches)."""


@dataclass(frozen=True)
class Endpoint:
    """A port of a named block."""

    block: str
    port: str

    def __str__(self) -> str:
        return f"{self.block}.{self.port}"


class Graph:
    """A set of named blocks connected by edges from output ports to input ports.

    Every input port is driven by at most one output; an output may fan out to any
    number of inputs. The graph must be acyclic (feedback lives inside blocks).
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.blocks: dict[str, Block] = {}
        # Destination input -> source output.
        self.edges: dict[Endpoint, Endpoint] = {}
        # Inputs driven from outside the graph (used by subsystems) and their kinds.
        self.external_inputs: dict[Endpoint, Kind] = {}
        self._order: list[str] | None = None

    # ------------------------------------------------------------ construction
    def add(self, name: str, block: Block) -> Block:
        """Add ``block`` under ``name`` and return it."""
        if not name or any(c in name for c in "./ ->"):
            raise GraphError(f"Invalid block name '{name}': use letters, digits and underscores.")
        if name in self.blocks:
            raise GraphError(f"A block named '{name}' already exists in graph '{self.name}'.")
        block.name = name
        self.blocks[name] = block
        self._order = None
        return block

    def create(self, name: str, type_name: str, **params: Any) -> Block:
        """Create a registered block by type name and add it."""
        from linkstudio.core.registry import create_block

        return self.add(name, create_block(type_name, **params))

    def remove(self, name: str) -> None:
        self._block(name)
        del self.blocks[name]
        self.edges = {d: s for d, s in self.edges.items() if d.block != name and s.block != name}
        self.external_inputs = {e: k for e, k in self.external_inputs.items() if e.block != name}
        self._order = None

    def connect(self, src: str, dst: str) -> None:
        """Connect output ``src`` (``"block.port"``) to input ``dst``."""
        s = self.resolve(src, output=True)
        d = self.resolve(dst, output=False)
        if d in self.edges:
            raise GraphError(f"Input '{d}' is already driven by '{self.edges[d]}'.")
        self.edges[d] = s
        self._order = None

    def chain(self, *refs: str) -> None:
        """Connect consecutive blocks through their default (first) ports.

        ``g.chain("src", "map", "awgn.in")`` is ``connect("src", "map")`` then
        ``connect("map", "awgn.in")``. A reference may name a port explicitly; it is
        then used as output when it is the source and as input when it is the target.
        """
        for a, b in zip(refs, refs[1:]):
            name, _, port = a.partition(".")
            src = a if port in self._block(name).output_ports else name
            self.connect(src, b)

    def resolve(self, ref: str, output: bool) -> Endpoint:
        """Turn ``"block[.port]"`` into an :class:`Endpoint`, checking it exists."""
        name, _, port = ref.partition(".")
        block = self._block(name)
        ports = block.output_ports if output else block.input_ports
        side = "output" if output else "input"
        if not port:
            if not ports:
                raise GraphError(f"Block '{name}' has no {side} ports.")
            port = next(iter(ports))
        if port not in ports:
            raise GraphError(f"Block '{name}' has no {side} port '{port}'. Available: {list(ports)}")
        return Endpoint(name, port)

    def _block(self, name: str) -> Block:
        try:
            return self.blocks[name]
        except KeyError:
            raise GraphError(f"No block named '{name}' in graph '{self.name}'. "
                             f"Blocks: {list(self.blocks)}") from None

    def __getitem__(self, name: str) -> Block:
        return self._block(name)

    def __contains__(self, name: object) -> bool:
        return name in self.blocks

    def __iter__(self) -> Iterator[str]:
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    # -------------------------------------------------------------- parameters
    def set_param(self, path: str, value: Any) -> None:
        """Set a parameter by path, e.g. ``"awgn.snr_db"`` or ``"rx/demap.method"``."""
        head, sep, rest = path.partition("/")
        if sep:
            self._block(head).set_param(rest, value)
            return
        name, _, param = path.partition(".")
        if not param:
            raise GraphError(f"Parameter path '{path}' must look like 'block.param'.")
        self._block(name).set_param(param, value)

    def get_param(self, path: str) -> Any:
        head, sep, rest = path.partition("/")
        if sep:
            return self._block(head).get_param(rest)
        name, _, param = path.partition(".")
        return self._block(name).get_param(param)

    # -------------------------------------------------------------- validation
    def validate(self) -> dict[Endpoint, Kind]:
        """Check the structure and return the resolved kind of every output port.

        Raises:
            GraphError: on unconnected required inputs, cycles or kind mismatches.
        """
        for name, block in self.blocks.items():
            for pname, port in block.input_ports.items():
                ep = Endpoint(name, pname)
                if ep in self.edges and ep in self.external_inputs:
                    raise GraphError(f"Input '{ep}' is driven both internally and externally.")
                if port.required and ep not in self.edges and ep not in self.external_inputs:
                    raise GraphError(f"Required input '{ep}' is not connected.")

        order = self._toposort()
        kinds: dict[Endpoint, Kind] = {}
        for name in order:
            block = self.blocks[name]
            in_kinds: dict[str, Kind] = {}
            for pname, port in block.input_ports.items():
                ep = Endpoint(name, pname)
                if ep in self.edges:
                    src: str = str(self.edges[ep])
                    kind = kinds[self.edges[ep]]
                elif ep in self.external_inputs:
                    src, kind = "external input", self.external_inputs[ep]
                else:
                    continue
                if not (kind & port.kinds):
                    raise GraphError(f"Kind mismatch: '{src}' produces {kind.label()} "
                                     f"but '{ep}' accepts {port.kinds.label()}.")
                in_kinds[pname] = kind & port.kinds
            for pname, port in block.output_ports.items():
                if port.same_as is not None:
                    kind = in_kinds.get(port.same_as, block.input_ports[port.same_as].kinds)
                else:
                    kind = port.kinds
                kinds[Endpoint(name, pname)] = kind
        self._order = order
        return kinds

    def _toposort(self) -> list[str]:
        succ: dict[str, set[str]] = {n: set() for n in self.blocks}
        for dst, src in self.edges.items():
            if dst.block == src.block:
                raise GraphError(f"Block '{dst.block}' is connected to itself.")
            succ[src.block].add(dst.block)
        indeg = {n: 0 for n in self.blocks}
        for targets in succ.values():
            for t in targets:
                indeg[t] += 1
        ready = [n for n in self.blocks if indeg[n] == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for t in sorted(succ[n], key=list(self.blocks).index):
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
        if len(order) != len(self.blocks):
            cyclic = [n for n in self.blocks if n not in order]
            raise GraphError(f"The graph contains a cycle involving: {cyclic}")
        return order

    def order(self) -> list[str]:
        """Execution order (validates the graph if needed)."""
        if self._order is None:
            self.validate()
        assert self._order is not None
        return self._order

    # --------------------------------------------------------------- execution
    def reset(self, seed: int, prefix: str = "") -> None:
        """Re-seed all blocks and clear their runtime state."""
        for name, block in self.blocks.items():
            block.reset(seed, prefix + name)

    def execute(self, ctx: RunContext, external: Mapping[Endpoint, Signal] | None = None) -> dict[str, Signal]:
        """Run one batch through the graph.

        Returns:
            Every produced signal keyed by ``"block.port"`` (subsystem internals
            are included as ``"sub/block.port"``).
        """
        external = external or {}
        produced: dict[Endpoint, Signal] = {}
        values: dict[str, Signal] = {}
        for name in self.order():
            block = self.blocks[name]
            inputs: dict[str, Signal] = {}
            for pname, port in block.input_ports.items():
                ep = Endpoint(name, pname)
                src = self.edges.get(ep)
                if src is not None:
                    sig = produced[src]
                elif ep in external:
                    sig = external[ep]
                else:
                    continue
                if sig.kind not in port.kinds:
                    raise BlockError(f"Input '{ep}' accepts {port.kinds.label()} but received {sig.kind.label()}.")
                inputs[pname] = sig
            try:
                outputs = block.process(inputs, ctx)
            except (BlockError, GraphError):
                raise
            except Exception as exc:  # add context, keep the original traceback
                raise BlockError(f"Block '{name}' ({block.type_name}) failed: {exc}") from exc
            for pname in block.output_ports:
                if pname not in outputs:
                    raise BlockError(f"Block '{name}' did not produce output '{pname}'.")
                produced[Endpoint(name, pname)] = outputs[pname]
                values[f"{name}.{pname}"] = outputs[pname]
            if block.is_composite:
                for key, sig in getattr(block, "last_values", {}).items():
                    values[f"{name}/{key}"] = sig
        return values

    def metrics(self) -> dict[str, float]:
        """Collect the metrics of every block as ``"block.metric"``."""
        out: dict[str, float] = {}
        for name, block in self.blocks.items():
            sep = "/" if block.is_composite else "."
            for key, value in block.metrics().items():
                out[f"{name}{sep}{key}"] = value
        return out

    # ------------------------------------------------------------------- misc
    def describe(self, indent: str = "") -> str:
        """Readable multi-line description of blocks and connections."""
        lines = [f"{indent}Graph '{self.name}' ({len(self.blocks)} blocks)"]
        for name in self.order():
            block = self.blocks[name]
            params = block.params.model_dump(exclude_defaults=True)
            ptxt = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{indent}  [{name}] {block.type_name}({ptxt})")
            for pname in block.input_ports:
                ep = Endpoint(name, pname)
                if ep in self.edges:
                    lines.append(f"{indent}      {pname} <- {self.edges[ep]}")
                elif ep in self.external_inputs:
                    lines.append(f"{indent}      {pname} <- (external)")
            if block.is_composite:
                lines.append(getattr(block, "graph").describe(indent + "      "))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        from linkstudio.core.scenario import graph_to_dict

        return graph_to_dict(self)
