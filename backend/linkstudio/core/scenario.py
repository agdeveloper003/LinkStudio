"""Scenario files: graphs plus run configuration in YAML (or plain dicts).

Format::

    name: qpsk-awgn
    seed: 1
    batch_size: 64
    blocks:
      src:   {type: random_bits, params: {n_bits: 2000}}
      map:   {type: mapper, params: {scheme: psk, order: 4}}
      awgn:  {type: awgn, params: {snr_db: 6}}
      demap: {type: demapper, params: {method: hard}}
      ber:   {type: ber_meter}
      tx:    {type: subsystem, file: tx.yaml}        # reusable subsystem
    chain: [src, map, awgn, demap, ber.rx]            # default-port connections
    edges:                                            # explicit connections
      - src.out -> ber.tx
    simulation: {stop: {metric: ber.bit_errors, threshold: 200, max_batches: 500}}
    sweep: {params: {awgn.snr_db: [0, 2, 4, 6]}}      # optional: run a sweep instead
    capture: [awgn.out]

A subsystem is declared inline (``graph: {blocks: ..., edges: ...}``) or from a file
(``file: path.yaml``) plus ``inputs`` / ``outputs`` port maps, which may also live
in the subsystem file itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from linkstudio.core.composite import Subsystem
from linkstudio.core.graph import Graph, GraphError
from linkstudio.core.registry import create_block
from linkstudio.core.simulation import Simulation, SimulationResult, StopCriteria
from linkstudio.core.sweep import Sweep, SweepResult


def graph_from_dict(spec: Mapping[str, Any], base_dir: Path | None = None, name: str = "graph") -> Graph:
    """Build a :class:`Graph` from a scenario/graph dictionary."""
    graph = Graph(spec.get("name", name))
    for bname, bspec in (spec.get("blocks") or {}).items():
        if isinstance(bspec, str):
            bspec = {"type": bspec}
        btype = bspec.get("type")
        if btype is None:
            raise GraphError(f"Block '{bname}' has no 'type'.")
        if btype == "subsystem":
            block = _subsystem_from_spec(bname, bspec, base_dir)
        else:
            block = create_block(btype, **(bspec.get("params") or {}))
        graph.add(bname, block)

    chains = spec.get("chain") or []
    if chains and all(isinstance(c, str) for c in chains):
        chains = [chains]
    for chain in chains:
        graph.chain(*chain)

    for edge in spec.get("edges") or []:
        if isinstance(edge, str):
            if "->" not in edge:
                raise GraphError(f"Edge '{edge}' must look like 'a.out -> b.in'.")
            src, dst = (s.strip() for s in edge.split("->"))
        else:
            src, dst = edge
        graph.connect(src, dst)
    return graph


def _subsystem_from_spec(name: str, bspec: Mapping[str, Any], base_dir: Path | None) -> Subsystem:
    if "file" in bspec:
        path = Path(bspec["file"])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        inner = yaml.safe_load(path.read_text(encoding="utf-8"))
        inner_dir = path.parent
    elif "graph" in bspec:
        inner, inner_dir = bspec["graph"], base_dir
    else:
        raise GraphError(f"Subsystem '{name}' needs either 'graph' or 'file'.")
    graph = graph_from_dict(inner, inner_dir, name=name)
    inputs = bspec.get("inputs", inner.get("inputs"))
    outputs = bspec.get("outputs", inner.get("outputs"))
    return Subsystem(graph, inputs, outputs)


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    """Serialize a graph (inverse of :func:`graph_from_dict`)."""
    return {
        "name": graph.name,
        "blocks": {name: block.to_spec() for name, block in graph.blocks.items()},
        "edges": [f"{src} -> {dst}" for dst, src in graph.edges.items()],
    }


@dataclass
class Scenario:
    """A graph together with how to run it."""

    graph: Graph
    name: str = "scenario"
    seed: int = 0
    batch_size: int = 32
    simulation: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] | None = None
    capture: list[str] = field(default_factory=list)

    def _stop(self, cfg: Mapping[str, Any]) -> StopCriteria | None:
        stop = cfg.get("stop")
        return StopCriteria(**stop) if stop else None

    def run(self, progress: bool = False) -> SimulationResult | SweepResult:
        """Run the sweep if one is configured, otherwise a single simulation."""
        if self.sweep:
            cfg = dict(self.sweep)
            return Sweep(self.graph, cfg.pop("params"), mode=cfg.pop("mode", "grid"),
                         batch_size=self.batch_size, seed=self.seed, n_batches=cfg.pop("n_batches", None),
                         stop=self._stop(cfg), metrics=cfg.pop("metrics", None)).run(progress=progress)
        sim = Simulation(self.graph, batch_size=self.batch_size, seed=self.seed)
        n_batches = self.simulation.get("n_batches")
        stop = self._stop(self.simulation)
        return sim.run(n_batches=n_batches, stop=stop, capture=self.capture, progress=progress)

    def to_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"name": self.name, "seed": self.seed, "batch_size": self.batch_size}
        spec.update({k: v for k, v in graph_to_dict(self.graph).items() if k != "name"})
        if self.simulation:
            spec["simulation"] = self.simulation
        if self.sweep:
            spec["sweep"] = self.sweep
        if self.capture:
            spec["capture"] = self.capture
        return spec


def scenario_from_dict(spec: Mapping[str, Any], base_dir: Path | None = None) -> Scenario:
    return Scenario(
        graph=graph_from_dict(spec, base_dir, name=spec.get("name", "scenario")),
        name=spec.get("name", "scenario"),
        seed=int(spec.get("seed", 0)),
        batch_size=int(spec.get("batch_size", 32)),
        simulation=dict(spec.get("simulation") or {}),
        sweep=dict(spec["sweep"]) if spec.get("sweep") else None,
        capture=list(spec.get("capture") or []),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario YAML file (relative subsystem files resolve next to it)."""
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return scenario_from_dict(spec, base_dir=path.parent)


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(scenario.to_dict(), sort_keys=False), encoding="utf-8")
