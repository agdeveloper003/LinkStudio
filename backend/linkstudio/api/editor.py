"""Conversion between scenario files / engine graphs and the editor JSON used by the UI.

Editor graph format::

    {
      "name": "qpsk", "seed": 1, "batch_size": 32,
      "blocks": {"awgn": {"type": "awgn", "params": {...all params...}, "position": {"x": 0, "y": 0},
                          "size": {"w": 280, "h": 240},  # optional, resized blocks only
                          "ports": {"in": {"side": "left", "offset": 0.5}}},
                 "tx":   {"type": "subsystem", "spec": {"file": "subsystems/tx.yaml"}, ...}},
      "edges": [{"source": "map.out", "target": "awgn.in"}],
      "probes": [{"key": "awgn.out", "view": "constellation"}],
      "simulation": {...}, "sweep": {...}
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linkstudio.core import Graph, GraphError, graph_from_dict
from linkstudio.core.registry import get_block_class


def editor_to_spec(editor: dict[str, Any]) -> dict[str, Any]:
    """Editor JSON -> scenario dictionary understood by :func:`graph_from_dict`."""
    blocks: dict[str, Any] = {}
    for name, b in (editor.get("blocks") or {}).items():
        if b.get("type") == "subsystem":
            blocks[name] = {"type": "subsystem", **(b.get("spec") or {})}
        else:
            blocks[name] = {"type": b["type"], "params": b.get("params") or {}}
    edges = [f"{e['source']} -> {e['target']}" for e in editor.get("edges") or []]
    return {"name": editor.get("name", "scenario"), "blocks": blocks, "edges": edges}


def build_graph(editor: dict[str, Any], base_dir: Path) -> Graph:
    return graph_from_dict(editor_to_spec(editor), base_dir, name=editor.get("name", "scenario"))


def port_info(graph: Graph, kinds: dict | None = None) -> dict[str, Any]:
    """Ports of every block instance (subsystems have instance-specific ports)."""
    info: dict[str, Any] = {}
    for name, block in graph.blocks.items():
        info[name] = {
            "inputs": [{"name": p, "kinds": port.kinds.label(), "required": port.required}
                       for p, port in block.input_ports.items()],
            "outputs": [{"name": p, "kinds": (kinds or {}).get(f"{name}.{p}", port.kinds).label()
                         if kinds else port.kinds.label(), "same_as": port.same_as}
                        for p, port in block.output_ports.items()],
        }
    return info


def validate_editor(editor: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Build and validate; returns errors instead of raising."""
    try:
        graph = build_graph(editor, base_dir)
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)], "ports": {}, "kinds": {}}
    try:
        resolved = graph.validate()
    except GraphError as exc:
        return {"ok": False, "errors": [str(exc)], "ports": port_info(graph), "kinds": {}}
    kinds = {str(ep): k for ep, k in resolved.items()}
    return {"ok": True, "errors": [], "ports": port_info(graph, kinds),
            "kinds": {k: v.label() for k, v in kinds.items()}}


def _auto_layout(graph: Graph, columns: int = 6) -> dict[str, dict[str, float]]:
    """Layering by topological depth, wrapped in a snake pattern every ``columns`` levels."""
    try:
        order = graph.order()
    except GraphError:
        order = list(graph.blocks)
    depth: dict[str, int] = {}
    for name in order:
        preds = [src.block for dst, src in graph.edges.items() if dst.block == name]
        depth[name] = 1 + max((depth.get(p, 0) for p in preds), default=-1)
    per_depth: dict[int, int] = {}
    for d in depth.values():
        per_depth[d] = per_depth.get(d, 0) + 1
    band_height: dict[int, int] = {}
    for d, count in per_depth.items():
        band_height[d // columns] = max(band_height.get(d // columns, 0), count)
    rows: dict[int, int] = {}
    pos = {}
    for name in order:
        d = depth[name]
        band, col = divmod(d, columns)
        if band % 2:
            col = columns - 1 - col
        r = rows.get(d, 0)
        rows[d] = r + 1
        y0 = sum(band_height[b] for b in range(band)) * 140 + band * 60
        pos[name] = {"x": 40 + col * 240, "y": 40 + y0 + r * 140}
    return pos


def spec_to_editor(spec: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Scenario dictionary (as loaded from YAML) -> editor JSON with full parameters."""
    graph = graph_from_dict(spec, base_dir, name=spec.get("name", "scenario"))
    layout = spec.get("layout") or {}
    port_layout = spec.get("port_layout") or {}
    auto = _auto_layout(graph)
    blocks: dict[str, Any] = {}
    for name, block in graph.blocks.items():
        p = layout.get(name)
        position = {"x": float(p[0]), "y": float(p[1])} if p else auto[name]
        size = {"w": float(p[2]), "h": float(p[3])} if p and len(p) >= 4 else None
        if block.is_composite:
            raw = dict(spec["blocks"][name])
            raw.pop("type", None)
            blocks[name] = {"type": "subsystem", "params": {}, "spec": raw, "position": position}
        else:
            blocks[name] = {"type": block.type_name, "params": block.params.model_dump(mode="json"),
                            "position": position}
        if size:
            blocks[name]["size"] = size
        if port_layout.get(name):
            blocks[name]["ports"] = {port: {"side": side, "offset": float(offset)}
                                     for port, (side, offset) in port_layout[name].items()}
    edges = [{"source": str(src), "target": str(dst)} for dst, src in graph.edges.items()]
    editor = {
        "name": spec.get("name", "scenario"),
        "seed": int(spec.get("seed", 0)),
        "batch_size": int(spec.get("batch_size", 32)),
        "blocks": blocks,
        "edges": edges,
        "probes": spec.get("probes") or [{"key": k} for k in spec.get("capture") or []],
        "simulation": spec.get("simulation") or {},
        "sweep": spec.get("sweep") or None,
        "description": spec.get("description", ""),
        "category": spec.get("category", ""),
    }
    for key in ("inputs", "outputs"):  # external ports of a subsystem file
        if spec.get(key):
            editor[key] = spec[key]
    editor["validation"] = validate_editor(editor, base_dir)
    return editor


def auto_layout_editor(editor: dict[str, Any], base_dir: Path) -> dict[str, dict[str, float]]:
    """Positions for every block of an editor graph (topological snake layout)."""
    return _auto_layout(build_graph(editor, base_dir))


def editor_to_python(editor: dict[str, Any]) -> str:
    """Generate a standalone Python script that rebuilds and runs the editor graph."""
    name = editor.get("name", "scenario")
    lines = [
        f'"""Generated by LinkStudio from scenario {name!r}.',
        "",
        "Run from the backend folder:  python this_script.py",
        '"""',
        "",
        "from pathlib import Path",
        "",
        "import linkstudio as cs",
    ]
    blocks = editor.get("blocks") or {}
    has_sub = any(b.get("type") == "subsystem" for b in blocks.values())
    if has_sub:
        lines += ["from linkstudio.core.scenario import _subsystem_from_spec", "",
                  'SCENARIOS = Path("scenarios")  # subsystem files are resolved from here']
    lines += ["", f"g = cs.Graph({name!r})"]
    for bname, b in blocks.items():
        if b.get("type") == "subsystem":
            lines.append(f"g.add({bname!r}, _subsystem_from_spec({bname!r}, {b.get('spec') or {}!r}, SCENARIOS))")
            continue
        spec = get_block_class(b["type"])(**(b.get("params") or {})).to_spec()
        args = "".join(f", {k}={v!r}" for k, v in (spec.get("params") or {}).items())
        lines.append(f"g.create({bname!r}, {b['type']!r}{args})")
    lines.append("")
    for e in editor.get("edges") or []:
        lines.append(f"g.connect({e['source']!r}, {e['target']!r})")
    ber = next((n for n, b in blocks.items() if b.get("type") == "ber_meter"), None)
    lines += ["", f"sim = cs.Simulation(g, batch_size={int(editor.get('batch_size', 32))}, "
                  f"seed={int(editor.get('seed', 0))})"]
    if ber:
        lines.append(f"result = sim.run(stop=cs.StopCriteria.errors({ber + '.bit_errors'!r}, 200, max_batches=200))")
    else:
        lines.append("result = sim.run(n_batches=10)")
    lines += ["print(result.summary())", ""]
    return "\n".join(lines)


def editor_to_yaml_dict(editor: dict[str, Any]) -> dict[str, Any]:
    """Editor JSON -> scenario dictionary for saving (only non-default parameters)."""
    blocks: dict[str, Any] = {}
    layout: dict[str, list[float]] = {}
    port_layout: dict[str, dict[str, list[Any]]] = {}
    for name, b in (editor.get("blocks") or {}).items():
        if b.get("type") == "subsystem":
            blocks[name] = {"type": "subsystem", **(b.get("spec") or {})}
        else:
            block = get_block_class(b["type"])(**(b.get("params") or {}))
            blocks[name] = block.to_spec()
        pos = b.get("position")
        if pos:
            layout[name] = [round(float(pos["x"]), 1), round(float(pos["y"]), 1)]
            if b.get("size"):  # resized block (scopes, displays): [x, y, w, h]
                layout[name] += [round(float(b["size"]["w"]), 1), round(float(b["size"]["h"]), 1)]
        if b.get("ports"):
            port_layout[name] = {port: [v["side"], round(float(v["offset"]), 3)] for port, v in b["ports"].items()}
    out: dict[str, Any] = {"name": editor.get("name", "scenario")}
    for key in ("category", "description"):
        if editor.get(key):
            out[key] = editor[key]
    out.update(seed=int(editor.get("seed", 0)), batch_size=int(editor.get("batch_size", 32)), blocks=blocks,
               edges=[f"{e['source']} -> {e['target']}" for e in editor.get("edges") or []])
    if editor.get("simulation"):
        out["simulation"] = editor["simulation"]
    if editor.get("sweep"):
        out["sweep"] = editor["sweep"]
    if editor.get("probes"):
        out["probes"] = editor["probes"]
    for key in ("inputs", "outputs"):
        if editor.get(key):
            out[key] = editor[key]
    out["layout"] = layout
    if port_layout:
        out["port_layout"] = port_layout
    return out
