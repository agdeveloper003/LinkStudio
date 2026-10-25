import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ViewportPortal,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type FinalConnectionState,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type MouseEvent as ReactMouseEvent } from "react";
import { api, downloadText, type Catalogue, type PortPlace, type Probe, type ScenarioItem, type Validation } from "./api";
import { BlockNode } from "./components/BlockNode";
import { Inspector } from "./components/Inspector";
import { LivePanel } from "./components/LivePanel";
import { DRAG_MIME, Palette } from "./components/Palette";
import { ObstacleContext, RoutedEdge } from "./components/RoutedEdge";
import { ScenarioBrowser } from "./components/ScenarioBrowser";
import { SimulatePanel } from "./components/SimulatePanel";
import { SweepPanel } from "./components/SweepPanel";
import { BlockActionsContext, CatalogueContext, EMPTY_RUN, PortMoveContext, RunDataContext, type RunData } from "./context";
import {
  applyValidation,
  defaultParams,
  edgeId,
  fromEditorGraph,
  numericParams,
  portsFromCatalogue,
  toEditorGraph,
  uniqueName,
  type BlockNodeType,
} from "./graph";
import { nodeRect, snapPosition, spread, straighten, type Guide } from "./snap";
import { isScopeType } from "./geometry";
import { kindColor, kindsCompatible } from "./theme";
import { useLive } from "./useLive";

const nodeTypes = { block: BlockNode };
const edgeTypes = { routed: RoutedEdge };
const DEFAULT_SCENARIO = "wireless/single_carrier_16qam.yaml";
const HISTORY_LIMIT = 100;
type Mode = "live" | "simulate" | "sweep";

interface Meta {
  name: string;
  description: string;
  category: string;
  path: string;
  seed: number;
  batch_size: number;
}

interface Snapshot {
  nodes: BlockNodeType[];
  edges: Edge[];
  probes: Probe[];
}

const EMPTY_META: Meta = { name: "untitled", description: "", category: "", path: "", seed: 1, batch_size: 32 };

/** Side panels: visibility and width, remembered in the browser. */
interface PanelLayout {
  left: boolean;
  right: boolean;
  leftWidth: number;
  rightWidth: number;
}

const DEFAULT_PANELS: PanelLayout = { left: true, right: true, leftWidth: 230, rightWidth: 320 };
const PANELS_KEY = "linkstudio.panels";

function loadPanels(): PanelLayout {
  try {
    return { ...DEFAULT_PANELS, ...JSON.parse(window.localStorage.getItem(PANELS_KEY) ?? "{}") };
  } catch {
    return DEFAULT_PANELS;
  }
}

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

export default function App() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [nodes, setNodes] = useState<BlockNodeType[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [meta, setMeta] = useState<Meta>(EMPTY_META);
  const [probes, setProbes] = useState<Probe[]>([]);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("live");
  const [panels, setPanels] = useState<PanelLayout>(loadPanels);
  const [runData, setRunData] = useState<RunData>(EMPTY_RUN);
  const [toast, setToast] = useState<{ text: string; error?: boolean } | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [historySize, setHistorySize] = useState({ undo: 0, redo: 0 });
  const [guides, setGuides] = useState<Guide[]>([]);
  const [maxWorkers, setMaxWorkers] = useState(1);
  const [workers, setWorkersState] = useState(() => {
    try {
      return Number(window.localStorage.getItem("linkstudio.workers")) || 0;
    } catch {
      return 0;
    }
  });
  const flow = useReactFlow();
  const live = useLive();
  const resizing = useRef<"left" | "right" | null>(null);
  const pendingTap = useRef<{ scope: string; edge: Edge } | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const clipboard = useRef<Snapshot | null>(null);
  const undoStack = useRef<Snapshot[]>([]);
  const redoStack = useRef<Snapshot[]>([]);
  const lastSnapshot = useRef<Snapshot | null>(null);
  const restoring = useRef(false);

  const notify = useCallback((text: string, error = false) => {
    setToast({ text, error });
    window.setTimeout(() => setToast(null), 3500);
  }, []);

  const resetHistory = useCallback((snap: Snapshot | null) => {
    undoStack.current = [];
    redoStack.current = [];
    lastSnapshot.current = snap;
    setHistorySize({ undo: 0, redo: 0 });
  }, []);

  // ------------------------------------------------------------ loading
  const loadScenario = useCallback(
    async (path: string, cat: Catalogue | null = catalogue) => {
      if (!cat) return;
      try {
        const g = await api.loadScenario(path);
        const { nodes: n, edges: e } = fromEditorGraph(g, cat);
        live.stop();
        restoring.current = true;
        setNodes(n);
        setEdges(e);
        setProbes(g.probes ?? []);
        resetHistory({ nodes: n, edges: e, probes: g.probes ?? [] });
        setValidation(g.validation ?? null);
        setMeta({
          name: g.name, description: g.description ?? "", category: g.category ?? "", path: g.path ?? path,
          seed: g.seed, batch_size: g.batch_size,
        });
        setSelected(null);
        setBrowserOpen(false);
        window.setTimeout(() => flow.fitView({ padding: 0.15, duration: 300 }), 50);
      } catch (err) {
        notify(`Cannot load ${path}: ${err instanceof Error ? err.message : err}`, true);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [catalogue, flow, notify, resetHistory],
  );

  useEffect(() => {
    (async () => {
      try {
        const [cat, list, health] = await Promise.all([api.catalogue(), api.scenarios(), api.health()]);
        setMaxWorkers(health.cpu_count);
        setWorkersState((w) => Math.min(w || Math.max(1, health.cpu_count - 1), health.cpu_count));
        setCatalogue(cat);
        setScenarios(list);
        const first = list.find((s) => s.path === DEFAULT_SCENARIO) ?? list.find((s) => !s.subsystem);
        if (first) await loadScenario(first.path, cat);
      } catch (err) {
        setServerError(`Cannot reach the LinkStudio server (${err instanceof Error ? err.message : err}). Start it with: python -m linkstudio serve`);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getGraph = useCallback(() => toEditorGraph(nodes, edges, { ...meta, probes }), [nodes, edges, meta, probes]);

  // ---------------------------------------------------------- validation
  const structureKey = useMemo(
    () => JSON.stringify([nodes.map((n) => [n.id, n.data.type, n.data.params, n.data.spec]), edges.map((e) => e.id)]),
    [nodes, edges],
  );
  const topologyKey = useMemo(
    () => JSON.stringify([nodes.map((n) => [n.id, n.data.type, n.data.spec]), edges.map((e) => e.id)]),
    [nodes, edges],
  );

  useEffect(() => {
    if (!catalogue || !nodes.length) {
      setValidation(null);
      return;
    }
    const t = window.setTimeout(async () => {
      try {
        const v = await api.validate(getGraph());
        setValidation(v);
        setNodes((nds) => applyValidation(nds, v));
      } catch {
        /* server unreachable: keep the last result */
      }
    }, 350);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structureKey, catalogue]);

  // ------------------------------------------------------------- history
  const positionsKey = useMemo(
    () => nodes.map((n) => `${n.id}:${Math.round(n.position.x)},${Math.round(n.position.y)},${n.data.size ? `${Math.round(n.data.size.w)}x${Math.round(n.data.size.h)}` : ""}`).join("|"),
    [nodes],
  );
  const probesKey = useMemo(() => JSON.stringify(probes), [probes]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      const current: Snapshot = { nodes, edges, probes };
      if (restoring.current) {
        restoring.current = false;
        lastSnapshot.current = current;
        return;
      }
      if (lastSnapshot.current) {
        undoStack.current = [...undoStack.current.slice(-HISTORY_LIMIT + 1), lastSnapshot.current];
        redoStack.current = [];
        setHistorySize({ undo: undoStack.current.length, redo: 0 });
      }
      lastSnapshot.current = current;
    }, 400);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structureKey, positionsKey, probesKey]);

  const restore = (snap: Snapshot) => {
    restoring.current = true;
    setNodes(snap.nodes.map((n) => ({ ...n, selected: false })));
    setEdges(snap.edges);
    setProbes(snap.probes);
    if (selected && !snap.nodes.some((n) => n.id === selected)) setSelected(null);
  };

  const undo = () => {
    const prev = undoStack.current.pop();
    if (!prev) return;
    redoStack.current.push({ nodes, edges, probes });
    restore(prev);
    setHistorySize({ undo: undoStack.current.length, redo: redoStack.current.length });
  };

  const redo = () => {
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push({ nodes, edges, probes });
    restore(next);
    setHistorySize({ undo: undoStack.current.length, redo: redoStack.current.length });
  };

  // -------------------------------------------------------- graph editing
  const snapped = useRef<{ id: string; pos: { x: number; y: number } } | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  const onNodesChange = useCallback((changes: NodeChange<BlockNodeType>[]) => {
    // A single block being dragged snaps to its connected ports and to neighbouring blocks.
    const drags = changes.filter((c) => c.type === "position" && c.dragging && c.position);
    if (drags.length === 1 && drags[0].type === "position" && drags[0].position) {
      const c = drags[0];
      const node = nodesRef.current.find((n) => n.id === c.id);
      if (node) {
        const snap = snapPosition(node, c.position!, nodesRef.current, edgesRef.current);
        c.position = snap.pos;
        snapped.current = { id: c.id, pos: snap.pos };
        setGuides(snap.guides);
      }
    } else if (changes.some((c) => c.type === "position" && !c.dragging)) {
      // The drag-end change carries the unsnapped position: keep the snapped one.
      for (const c of changes) {
        if (c.type === "position" && !c.dragging && c.position && snapped.current?.id === c.id) c.position = snapped.current.pos;
      }
      snapped.current = null;
      setGuides([]);
    }
    setNodes((nds) => applyNodeChanges(changes, nds));
    for (const c of changes) if (c.type === "remove") setProbes((ps) => ps.filter((p) => !p.key.startsWith(`${c.id}.`)));
  }, []);

  const movePort = useCallback((node: string, key: string, place: PortPlace) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== node) return n;
        const old = n.data.ports?.[key];
        if (old && old.side === place.side && old.offset === place.offset) return n;
        return { ...n, data: { ...n.data, ports: { ...n.data.ports, [key]: place } } };
      }),
    );
  }, []);

  const obstacles = useMemo(
    () => nodes.map((n) => nodeRect(n)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [positionsKey, structureKey, JSON.stringify(nodes.map((n) => n.data.ports ?? null))],
  );

  const onEdgesChange = useCallback((changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);

  const outputKinds = useCallback(
    (node: string, port: string) => {
      const resolved = validation?.kinds[`${node}.${port}`];
      if (resolved) return resolved;
      const n = nodes.find((x) => x.id === node);
      return n?.data.outputs.find((o) => o.name === port)?.kinds ?? "ANY";
    },
    [nodes, validation],
  );

  const isValidConnection = useCallback(
    (c: Connection | Edge) => {
      if (!c.source || !c.target || c.source === c.target) return false;
      if (edges.some((e) => e.target === c.target && e.targetHandle === c.targetHandle)) return false;
      const target = nodes.find((n) => n.id === c.target);
      const accepted = target?.data.inputs.find((i) => i.name === c.targetHandle)?.kinds ?? "ANY";
      return kindsCompatible(outputKinds(c.source, c.sourceHandle ?? ""), accepted);
    },
    [edges, nodes, outputKinds],
  );

  const onConnect = useCallback((c: Connection) => {
    if (!c.sourceHandle || !c.targetHandle) return;
    setEdges((eds) => [
      ...eds,
      { id: edgeId(c.source, c.sourceHandle!, c.target, c.targetHandle!), source: c.source, sourceHandle: c.sourceHandle, target: c.target, targetHandle: c.targetHandle },
    ]);
  }, []);

  const setParam = useCallback(
    (node: string, param: string, value: unknown) => {
      setNodes((nds) => nds.map((n) => (n.id === node ? { ...n, data: { ...n.data, params: { ...n.data.params, [param]: value } } } : n)));
      if (live.running) live.setParam(`${node}.${param}`, value);
    },
    [live],
  );

  // ------------------------------------------------------------ scope taps
  /**
   * The link under a screen point, ignoring links into scopes. Several links can be under
   * the pointer (their click zones are wider than the lines, or they overlap): the nearest
   * one carrying a signal of the ``accepted`` kinds wins, else the nearest one.
   */
  const edgeAt = useCallback((x: number, y: number, accepted = "ANY"): Edge | undefined => {
    let best: { edge: Edge; d: number } | undefined;
    for (const el of document.elementsFromPoint(x, y)) {
      const g = el.closest(".react-flow__edge") as SVGGElement | null;
      const edge = g?.dataset.id ? edgesRef.current.find((e) => e.id === g.dataset.id) : undefined;
      const target = edge && nodesRef.current.find((n) => n.id === edge.target);
      if (!edge || !target || isScopeType(target.data.type)) continue;
      const path = g!.querySelector<SVGPathElement>("path.react-flow__edge-path");
      const ctm = path?.getScreenCTM();
      if (!path || !ctm) continue;
      let d = Infinity;
      const length = path.getTotalLength();
      for (let s = 0; s <= length; s += 3) {
        const p = new DOMPoint(path.getPointAtLength(s).x, path.getPointAtLength(s).y).matrixTransform(ctm);
        d = Math.min(d, Math.hypot(p.x - x, p.y - y));
      }
      if (!kindsCompatible(outputKinds(edge.source, edge.sourceHandle ?? ""), accepted)) d += 1000;
      if (!best || d < best.d) best = { edge, d };
    }
    return best?.edge;
  }, [outputKinds]);

  /**
   * Connect a scope input to the output feeding ``edge``: the scope taps that link.
   * Without ``port``, the first free input is used. The screen switches to the new signal.
   */
  const tapLink = useCallback(
    (scope: string, edge: Edge, port?: string) => {
      const node = nodesRef.current.find((n) => n.id === scope);
      if (!node) return;
      const used = new Set(edgesRef.current.filter((e) => e.target === scope).map((e) => e.targetHandle));
      const input = port ?? node.data.inputs.find((i) => !used.has(i.name))?.name;
      if (!input) {
        notify("All the scope inputs are in use: drag one of them onto the link instead.", true);
        return;
      }
      const accepted = node.data.inputs.find((i) => i.name === input)?.kinds ?? "ANY";
      const kinds = outputKinds(edge.source, edge.sourceHandle ?? "");
      if (!kindsCompatible(kinds, accepted)) {
        notify(`This scope shows ${accepted} signals; the link carries ${kinds}.`, true);
        return;
      }
      const sourceHandle = edge.sourceHandle ?? "";
      const others = edgesRef.current.filter((e) => !(e.target === scope && e.targetHandle === input));
      setEdges([...others, { id: edgeId(edge.source, sourceHandle, scope, input), source: edge.source, sourceHandle, target: scope, targetHandle: input }]);
      setParam(scope, "channel", node.data.inputs.findIndex((i) => i.name === input) + 1);
    },
    [outputKinds, notify, setParam],
  );

  /** Dragging from a scope input and releasing on a link connects the scope to that link. */
  const onConnectEnd = useCallback(
    (e: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      if (state.isValid || !state.fromHandle || state.fromHandle.type !== "target") return;
      const scope = state.fromHandle.nodeId;
      if (!isScopeType(nodesRef.current.find((n) => n.id === scope)?.data.type ?? "")) return;
      const pt = "changedTouches" in e ? e.changedTouches[0] : e;
      const accepted = nodesRef.current.find((n) => n.id === scope)?.data.inputs.find((i) => i.name === state.fromHandle!.id)?.kinds;
      const edge = edgeAt(pt.clientX, pt.clientY, accepted);
      if (edge) tapLink(scope, edge, state.fromHandle.id ?? undefined);
    },
    [edgeAt, tapLink],
  );

  const addBlock = useCallback(
    (type: string, position?: { x: number; y: number }) => {
      if (!catalogue) return;
      const info = catalogue.blocks[type];
      setNodes((nds) => {
        const name = uniqueName(type, new Set(nds.map((n) => n.id)));
        let pos = position;
        if (!pos) {
          // Centre of the visible canvas, moving right while it overlaps existing blocks.
          const rect = canvasRef.current?.getBoundingClientRect();
          const centre = flow.screenToFlowPosition({
            x: rect ? rect.left + rect.width / 2 - 90 : window.innerWidth / 2,
            y: rect ? rect.top + rect.height / 3 : window.innerHeight / 3,
          });
          pos = { ...centre };
          while (nds.some((n) => Math.abs(n.position.x - pos!.x) < 210 && Math.abs(n.position.y - pos!.y) < 110)) {
            pos = { x: pos.x + 230, y: pos.y };
          }
        }
        const ports = portsFromCatalogue(info);
        setSelected(name);
        return [
          ...nds.map((n) => ({ ...n, selected: false })),
          { id: name, type: "block", position: pos, selected: true, data: { type, params: defaultParams(info.params), ...ports, probed: [] } },
        ];
      });
    },
    [catalogue, flow],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData(DRAG_MIME);
      if (!type) return;
      // A scope dropped on a link is connected to it, and placed below the drop point.
      const accepted = Object.values(catalogue?.blocks[type]?.inputs ?? {})[0]?.kinds;
      const edge = isScopeType(type) ? edgeAt(e.clientX, e.clientY, accepted) : undefined;
      const name = uniqueName(type, new Set(nodesRef.current.map((n) => n.id)));
      const offset = edge ? { x: -40, y: 40 } : { x: -90, y: -20 };
      addBlock(type, flow.screenToFlowPosition({ x: e.clientX + offset.x, y: e.clientY + offset.y }));
      if (edge) pendingTap.current = { scope: name, edge };
    },
    [addBlock, flow, edgeAt, catalogue],
  );

  const resizeBlock = useCallback((node: string, size: { w: number; h: number }) => {
    setNodes((nds) => nds.map((n) => (n.id === node ? { ...n, data: { ...n.data, size } } : n)));
  }, []);
  const runValue = useMemo(() => ({ ...runData, running: live.running }), [runData, live.running]);
  const blockActions = useMemo(() => ({ setParam, resize: resizeBlock }), [setParam, resizeBlock]);

  const renameBlock = useCallback(
    (oldName: string, newName: string): string | null => {
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(newName)) return "Use letters, digits and underscores.";
      if (nodes.some((n) => n.id === newName)) return "Name already used.";
      const swap = (id: string) => (id === oldName ? newName : id);
      setNodes((nds) => nds.map((n) => (n.id === oldName ? { ...n, id: newName } : n)));
      setEdges((eds) =>
        eds.map((e) => {
          const source = swap(e.source);
          const target = swap(e.target);
          return { ...e, source, target, id: edgeId(source, e.sourceHandle!, target, e.targetHandle!) };
        }),
      );
      setProbes((ps) => ps.map((p) => (p.key.startsWith(`${oldName}.`) ? { ...p, key: newName + p.key.slice(oldName.length) } : p)));
      setSelected(newName);
      return null;
    },
    [nodes],
  );

  const updateProbes = useCallback(
    (next: Probe[]) => {
      setProbes(next);
      setNodes((nds) =>
        nds.map((n) => {
          const probed = next.filter((p) => p.key.startsWith(`${n.id}.`)).map((p) => p.key.slice(n.id.length + 1));
          return probed.join() === n.data.probed.join() ? n : { ...n, data: { ...n.data, probed } };
        }),
      );
    },
    [],
  );

  const deleteBlock = (name: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== name));
    setEdges((eds) => eds.filter((e) => e.source !== name && e.target !== name));
    updateProbes(probes.filter((p) => !p.key.startsWith(`${name}.`)));
    setSelected(null);
  };

  // ---------------------------------------------------- copy / paste
  const selectionSnapshot = (ids?: string[]): Snapshot | null => {
    const set = new Set(ids ?? nodes.filter((n) => n.selected).map((n) => n.id));
    if (!set.size && selected) set.add(selected);
    if (!set.size) return null;
    return {
      nodes: nodes.filter((n) => set.has(n.id)),
      edges: edges.filter((e) => set.has(e.source) && set.has(e.target)),
      probes: probes.filter((p) => set.has(p.key.split(".")[0])),
    };
  };

  const pasteSnapshot = (snap: Snapshot) => {
    const taken = new Set(nodes.map((n) => n.id));
    const rename: Record<string, string> = {};
    for (const n of snap.nodes) {
      rename[n.id] = uniqueName(n.id.replace(/_\d+$/, ""), taken);
      taken.add(rename[n.id]);
    }
    const newNodes = snap.nodes.map((n) => ({
      ...n,
      id: rename[n.id],
      selected: true,
      position: { x: n.position.x + 40, y: n.position.y + 60 },
      data: { ...n.data, probed: [] },
    }));
    const newEdges = snap.edges.map((e) => ({
      ...e,
      source: rename[e.source],
      target: rename[e.target],
      id: edgeId(rename[e.source], e.sourceHandle!, rename[e.target], e.targetHandle!),
    }));
    setNodes((nds) => [...nds.map((n) => ({ ...n, selected: false })), ...newNodes]);
    setEdges((eds) => [...eds, ...newEdges]);
    if (newNodes.length === 1) setSelected(newNodes[0].id);
    notify(`Pasted ${newNodes.length} block${newNodes.length > 1 ? "s" : ""}`);
  };

  const duplicate = (ids?: string[]) => {
    const snap = selectionSnapshot(ids);
    if (snap) pasteSnapshot(snap);
  };

  // ------------------------------------------------------------ toolbar
  const saveAs = async (forcePrompt = true) => {
    let path = meta.path;
    if (forcePrompt || !path || path.startsWith("subsystems/")) {
      const answer = window.prompt("Save scenario as (path inside backend/scenarios):", path || `${meta.name.replace(/\W+/g, "_")}.yaml`);
      if (!answer) return;
      path = answer;
    }
    try {
      const res = await api.saveScenario(path, getGraph());
      setMeta((m) => ({ ...m, path: res.path }));
      setScenarios(await api.scenarios());
      notify(`Saved ${res.path}`);
    } catch (err) {
      notify(`Save failed: ${err instanceof Error ? err.message : err}`, true);
    }
  };

  const newScenario = () => {
    live.stop();
    restoring.current = true;
    setNodes([]);
    setEdges([]);
    setProbes([]);
    resetHistory({ nodes: [], edges: [], probes: [] });
    setMeta(EMPTY_META);
    setSelected(null);
    setBrowserOpen(false);
    flow.setViewport({ x: 0, y: 0, zoom: 1 });
  };

  const autoLayout = async () => {
    if (!nodes.length) return;
    try {
      const { positions } = await api.layout(getGraph());
      setNodes((nds) => spread(straighten(nds.map((n) => (positions[n.id] ? { ...n, position: positions[n.id] } : n)), edges)));
      window.setTimeout(() => flow.fitView({ padding: 0.15, duration: 300 }), 50);
    } catch (err) {
      notify(`Layout failed: ${err instanceof Error ? err.message : err}`, true);
    }
  };

  const exportPython = async () => {
    try {
      const { code } = await api.exportPython(getGraph());
      downloadText(`${meta.name.replace(/\W+/g, "_") || "scenario"}.py`, code, "text/x-python");
    } catch (err) {
      notify(`Export failed: ${err instanceof Error ? err.message : err}`, true);
    }
  };

  // --------------------------------------------- live: restart on topology change
  const liveTopology = useRef(topologyKey);
  useEffect(() => {
    if (!live.running) {
      liveTopology.current = topologyKey;
      return;
    }
    if (liveTopology.current === topologyKey) return;
    const t = window.setTimeout(() => {
      liveTopology.current = topologyKey;
      if (validation?.ok !== false) live.start(getGraph());
    }, 700);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topologyKey, live.running]);

  // ---------------------------------------------------------- shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const key = e.key.toLowerCase();
      if (key === "s") {
        e.preventDefault();
        void saveAs(e.shiftKey);
        return;
      }
      if (key === "o") {
        e.preventDefault();
        setBrowserOpen(true);
        return;
      }
      if (isTyping(e)) return;
      if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (key === "y" || (key === "z" && e.shiftKey)) {
        e.preventDefault();
        redo();
      } else if (key === "c") {
        const snap = selectionSnapshot();
        if (snap) {
          clipboard.current = snap;
          notify(`Copied ${snap.nodes.length} block${snap.nodes.length > 1 ? "s" : ""}`);
        }
      } else if (key === "v" && clipboard.current) {
        e.preventDefault();
        pasteSnapshot(clipboard.current);
      } else if (key === "d") {
        e.preventDefault();
        duplicate();
      } else if (key === "l") {
        e.preventDefault();
        void autoLayout();
      } else if (key === "b") {
        e.preventDefault();
        togglePanel("left");
      } else if (key === "i") {
        e.preventDefault();
        togglePanel("right");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // A scope dropped on a link is connected once its node exists.
  useEffect(() => {
    const p = pendingTap.current;
    if (p && nodes.some((n) => n.id === p.scope)) {
      pendingTap.current = null;
      tapLink(p.scope, p.edge);
    }
  }, [nodes, tapLink]);

  // ---------------------------------------------------------- scope data
  useEffect(() => {
    const f = live.frame;
    if (f) setRunData({ probes: f.probes ?? {}, metrics: f.metrics ?? {}, displays: f.displays ?? {}, running: true });
  }, [live.frame]);
  const onSimulationResult = useCallback(
    (probes: RunData["probes"], metrics: RunData["metrics"]) => setRunData({ probes, metrics, displays: {}, running: false }),
    [],
  );

  // --------------------------------------------------- side panel layout
  useEffect(() => {
    try {
      window.localStorage.setItem(PANELS_KEY, JSON.stringify(panels));
    } catch {
      /* private mode: not remembered */
    }
  }, [panels]);

  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (resizing.current === "left") {
        setPanels((p) => ({ ...p, leftWidth: Math.min(Math.max(e.clientX, 160), window.innerWidth / 2) }));
      } else if (resizing.current === "right") {
        setPanels((p) => ({ ...p, rightWidth: Math.min(Math.max(window.innerWidth - e.clientX, 240), 720) }));
      }
    };
    const up = () => {
      resizing.current = null;
      document.body.classList.remove("col-resizing");
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, []);

  const startResize = (side: "left" | "right") => (e: ReactMouseEvent) => {
    e.preventDefault();
    resizing.current = side;
    document.body.classList.add("col-resizing");
  };
  const togglePanel = (side: "left" | "right") => setPanels((p) => ({ ...p, [side]: !p[side] }));

  const scopeNames = useMemo(() => new Set(nodes.filter((n) => isScopeType(n.data.type)).map((n) => n.id)), [nodes]);
  const styledEdges = useMemo(
    () =>
      edges.map((e) => {
        const color = kindColor(outputKinds(e.source, e.sourceHandle ?? ""));
        const tap = scopeNames.has(e.target);
        return { ...e, type: "routed", data: { tap }, style: { stroke: color, strokeWidth: e.selected ? 3 : 2 }, animated: live.running && !tap };
      }),
    [edges, outputKinds, live.running, scopeNames],
  );

  const resources = {
    batch: meta.batch_size,
    workers: Math.max(1, workers),
    maxWorkers,
    setBatch: (n: number) => setMeta((m) => ({ ...m, batch_size: n })),
    setWorkers: (n: number) => {
      setWorkersState(n);
      try {
        window.localStorage.setItem("linkstudio.workers", String(n));
      } catch {
        /* private mode: not remembered */
      }
    },
  };

  const selectedNode = nodes.find((n) => n.id === selected) ?? null;
  const blocked = !nodes.length
    ? "The scenario is empty."
    : validation && !validation.ok
      ? `Fix the graph first: ${validation.errors[0]}`
      : undefined;
  const sweepParams = useMemo(() => (catalogue ? numericParams(nodes, catalogue) : []), [nodes, catalogue]);
  const categories = useMemo(() => Array.from(new Set(scenarios.map((s) => s.category))), [scenarios]);

  if (serverError) return <div className="fatal">{serverError}</div>;
  if (!catalogue) return <div className="fatal">Loading LinkStudio…</div>;

  return (
    <CatalogueContext.Provider value={catalogue}>
    <PortMoveContext.Provider value={movePort}>
    <ObstacleContext.Provider value={obstacles}>
    <RunDataContext.Provider value={runValue}>
    <BlockActionsContext.Provider value={blockActions}>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <svg viewBox="0 0 32 32" width="22" height="22"><path d="M2 16 Q 6 4 10 16 T 18 16 T 26 16 T 30 16" fill="none" stroke="#6ea8fe" strokeWidth="3" /></svg>
            <span className="brand-name">LinkStudio</span>
          </div>
          <button className="scenario-button" onClick={() => setBrowserOpen(true)} title="Open a scenario (Ctrl+O)">
            <span className="scenario-name">{meta.name || "untitled"}</span>
            <span className="muted small">{meta.path || "unsaved"}</span>
            <span className="chev">▾</span>
          </button>
          <div className="toolbar">
            <button onClick={newScenario} title="New blank scenario">New</button>
            <button onClick={() => saveAs(false)} title="Save (Ctrl+S)">Save</button>
            <button onClick={() => saveAs(true)} title="Save as (Ctrl+Shift+S)">Save as</button>
            <span className="sep" />
            <button onClick={undo} disabled={!historySize.undo} title="Undo (Ctrl+Z)">↶</button>
            <button onClick={redo} disabled={!historySize.redo} title="Redo (Ctrl+Y)">↷</button>
            <button onClick={autoLayout} title="Arrange blocks automatically (Ctrl+L)">Layout</button>
            <button onClick={exportPython} title="Download a Python script that rebuilds and runs this scenario">Export .py</button>
            <span className="sep" />
            <button className={`icon-toggle${panels.left ? " active" : ""}`} onClick={() => togglePanel("left")} title="Show / hide the blocks panel (Ctrl+B)">
              <PanelIcon side="left" />
            </button>
            <button className={`icon-toggle${panels.right ? " active" : ""}`} onClick={() => togglePanel("right")} title="Show / hide the inspector (Ctrl+I)">
              <PanelIcon side="right" />
            </button>
          </div>
          <div className="spacer" />
          <label className="inline">
            seed
            <input type="number" value={meta.seed} onChange={(e) => setMeta({ ...meta, seed: Number(e.target.value) })} />
          </label>
          <div className="mode-switch">
            {(["live", "simulate", "sweep"] as Mode[]).map((m) => (
              <button key={m} className={mode === m ? "active" : ""} onClick={() => setMode(m)}>
                {m === "live" ? "Live" : m === "simulate" ? "Simulate" : "Sweep"}
                {m === "live" && live.running && <span className="live-dot" />}
              </button>
            ))}
          </div>
          <div className="run-bar">
            <LivePanel live={live} onStart={() => live.start(getGraph())} blocked={blocked} active={mode === "live"} />
            <SimulatePanel getGraph={getGraph} blocked={blocked} resources={resources} active={mode === "simulate"} onResult={onSimulationResult} />
            <SweepPanel getGraph={getGraph} numericParams={sweepParams} theory={catalogue.theory} scenarioName={meta.name} blocked={blocked} resources={resources} active={mode === "sweep"} />
          </div>
        </header>

        <div
          className="main"
          style={{
            gridTemplateColumns: [
              panels.left ? `min(${panels.leftWidth}px, 50vw) 0px` : "",
              "minmax(0, 1fr)",
              panels.right ? `0px ${panels.rightWidth}px` : "",
            ].join(" "),
          }}
        >
          {panels.left && (
            <>
              <Palette catalogue={catalogue} onAdd={(t) => addBlock(t)} onClose={() => togglePanel("left")} />
              <div className="col-resizer" onMouseDown={startResize("left")} title="Drag to resize" />
            </>
          )}
          <div className="canvas" ref={canvasRef} onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
            <ReactFlow<BlockNodeType>
              nodes={nodes}
              edges={styledEdges}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onConnectEnd={onConnectEnd}
              isValidConnection={isValidConnection}
              onNodeClick={(_, n) => setSelected(n.id)}
              onPaneClick={() => setSelected(null)}
              deleteKeyCode={["Delete", "Backspace"]}
              multiSelectionKeyCode={["Shift", "Meta", "Control"]}
              fitView
              minZoom={0.2}
              proOptions={{ hideAttribution: true }}
              defaultEdgeOptions={{ type: "routed" }}
              onNodeDragStop={() => setGuides([])}
            >
              <ViewportPortal>
                {guides.map((g, k) => (
                  <div
                    key={k}
                    className="align-guide"
                    style={{ left: g.x1, top: g.y1, width: Math.max(g.x2 - g.x1, 1), height: Math.max(g.y2 - g.y1, 1) }}
                  />
                ))}
              </ViewportPortal>
              <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#2a3040" />
              <Controls />
              <MiniMap pannable zoomable nodeColor="#2b3245" maskColor="rgba(10,12,18,0.6)" style={{ width: 150, height: 96 }} />
            </ReactFlow>
            {!nodes.length && (
              <div className="canvas-hint">
                <div>Drag blocks from the library, or</div>
                <button className="primary" onClick={() => setBrowserOpen(true)}>Open a ready-made scenario</button>
              </div>
            )}
          </div>
          {panels.right && (
            <>
              <div className="col-resizer" onMouseDown={startResize("right")} title="Drag to resize" />
              <Inspector
            catalogue={catalogue}
            node={selectedNode}
            validation={validation}
            scenario={{ name: meta.name, description: meta.description, category: meta.category, categories, blocks: nodes.length, edges: edges.length }}
            onParam={setParam}
            onRename={renameBlock}
            onDelete={deleteBlock}
            onDuplicate={(id) => duplicate([id])}
            onScenarioChange={(p) => setMeta({ ...meta, ...p })}
            onClose={() => togglePanel("right")}
            getGraph={getGraph}
            structureKey={structureKey}
              />
            </>
          )}
        </div>

        {browserOpen && (
          <ScenarioBrowser
            scenarios={scenarios}
            current={meta.path}
            onOpen={(p) => loadScenario(p)}
            onBlank={newScenario}
            onClose={() => setBrowserOpen(false)}
          />
        )}
        {toast && <div className={`toast${toast.error ? " error" : ""}`}>{toast.text}</div>}
      </div>
    </BlockActionsContext.Provider>
    </RunDataContext.Provider>
    </ObstacleContext.Provider>
    </PortMoveContext.Provider>
    </CatalogueContext.Provider>
  );
}

function PanelIcon({ side }: { side: "left" | "right" }) {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1.5" y="2.5" width="13" height="11" rx="1.5" />
      <rect x={side === "left" ? 1.5 : 10} y="2.5" width="4.5" height="11" fill="currentColor" stroke="none" opacity="0.6" />
    </svg>
  );
}
