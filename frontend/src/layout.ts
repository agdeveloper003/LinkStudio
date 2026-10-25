// Block orientation and the automatic layout.
//
// A block faces right (inputs on the left side, outputs on the right) or left (the
// mirror image), so that a chain can run in either direction without crossing links.
// The layout solver arranges the blocks in layers along the signal flow, keeps the main
// chain on one straight line and, when the chain is long, wraps it into rows that run
// alternately left-to-right and right-to-left (a snake / horseshoe), turning the blocks of
// the reversed rows to face left. Scopes and meters fed from their bottom side sit just
// above the link they tap.

import type { Edge } from "@xyflow/react";
import type { PortPlace } from "./api";
import { blockSize, horizontal, portKey, portPlaces } from "./geometry";
import type { BlockData, BlockNodeType } from "./graph";
import { nodeRect, portAnchor, spread } from "./snap";

export type Facing = "right" | "left";

const mirror = (p: PortPlace): PortPlace =>
  p.side === "left" ? { side: "right", offset: p.offset } : p.side === "right" ? { side: "left", offset: p.offset } : { side: p.side, offset: 1 - p.offset };

/**
 * Stored port places for a block facing ``facing`` with its side ports in the default
 * order. Ports the user put on the top or bottom side stay there.
 */
export function orientPorts(data: BlockData, facing: Facing): Record<string, PortPlace> | undefined {
  const keep = Object.fromEntries(Object.entries(data.ports ?? {}).filter(([, p]) => !horizontal(p.side)));
  if (facing === "right") return Object.keys(keep).length ? keep : undefined;
  const out: Record<string, PortPlace> = { ...keep };
  for (const [k, p] of Object.entries(portPlaces({ ...data, ports: keep }))) if (horizontal(p.side)) out[k] = mirror(p);
  return out;
}

const samePlaces = (a: Record<string, PortPlace>, b: Record<string, PortPlace>) =>
  Object.keys(a).length === Object.keys(b).length &&
  Object.entries(a).every(([k, p]) => b[k] && b[k].side === p.side && Math.abs(b[k].offset - p.offset) < 1e-6);

/** Which way a block faces, or null when the user arranged its side ports by hand. */
export function facing(data: BlockData): Facing | null {
  const now = portPlaces(data);
  if (!Object.values(now).some((p) => horizontal(p.side))) return null; // nothing on the sides
  for (const f of ["right", "left"] as Facing[]) {
    if (samePlaces(now, portPlaces({ ...data, ports: orientPorts(data, f) }))) return f;
  }
  return null;
}

/** True when the block's side inputs are on the right and its side outputs on the left. */
export function facesLeft(data: BlockData): boolean {
  const places = portPlaces(data);
  const sides = (dir: "in" | "out") =>
    Object.entries(places).filter(([k, p]) => k.startsWith(`${dir}:`) && horizontal(p.side)).map(([, p]) => p.side);
  const ins = sides("in");
  const outs = sides("out");
  return ins.length + outs.length > 0 && ins.every((s) => s === "right") && outs.every((s) => s === "left");
}

/** Turn a block around: the side ports swap sides (top / bottom ones mirror their position). */
export function flipPorts(data: BlockData): Record<string, PortPlace> | undefined {
  const f = facing(data);
  if (f) return orientPorts(data, f === "right" ? "left" : "right");
  return Object.fromEntries(Object.entries(portPlaces(data)).map(([k, p]) => [k, mirror(p)]));
}

/**
 * Which way a block should face given where its neighbours are: +1 for every link
 * that arrives from the left or leaves to the right, -1 for the opposite.
 */
function neighbourScore(n: BlockNodeType, at: { x: number; y: number }, byId: Map<string, BlockNodeType>, edges: Edge[]): number {
  const me = nodeRect(n, at);
  let score = 0;
  for (const e of edges) {
    const other = e.target === n.id ? byId.get(e.source) : e.source === n.id ? byId.get(e.target) : undefined;
    if (!other || other.id === n.id) continue;
    const r = nodeRect(other);
    const side = r.x + r.w <= me.x ? -1 : r.x >= me.x + me.w ? 1 : 0; // other block on my left / right
    score += e.target === n.id ? -side : side;
  }
  return score;
}

/**
 * After blocks were dragged: turn a moved block around when its neighbours moved to the
 * other side (the links would otherwise loop around it). Only blocks whose side ports are
 * in a default arrangement are turned, and only when the drag changed the verdict.
 */
export function autoOrient(nodes: BlockNodeType[], edges: Edge[], moved: Map<string, { x: number; y: number }>): BlockNodeType[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  let changed = false;
  const out = nodes.map((n) => {
    const start = moved.get(n.id);
    if (!start) return n;
    const f = facing(n.data);
    if (!f) return n;
    const before = Math.sign(neighbourScore(n, start, byId, edges));
    const after = Math.sign(neighbourScore(n, n.position, byId, edges));
    if (after === before || after === 0) return n;
    const want: Facing = after > 0 ? "right" : "left";
    if (want === f) return n;
    changed = true;
    return { ...n, data: { ...n.data, ports: orientPorts(n.data, want) } };
  });
  return changed ? out : nodes;
}

// ------------------------------------------------------------------ layout

const GAP_X = 72; // between columns: room for the links and their bends
const GAP_LANE = 44; // between the stacked blocks of one column
const GAP_ROW = 96; // between the rows of a wrapped chain: room for the turn
const TAP_GAP = 40; // between a tapping scope and the row under it

/** Blocks fed only from their bottom (scopes, power meters) and without outputs. */
function isTap(n: BlockNodeType): boolean {
  if (n.data.outputs.length || !n.data.inputs.length) return false;
  const places = portPlaces(n.data);
  return n.data.inputs.every((p) => places[portKey("in", p.name)]?.side === "bottom");
}

/**
 * Arrange the blocks along the signal flow. ``aspect`` is the width / height of the canvas:
 * a long chain wraps into as many rows as make the drawing fit it best.
 */
export function layoutGraph(nodes: BlockNodeType[], edges: Edge[], aspect = 1.6): BlockNodeType[] {
  if (!nodes.length) return nodes;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const links = edges.filter((e) => byId.has(e.source) && byId.has(e.target) && e.source !== e.target);
  const taps = new Set(nodes.filter((n) => links.some((e) => e.target === n.id) && isTap(n)).map((n) => n.id));
  const main = nodes.filter((n) => !taps.has(n.id));
  const mainLinks = links.filter((e) => !taps.has(e.target));
  const linked = new Set(mainLinks.flatMap((e) => [e.source, e.target]));
  const loose = main.filter((n) => !linked.has(n.id));
  const chain = main.filter((n) => linked.has(n.id));
  const preds = new Map<string, string[]>(chain.map((n) => [n.id, []]));
  const succs = new Map<string, string[]>(chain.map((n) => [n.id, []]));
  for (const e of mainLinks) {
    if (!preds.get(e.target)!.includes(e.source)) preds.get(e.target)!.push(e.source);
    if (!succs.get(e.source)!.includes(e.target)) succs.get(e.source)!.push(e.target);
  }

  // Layers: longest path from the sources (links closing a cycle are ignored).
  const layer = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (id: string): number => {
    if (layer.has(id)) return layer.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const d = Math.max(-1, ...preds.get(id)!.map(depth)) + 1;
    visiting.delete(id);
    layer.set(id, d);
    return d;
  };
  chain.forEach((n) => depth(n.id));
  // A source feeding a late block (a training sequence, a second stream) sits just before it.
  for (const n of chain) {
    if (preds.get(n.id)!.length) continue;
    const next = succs.get(n.id)!.map((s) => layer.get(s)!);
    if (next.length) layer.set(n.id, Math.max(layer.get(n.id)!, Math.min(...next) - 1));
  }
  const nLayers = Math.max(0, ...layer.values()) + 1;

  // Main chain: the longest path, kept on one straight line.
  const spine = new Set<string>();
  if (chain.length) {
    let cur: string | undefined = chain.reduce((a, b) => (layer.get(b.id)! > layer.get(a.id)! ? b : a)).id;
    while (cur) {
      spine.add(cur);
      const l: number = layer.get(cur)! - 1;
      const cands: string[] = preds.get(cur)!.filter((p) => layer.get(p) === l);
      cur = cands.sort((a, b) => (preds.get(b)!.length ? 1 : 0) - (preds.get(a)!.length ? 1 : 0))[0];
    }
  }

  // Order inside the layers: the spine first, the others by the mean place of their neighbours.
  const layers: string[][] = Array.from({ length: nLayers }, () => []);
  for (const n of chain) layers[layer.get(n.id)!].push(n.id);
  const index = new Map<string, number>();
  const reindex = () => layers.forEach((ids) => ids.forEach((id, k) => index.set(id, k)));
  reindex();
  const mean = (ids: string[], fallback: number) => (ids.length ? ids.reduce((s, id) => s + (index.get(id) ?? 0), 0) / ids.length : fallback);
  for (let sweep = 0; sweep < 4; sweep++) {
    const down = sweep % 2 === 0;
    const order = down ? layers : [...layers].reverse();
    for (const ids of order) {
      const key = new Map(ids.map((id) => [id, spine.has(id) ? -1 : mean(down ? preds.get(id)! : succs.get(id)!, index.get(id)!)]));
      ids.sort((a, b) => key.get(a)! - key.get(b)!);
      ids.forEach((id, k) => index.set(id, k));
    }
  }

  // Rows: wrap the layers so that the drawing fits the canvas shape.
  const size = (id: string) => blockSize(byId.get(id)!.data);
  const layerW = layers.map((ids) => Math.max(0, ...ids.map((id) => size(id).w)));
  const layerH = layers.map((ids) => ids.reduce((s, id) => s + size(id).h, 0) + GAP_LANE * Math.max(0, ids.length - 1));
  const totalW = layerW.reduce((s, w) => s + w, 0) + GAP_X * Math.max(0, nLayers - 1);
  const rowH = Math.max(1, ...layerH) + (taps.size ? 160 : 0);
  let rows = nLayers > 4 ? Math.round(Math.sqrt(totalW / (aspect * (rowH + GAP_ROW)))) : 1;
  rows = Math.max(1, Math.min(rows, Math.ceil(nLayers / 2)));
  const perRow = Math.ceil(nLayers / Math.max(rows, 1));
  const rowOf = (l: number) => Math.floor(l / perRow);
  const colOf = (l: number) => (rowOf(l) % 2 ? perRow - 1 - (l % perRow) : l % perRow);
  const colW = Array.from({ length: perRow }, (_, c) => Math.max(0, ...layerW.filter((_, l) => colOf(l) === c)));
  const colX: number[] = [];
  colW.reduce((x, w, c) => ((colX[c] = x), x + w + GAP_X), 0);

  // Place: faced ports first (sizes do not depend on facing), then x by column, y by lane.
  const placed = new Map<string, BlockNodeType>();
  const facingOf = (id: string): Facing => (rowOf(layer.get(id)!) % 2 ? "left" : "right");
  for (const n of chain) {
    const f = facingOf(n.id);
    placed.set(n.id, { ...n, data: { ...n.data, ports: facing(n.data) === f ? n.data.ports : orientPorts(n.data, f) } });
  }
  const anchorY = (id: string, key: string) => {
    const n = placed.get(id)!;
    return portAnchor({ ...n, position: { x: 0, y: 0 } }, key)?.y;
  };
  const tapsOfRow = (r: number) =>
    [...taps].filter((t) => {
      const src = links.find((e) => e.target === t)?.source;
      return src && layer.has(src) && rowOf(layer.get(src)!) === r;
    });
  let top = 0;
  const nRows = Math.ceil(nLayers / perRow);
  for (let r = 0; r < nRows; r++) {
    const rowTaps = tapsOfRow(r);
    const tapH = rowTaps.length ? Math.max(...rowTaps.map((t) => size(t).h)) + TAP_GAP : 0;
    const y0 = top + tapH;
    const ls = Array.from({ length: perRow }, (_, k) => r * perRow + k).filter((l) => l < nLayers);
    // Lane 0: every block lines its input up with the output feeding it.
    const y = new Map<string, number>();
    for (const l of ls) {
      const id = layers[l][0];
      if (!id) continue;
      const e = mainLinks.find((x) => x.target === id && y.has(x.source) && layers[layer.get(x.source)!][0] === x.source);
      if (e) {
        const a = anchorY(e.source, portKey("out", e.sourceHandle ?? ""));
        const b = anchorY(id, portKey("in", e.targetHandle ?? ""));
        y.set(id, a !== undefined && b !== undefined ? y.get(e.source)! + a - b : y0);
      } else y.set(id, y0);
    }
    const lift = Math.min(y0, ...y.values()) - y0;
    for (const [id, v] of y) y.set(id, v - lift);
    let laneTop = Math.max(y0, ...[...y].map(([id, v]) => v + size(id).h)) + GAP_LANE;
    // Further lanes, one per depth, under the spine.
    const depthMax = Math.max(0, ...ls.map((l) => layers[l].length));
    for (let k = 1; k < depthMax; k++) {
      let bottom = laneTop;
      for (const l of ls) {
        const id = layers[l][k];
        if (!id) continue;
        // Line up with a block feeding it from the same lane when that keeps the lane order.
        const e = mainLinks.find((x) => x.target === id && y.has(x.source) && layers[layer.get(x.source)!][k] === x.source);
        let v = laneTop;
        if (e) {
          const a = anchorY(e.source, portKey("out", e.sourceHandle ?? ""));
          const b = anchorY(id, portKey("in", e.targetHandle ?? ""));
          if (a !== undefined && b !== undefined) v = Math.max(laneTop, y.get(e.source)! + a - b);
        }
        y.set(id, v);
        bottom = Math.max(bottom, v + size(id).h);
      }
      laneTop = bottom + GAP_LANE;
    }
    for (const l of ls) {
      for (const id of layers[l]) {
        const c = colOf(l);
        const n = placed.get(id)!;
        placed.set(id, { ...n, position: { x: colX[c] + (colW[c] - size(id).w) / 2, y: y.get(id)! } });
      }
    }
    // Scopes above the link they tap, centred on the gap after the source block.
    let nextFree = -Infinity;
    const tapX = (t: string) => {
      const src = placed.get(links.find((e) => e.target === t)!.source)!;
      const s = nodeRect(src);
      const gap = facingOf(src.id) === "right" ? s.x + s.w + GAP_X / 2 : s.x - GAP_X / 2;
      return gap - size(t).w / 2;
    };
    for (const t of rowTaps.sort((a, b) => tapX(a) - tapX(b))) {
      const x = Math.max(tapX(t), nextFree);
      const n = byId.get(t)!;
      placed.set(t, { ...n, position: { x, y: top + tapH - TAP_GAP - size(t).h } });
      nextFree = x + size(t).w + 24;
    }
    top = laneTop - GAP_LANE + GAP_ROW;
  }

  // Scopes tapping something that is not in the chain, and unconnected blocks: a last row.
  let x = 0;
  for (const n of [...nodes.filter((m) => taps.has(m.id) && !placed.has(m.id)), ...loose]) {
    placed.set(n.id, { ...n, position: { x, y: top } });
    x += size(n.id).w + GAP_X;
  }
  // Keep the drawing where it was (top-left corner), and resolve any overlap left.
  const minX = Math.min(...nodes.map((n) => n.position.x));
  const minY = Math.min(...nodes.map((n) => n.position.y));
  const boxX = Math.min(...[...placed.values()].map((n) => n.position.x));
  const boxY = Math.min(...[...placed.values()].map((n) => n.position.y));
  const out = nodes.map((n) => {
    const p = placed.get(n.id) ?? n;
    return { ...p, position: { x: Math.round(p.position.x - boxX + minX), y: Math.round(p.position.y - boxY + minY) } };
  });
  return spread(out);
}
