// Alignment while dragging a block: connected ports line up (straight links) and block
// edges / centres align with neighbouring blocks. Returns the snapped position and the
// guide lines to draw.

import type { Edge } from "@xyflow/react";
import { blockSize, horizontal, placePoint, portKey, portPlaces, type Point, type Rect } from "./geometry";
import type { BlockNodeType } from "./graph";

const PORT_SNAP = 12; // px (flow units) within which connected ports align
const EDGE_SNAP = 7; // px within which block edges / centres align

export interface Guide {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Candidate {
  d: number;
  guide: (moved: Rect) => Guide;
}

/** Absolute position and side of one port of a block. */
export function portAnchor(node: BlockNodeType, key: string, at: Point = node.position) {
  const places = portPlaces(node.data);
  const place = places[key];
  if (!place) return null;
  const p = placePoint(place, blockSize(node.data, places));
  return { x: at.x + p.x, y: at.y + p.y, side: place.side };
}

export function nodeRect(node: BlockNodeType, at: Point = node.position): Rect {
  const { w, h } = blockSize(node.data);
  return { x: at.x, y: at.y, w, h };
}

function best(cands: Candidate[], limit: number): Candidate | null {
  let out: Candidate | null = null;
  for (const c of cands) if (Math.abs(c.d) <= limit && (!out || Math.abs(c.d) < Math.abs(out.d))) out = c;
  return out;
}

export function snapPosition(node: BlockNodeType, pos: Point, nodes: BlockNodeType[], edges: Edge[]): { pos: Point; guides: Guide[] } {
  const me = nodeRect(node, pos);
  const others = nodes.filter((n) => n.id !== node.id);
  const portX: Candidate[] = [];
  const portY: Candidate[] = [];
  for (const e of edges) {
    if (e.source !== node.id && e.target !== node.id) continue;
    const mine = e.source === node.id ? portKey("out", e.sourceHandle ?? "") : portKey("in", e.targetHandle ?? "");
    const otherId = e.source === node.id ? e.target : e.source;
    const otherKey = e.source === node.id ? portKey("in", e.targetHandle ?? "") : portKey("out", e.sourceHandle ?? "");
    const other = others.find((n) => n.id === otherId);
    if (!other) continue;
    const a = portAnchor(node, mine, pos);
    const b = portAnchor(other, otherKey);
    if (!a || !b) continue;
    if (horizontal(a.side) && horizontal(b.side)) {
      portY.push({ d: b.y - a.y, guide: () => ({ x1: Math.min(a.x, b.x), y1: b.y, x2: Math.max(a.x, b.x), y2: b.y }) });
    } else if (!horizontal(a.side) && !horizontal(b.side)) {
      portX.push({ d: b.x - a.x, guide: () => ({ x1: b.x, y1: Math.min(a.y, b.y), x2: b.x, y2: Math.max(a.y, b.y) }) });
    }
  }

  const edgeX: Candidate[] = [];
  const edgeY: Candidate[] = [];
  for (const o of others) {
    const r = nodeRect(o);
    const top = Math.min(r.y, me.y);
    const bottom = Math.max(r.y + r.h, me.y + me.h);
    const left = Math.min(r.x, me.x);
    const right = Math.max(r.x + r.w, me.x + me.w);
    for (const [mine, theirs] of [[me.x, r.x], [me.x + me.w / 2, r.x + r.w / 2], [me.x + me.w, r.x + r.w]]) {
      edgeX.push({ d: theirs - mine, guide: () => ({ x1: theirs, y1: top, x2: theirs, y2: bottom }) });
    }
    for (const [mine, theirs] of [[me.y, r.y], [me.y + me.h / 2, r.y + r.h / 2], [me.y + me.h, r.y + r.h]]) {
      edgeY.push({ d: theirs - mine, guide: () => ({ x1: left, y1: theirs, x2: right, y2: theirs }) });
    }
  }

  const bx = best(portX, PORT_SNAP) ?? best(edgeX, EDGE_SNAP);
  const by = best(portY, PORT_SNAP) ?? best(edgeY, EDGE_SNAP);
  const snapped = { x: pos.x + (bx?.d ?? 0), y: pos.y + (by?.d ?? 0) };
  const moved = nodeRect(node, snapped);
  const guides = [bx, by].filter((c): c is Candidate => !!c).map((c) => c.guide(moved));
  return { pos: snapped, guides };
}

/** After an automatic layout: move blocks fed by a single link so that the link is straight. */
export function straighten(nodes: BlockNodeType[], edges: Edge[]): BlockNodeType[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const order = [...nodes].sort((a, b) => a.position.x - b.position.x);
  for (const n of order) {
    const incoming = edges.filter((e) => e.target === n.id);
    if (incoming.length !== 1) continue;
    const e = incoming[0];
    const src = byId.get(e.source);
    if (!src) continue;
    const a = portAnchor(src, portKey("out", e.sourceHandle ?? ""));
    const b = portAnchor(n, portKey("in", e.targetHandle ?? ""));
    if (!a || !b || !horizontal(a.side) || !horizontal(b.side)) continue;
    const dy = a.y - b.y;
    if (Math.abs(dy) > 60 || a.x >= b.x) continue;
    const moved = { ...n, position: { x: n.position.x, y: n.position.y + dy } };
    const r = nodeRect(moved);
    const overlaps = nodes.some((o) => {
      if (o.id === n.id) return false;
      const q = nodeRect(byId.get(o.id)!);
      return r.x < q.x + q.w + 10 && q.x < r.x + r.w + 10 && r.y < q.y + q.h + 10 && q.y < r.y + r.h + 10;
    });
    if (!overlaps) byId.set(n.id, moved);
  }
  return nodes.map((n) => byId.get(n.id)!);
}

/**
 * After the automatic layout (a fixed grid): push blocks down until none overlaps a block
 * above it, using the real block sizes. Blocks with ports on the bottom side keep extra
 * room below for the links that enter there.
 */
export function spread(nodes: BlockNodeType[]): BlockNodeType[] {
  const GAP = 30;
  const placed: { id: string; r: Rect; below: number }[] = [];
  const out = new Map<string, BlockNodeType>();
  for (const n of [...nodes].sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x)) {
    let r = nodeRect(n);
    for (let moved = true; moved; ) {
      moved = false;
      for (const o of placed) {
        const overlapX = r.x < o.r.x + o.r.w + 10 && o.r.x < r.x + r.w + 10;
        const bottom = o.r.y + o.r.h + o.below;
        if (overlapX && r.y < bottom && r.y + r.h > o.r.y - GAP) {
          r = { ...r, y: bottom };
          moved = true;
        }
      }
    }
    const south = Object.values(portPlaces(n.data)).some((p) => p.side === "bottom");
    placed.push({ id: n.id, r, below: GAP + (south ? 40 : 0) });
    out.set(n.id, r.y === n.position.y ? n : { ...n, position: { x: n.position.x, y: r.y } });
  }
  return nodes.map((n) => out.get(n.id)!);
}
