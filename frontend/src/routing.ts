// Orthogonal link routing around blocks.
//
// The candidate coordinates are the edges of every block (inflated by a margin), the
// channels half-way between them and the two port stubs. A* searches this sparse grid
// for the path with the smallest length plus a penalty per bend, never entering a block.

import type { Side } from "./api";
import type { Point, Rect } from "./geometry";

// Clearance kept around blocks and straight stub at each port; the tight setting is
// used when blocks are too close for the normal one.
const SETTINGS = [{ margin: 14, stub: 20 }, { margin: 4, stub: 8 }];
const BEND = 30; // cost of a bend, in pixels

const DIRS: Record<Side, [number, number]> = { left: [-1, 0], right: [1, 0], top: [0, -1], bottom: [0, 1] };
const STEP: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]]; // direction index -> unit step

function dirIndex(dx: number, dy: number): number {
  return dx > 0 ? 0 : dx < 0 ? 1 : dy > 0 ? 2 : 3;
}

const EPS = 0.5; // coordinates closer than this are the same grid line

/** Sorted grid coordinates (values within EPS merged, ``keep`` values exact) plus channel midpoints. */
function uniqueSorted(values: number[], keep: number[], margin: number): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  const s: number[] = [];
  for (const v of sorted) {
    const n = s.length;
    if (n && v - s[n - 1] < EPS) {
      if (keep.includes(v)) s[n - 1] = v; // port stubs keep their exact coordinate
      continue;
    }
    s.push(v);
  }
  const out: number[] = [];
  for (let k = 0; k < s.length; k++) {
    out.push(s[k]);
    if (k + 1 < s.length && s[k + 1] - s[k] > 2 * margin) out.push((s[k] + s[k + 1]) / 2);
  }
  return out;
}

class Heap {
  private items: [number, number][] = [];
  get size() {
    return this.items.length;
  }
  push(priority: number, value: number) {
    const a = this.items;
    a.push([priority, value]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop(): number {
    const a = this.items;
    const top = a[0][1];
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

/** Orthogonal path from port ``s`` (leaving towards ``sSide``) to port ``t``, or null. */
export function routeOrthogonal(s: Point, sSide: Side, t: Point, tSide: Side, blocks: Rect[]): Point[] | null {
  for (let k = 0; k < SETTINGS.length; k++) {
    const path = search(s, sSide, t, tSide, blocks, SETTINGS[k].margin, SETTINGS[k].stub, k === SETTINGS.length - 1);
    if (path) return path;
  }
  return null;
}

function search(s: Point, sSide: Side, t: Point, tSide: Side, blocks: Rect[], margin: number, stub: number,
                last: boolean): Point[] | null {
  const [sdx, sdy] = DIRS[sSide];
  const [tdx, tdy] = DIRS[tSide];
  const s1 = { x: s.x + sdx * stub, y: s.y + sdy * stub };
  const t1 = { x: t.x + tdx * stub, y: t.y + tdy * stub };
  const boxes = blocks.map((b) => ({ l: b.x - margin, r: b.x + b.w + margin, t: b.y - margin, b: b.y + b.h + margin }));
  const inside = (x: number, y: number) => boxes.some((b) => x > b.l + 0.5 && x < b.r - 0.5 && y > b.t + 0.5 && y < b.b - 0.5);
  if (!last && (inside(s1.x, s1.y) || inside(t1.x, t1.y))) return null; // too tight: retry with less clearance

  const xs = uniqueSorted([s1.x, t1.x, ...boxes.flatMap((b) => [b.l, b.r])], [s1.x, t1.x], margin);
  const ys = uniqueSorted([s1.y, t1.y, ...boxes.flatMap((b) => [b.t, b.b])], [s1.y, t1.y], margin);
  const nx = xs.length;
  const ny = ys.length;
  const idx = (i: number, j: number) => j * nx + i;
  const nearest = (arr: number[], v: number) => arr.findIndex((a) => Math.abs(a - v) < EPS);
  const si = nearest(xs, s1.x);
  const sj = nearest(ys, s1.y);
  const ti = nearest(xs, t1.x);
  const tj = nearest(ys, t1.y);
  if (si < 0 || sj < 0 || ti < 0 || tj < 0) return null;
  const start = idx(si, sj);
  const goal = idx(ti, tj);
  const free = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) free[idx(i, j)] = inside(xs[i], ys[j]) ? 0 : 1;
  free[start] = free[goal] = 1;

  // State = node * 4 + direction of arrival.
  const cost = new Float64Array(nx * ny * 4).fill(Infinity);
  const prev = new Int32Array(nx * ny * 4).fill(-1);
  const heap = new Heap();
  const s0 = start * 4 + dirIndex(sdx, sdy);
  const h = (i: number, j: number) => Math.abs(xs[i] - xs[ti]) + Math.abs(ys[j] - ys[tj]);
  cost[s0] = 0;
  heap.push(h(si, sj), s0);
  const arrive = dirIndex(-tdx, -tdy); // final heading into the target port
  let best = -1;
  let bestCost = Infinity;
  while (heap.size) {
    const state = heap.pop();
    const c = cost[state];
    if (c >= bestCost) break;
    const node = state >> 2;
    const d = state & 3;
    if (node === goal) {
      const total = c + (d === arrive ? 0 : BEND * (d === (arrive ^ 1) ? 2 : 1));
      if (total < bestCost) {
        bestCost = total;
        best = state;
      }
      continue;
    }
    const i = node % nx;
    const j = (node - i) / nx;
    for (let nd = 0; nd < 4; nd++) {
      if (nd === (d ^ 1)) continue; // no U-turn on the spot
      const ni = i + STEP[nd][0];
      const nj = j + STEP[nd][1];
      if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
      const next = idx(ni, nj);
      if (!free[next] || inside((xs[i] + xs[ni]) / 2, (ys[j] + ys[nj]) / 2)) continue;
      const nc = c + Math.abs(xs[ni] - xs[i]) + Math.abs(ys[nj] - ys[j]) + (nd === d ? 0 : BEND);
      const ns = next * 4 + nd;
      if (nc < cost[ns]) {
        cost[ns] = nc;
        prev[ns] = state;
        heap.push(nc + h(ni, nj), ns);
      }
    }
  }
  if (best < 0) return null;
  const pts: Point[] = [t];
  for (let st = best; st >= 0; st = prev[st]) {
    const node = st >> 2;
    const i = node % nx;
    pts.push({ x: xs[i], y: ys[(node - i) / nx] });
  }
  pts.push(s);
  pts.reverse();
  return simplify(orthogonalize(pts));
}

/** Make every segment exactly horizontal or vertical (sub-pixel offsets come from the ports). */
function orthogonalize(pts: Point[]): Point[] {
  const out = pts.map((p) => ({ ...p }));
  const n = out.length;
  // Interior points follow the endpoints: the first bend lines up with the source port,
  // the last bend with the target port.
  for (let k = 1; k < n - 1; k++) {
    const a = out[k - 1];
    const b = out[k];
    if (Math.abs(b.y - a.y) < 1) b.y = a.y;
    else if (Math.abs(b.x - a.x) < 1) b.x = a.x;
  }
  for (let k = n - 2; k > 0; k--) {
    const a = out[k + 1];
    const b = out[k];
    if (Math.abs(b.y - a.y) < 1) b.y = a.y;
    else if (Math.abs(b.x - a.x) < 1) b.x = a.x;
  }
  return out;
}

/** Drop repeated and collinear points (within a pixel). */
function simplify(pts: Point[]): Point[] {
  const out: Point[] = [];
  const same = (u: number, v: number) => Math.abs(u - v) < 1;
  for (const p of pts) {
    const n = out.length;
    if (n && same(out[n - 1].x, p.x) && same(out[n - 1].y, p.y)) {
      out[n - 1] = n === 1 ? out[0] : p;
      continue;
    }
    if (n >= 2) {
      const a = out[n - 2];
      const b = out[n - 1];
      if ((same(a.x, b.x) && same(b.x, p.x)) || (same(a.y, b.y) && same(b.y, p.y))) {
        out[n - 1] = p;
        continue;
      }
    }
    out.push(p);
  }
  return out;
}

/** SVG path through ``pts`` with rounded corners. */
export function roundedPath(pts: Point[], radius = 7): string {
  if (pts.length < 2) return "";
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let k = 1; k < pts.length - 1; k++) {
    const a = pts[k - 1];
    const b = pts[k];
    const c = pts[k + 1];
    const r = Math.min(radius, Math.hypot(b.x - a.x, b.y - a.y) / 2, Math.hypot(c.x - b.x, c.y - b.y) / 2);
    const dir = (d: number) => (Math.abs(d) < 0.5 ? 0 : Math.sign(d));
    const ux = dir(b.x - a.x);
    const uy = dir(b.y - a.y);
    const vx = dir(c.x - b.x);
    const vy = dir(c.y - b.y);
    d += ` L ${b.x - ux * r} ${b.y - uy * r} Q ${b.x} ${b.y} ${b.x + vx * r} ${b.y + vy * r}`;
  }
  const last = pts[pts.length - 1];
  return `${d} L ${last.x} ${last.y}`;
}

/** Point half-way along the path (for labels / interaction). */
export function midpoint(pts: Point[]): Point {
  let total = 0;
  for (let k = 1; k < pts.length; k++) total += Math.abs(pts[k].x - pts[k - 1].x) + Math.abs(pts[k].y - pts[k - 1].y);
  let half = total / 2;
  for (let k = 1; k < pts.length; k++) {
    const len = Math.abs(pts[k].x - pts[k - 1].x) + Math.abs(pts[k].y - pts[k - 1].y);
    if (half <= len && len > 0) {
      const f = half / len;
      return { x: pts[k - 1].x + f * (pts[k].x - pts[k - 1].x), y: pts[k - 1].y + f * (pts[k].y - pts[k - 1].y) };
    }
    half -= len;
  }
  return pts[0];
}

export interface Link {
  id: string;
  /** Links leaving the same output form one net: they may share a track. */
  net: string;
  s: Point;
  sSide: Side;
  t: Point;
  tSide: Side;
}

const TRACK = 6; // spacing of parallel links sharing a channel

/**
 * Route every link, then pull apart the parallel segments of different nets that the
 * router laid on the same line (they would read as one link). The first and last
 * segments (the port stubs) stay put; moving an inner segment sideways only lengthens
 * or shortens its two neighbours, so the path stays orthogonal.
 */
export function routeLinks(links: Link[], blocks: Rect[]): Map<string, Point[]> {
  const routes = new Map<string, Point[]>();
  for (const l of links) {
    const pts = routeOrthogonal(l.s, l.sSide, l.t, l.tSide, blocks);
    if (pts) routes.set(l.id, pts);
  }
  interface Seg {
    id: string;
    net: string;
    k: number; // segment from pts[k] to pts[k + 1]
    horiz: boolean;
    at: number;
    lo: number;
    hi: number;
  }
  const groups = new Map<string, Seg[]>();
  const netOf = new Map(links.map((l) => [l.id, l.net]));
  for (const [id, pts] of routes) {
    for (let k = 1; k < pts.length - 2; k++) {
      const a = pts[k];
      const b = pts[k + 1];
      const horiz = Math.abs(a.y - b.y) < 0.5;
      const seg: Seg = {
        id, net: netOf.get(id)!, k, horiz,
        at: horiz ? a.y : a.x,
        lo: horiz ? Math.min(a.x, b.x) : Math.min(a.y, b.y),
        hi: horiz ? Math.max(a.x, b.x) : Math.max(a.y, b.y),
      };
      const key = `${horiz ? "h" : "v"}${Math.round(seg.at * 2) / 2}`;
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(seg);
    }
  }
  const shifts: { seg: Seg; d: number }[] = [];
  for (const segs of groups.values()) {
    if (new Set(segs.map((s) => s.net)).size < 2) continue;
    segs.sort((a, b) => a.lo - b.lo);
    // Clusters of overlapping segments along the line.
    let cluster: Seg[] = [];
    let end = -Infinity;
    const flush = () => {
      const nets = [...new Set(cluster.map((s) => s.net))].sort();
      if (nets.length > 1) {
        const step = Math.min(TRACK, 18 / (nets.length - 1));
        for (const seg of cluster) shifts.push({ seg, d: (nets.indexOf(seg.net) - (nets.length - 1) / 2) * step });
      }
      cluster = [];
    };
    for (const seg of segs) {
      if (cluster.length && seg.lo >= end - 1) flush();
      cluster.push(seg);
      end = cluster.length === 1 ? seg.hi : Math.max(end, seg.hi);
    }
    flush();
  }
  for (const { seg, d } of shifts) {
    if (!d) continue;
    const pts = routes.get(seg.id)!;
    for (const p of [pts[seg.k], pts[seg.k + 1]]) {
      if (seg.horiz) p.y += d;
      else p.x += d;
    }
  }
  return routes;
}
