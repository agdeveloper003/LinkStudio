// Block geometry: size of a block node and where each port sits on its border.
//
// Ports are keyed "in:<name>" / "out:<name>". A port on the left or right side has an
// offset along the body (below the header); on the top or bottom side, along the width.
// Sizes are computed here (not measured from the DOM) so that snapping and routing can
// use them while a block is being dragged.

import type { PortPlace, Side } from "./api";
import type { BlockData } from "./graph";

export const NODE_WIDTH = 184;
export const HEADER = 52; // name, type and parameter summary
export const ROW = 22; // vertical pitch of side ports
const BOTTOM_PAD = 6;
const TOP_PITCH = 64; // horizontal pitch of top/bottom ports

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export const portKey = (dir: "in" | "out", name: string) => `${dir}:${name}`;

/** Every port with its place: stored overrides, defaults for the others. */
export function portPlaces(data: BlockData): Record<string, PortPlace> {
  const stored = data.ports ?? {};
  const keys = [...data.inputs.map((p) => portKey("in", p.name)), ...data.outputs.map((p) => portKey("out", p.name))];
  const out: Record<string, PortPlace> = {};
  // Default: inputs down the left side, outputs down the right side, one row each.
  const left = keys.filter((k) => !stored[k] && k.startsWith("in:"));
  const right = keys.filter((k) => !stored[k] && k.startsWith("out:"));
  const sideCount = (side: Side) => keys.filter((k) => stored[k]?.side === side).length;
  const rows = Math.max(left.length + sideCount("left"), right.length + sideCount("right"), 1);
  const used: Record<string, Set<number>> = { left: new Set(), right: new Set() };
  for (const k of keys) {
    const s = stored[k];
    if (s && (s.side === "left" || s.side === "right")) used[s.side].add(Math.round(s.offset * rows - 0.5));
  }
  const fill = (list: string[], side: "left" | "right") => {
    let row = 0;
    for (const k of list) {
      while (used[side].has(row)) row += 1;
      out[k] = { side, offset: (row + 0.5) / rows };
      row += 1;
    }
  };
  fill(left, "left");
  fill(right, "right");
  for (const k of keys) if (stored[k]) out[k] = stored[k];
  return out;
}

/** Width and height of a block node. */
export function blockSize(data: BlockData, places = portPlaces(data)): { w: number; h: number } {
  const count = (side: Side) => Object.values(places).filter((p) => p.side === side).length;
  const rows = Math.max(count("left"), count("right"), 1);
  const cols = Math.max(count("top"), count("bottom"));
  return { w: Math.max(NODE_WIDTH, (cols + 1) * TOP_PITCH), h: HEADER + rows * ROW + BOTTOM_PAD };
}

/** Port position relative to the block's top-left corner. */
export function placePoint(place: PortPlace, size: { w: number; h: number }): Point {
  const body = size.h - HEADER - BOTTOM_PAD;
  switch (place.side) {
    case "left":
      return { x: 0, y: HEADER + place.offset * body };
    case "right":
      return { x: size.w, y: HEADER + place.offset * body };
    case "top":
      return { x: place.offset * size.w, y: 0 };
    default:
      return { x: place.offset * size.w, y: size.h };
  }
}

/** Nearest side and snapped offset for a pointer at ``p`` (relative to the block). */
export function placeFromPoint(p: Point, size: { w: number; h: number }): PortPlace {
  const body = size.h - HEADER - BOTTOM_PAD;
  const dist: [Side, number][] = [
    ["left", Math.abs(p.x)],
    ["right", Math.abs(size.w - p.x)],
    ["top", Math.abs(p.y)],
    ["bottom", Math.abs(size.h - p.y)],
  ];
  const side = dist.reduce((a, b) => (b[1] < a[1] ? b : a))[0];
  if (side === "left" || side === "right") {
    const steps = Math.max(1, Math.round((2 * body) / ROW)); // half-row grid
    const k = Math.round(((p.y - HEADER) / body) * steps);
    return { side, offset: Math.min(steps - 1, Math.max(1, k)) / steps };
  }
  const steps = 12;
  const k = Math.round((p.x / size.w) * steps);
  return { side, offset: Math.min(steps - 1, Math.max(1, k)) / steps };
}

export const horizontal = (side: Side) => side === "left" || side === "right";
