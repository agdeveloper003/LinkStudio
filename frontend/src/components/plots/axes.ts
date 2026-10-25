// Cartesian axes (grid, ticks, labels) for the canvas plots.

import { niceTicks, tickText } from "../../units";
import { PLOT_COLORS } from "./useSize";

export interface Frame {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Axis {
  lo: number;
  hi: number;
  label: string;
  ticks?: number;
}

const FONT = "10.5px Inter, system-ui, sans-serif";
export const MARGIN = { left: 50, right: 10, top: 8, bottom: 34 };

/** Plot area inside a canvas of the given size (optionally square). */
export function plotFrame(width: number, height: number, square = false): Frame {
  let w = Math.max(width - MARGIN.left - MARGIN.right, 10);
  let h = Math.max(height - MARGIN.top - MARGIN.bottom, 10);
  if (square) w = h = Math.min(w, h);
  const left = MARGIN.left + (width - MARGIN.left - MARGIN.right - w) / 2;
  return { left, top: MARGIN.top, width: w, height: h };
}

/** Draw grid, tick labels and axis titles; returns data -> pixel mappings. */
export function drawAxes(ctx: CanvasRenderingContext2D, f: Frame, x: Axis, y: Axis) {
  const toX = (v: number) => f.left + ((v - x.lo) / (x.hi - x.lo || 1)) * f.width;
  const toY = (v: number) => f.top + f.height - ((v - y.lo) / (y.hi - y.lo || 1)) * f.height;
  const xt = niceTicks(x.lo, x.hi, x.ticks ?? Math.max(4, Math.floor(f.width / 70)));
  const yt = niceTicks(y.lo, y.hi, y.ticks ?? Math.max(4, Math.floor(f.height / 40)));

  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = PLOT_COLORS.grid;
  ctx.beginPath();
  for (const v of xt) {
    ctx.moveTo(Math.round(toX(v)) + 0.5, f.top);
    ctx.lineTo(Math.round(toX(v)) + 0.5, f.top + f.height);
  }
  for (const v of yt) {
    ctx.moveTo(f.left, Math.round(toY(v)) + 0.5);
    ctx.lineTo(f.left + f.width, Math.round(toY(v)) + 0.5);
  }
  ctx.stroke();
  ctx.strokeStyle = PLOT_COLORS.axis;
  ctx.strokeRect(f.left + 0.5, f.top + 0.5, f.width - 1, f.height - 1);

  ctx.fillStyle = PLOT_COLORS.text;
  ctx.font = FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of xt) ctx.fillText(tickText(v, xt), toX(v), f.top + f.height + 4);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yt) ctx.fillText(tickText(v, yt), f.left - 5, toY(v));

  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(x.label, f.left + f.width / 2, f.top + f.height + MARGIN.bottom - 1);
  ctx.translate(11, f.top + f.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText(y.label, 0, 0);
  ctx.restore();
  return { toX, toY };
}
