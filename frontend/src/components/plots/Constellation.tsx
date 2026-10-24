import { useEffect, useRef } from "react";
import { PLOT_COLORS, setupCanvas, useSize } from "./useSize";

interface Props {
  i: number[];
  q: number[];
  refI?: number[];
  refQ?: number[];
  color?: string;
}

/** I/Q scatter plot on a square, auto-scaled canvas. */
export function Constellation({ i, q, refI, refQ, color = "#34d399" }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || width < 10 || height < 10) return;
    const side = Math.min(width, height);
    const ctx = setupCanvas(canvas.current, side, side);
    if (!ctx) return;
    let lim = 0;
    for (let k = 0; k < i.length; k++) lim = Math.max(lim, Math.abs(i[k]), Math.abs(q[k]));
    for (let k = 0; k < (refI?.length ?? 0); k++) lim = Math.max(lim, Math.abs(refI![k]), Math.abs(refQ![k]));
    lim = lim > 0 ? lim * 1.15 : 1.5;
    const toX = (v: number) => side / 2 + (v / lim) * (side / 2);
    const toY = (v: number) => side / 2 - (v / lim) * (side / 2);

    ctx.clearRect(0, 0, side, side);
    ctx.strokeStyle = PLOT_COLORS.grid;
    ctx.lineWidth = 1;
    const step = lim > 3 ? 1 : lim > 1.2 ? 0.5 : 0.25;
    for (let v = -Math.floor(lim / step) * step; v <= lim; v += step) {
      ctx.beginPath();
      ctx.moveTo(toX(v), 0);
      ctx.lineTo(toX(v), side);
      ctx.moveTo(0, toY(v));
      ctx.lineTo(side, toY(v));
      ctx.stroke();
    }
    ctx.strokeStyle = PLOT_COLORS.axis;
    ctx.beginPath();
    ctx.moveTo(side / 2, 0);
    ctx.lineTo(side / 2, side);
    ctx.moveTo(0, side / 2);
    ctx.lineTo(side, side / 2);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.globalAlpha = i.length > 1500 ? 0.35 : 0.55;
    const r = side > 300 ? 1.6 : 1.3;
    for (let k = 0; k < i.length; k++) {
      ctx.beginPath();
      ctx.arc(toX(i[k]), toY(q[k]), r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (refI && refQ) {
      ctx.strokeStyle = "#f87171";
      ctx.lineWidth = 1.5;
      for (let k = 0; k < refI.length; k++) {
        const x = toX(refI[k]);
        const y = toY(refQ[k]);
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4);
        ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x - 4, y + 4);
        ctx.lineTo(x + 4, y - 4);
        ctx.stroke();
      }
    }
    ctx.fillStyle = PLOT_COLORS.text;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText(`±${lim.toFixed(2)}`, 4, 12);
  }, [i, q, refI, refQ, width, height, color]);

  return (
    <div ref={box} className="plot-box centered">
      <canvas ref={canvas} />
    </div>
  );
}
