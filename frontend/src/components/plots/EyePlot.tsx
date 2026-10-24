import { useEffect, useRef } from "react";
import { PLOT_COLORS, setupCanvas, useSize } from "./useSize";

/** Overlaid two-symbol traces, centred on the sampling instant. */
export function EyePlot({ t, traces }: { t: number[]; traces: number[][] }) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || width < 10 || height < 10 || !t.length) return;
    const ctx = setupCanvas(canvas.current, width, height);
    if (!ctx) return;
    const t0 = t[0];
    const t1 = t[t.length - 1];
    const toX = (v: number) => ((v - t0) / (t1 - t0)) * (width - 8) + 4;
    const toY = (v: number) => height / 2 - v * (height / 2 - 8);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = PLOT_COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    ctx.strokeStyle = "rgba(248,113,113,0.6)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(0), 0);
    ctx.lineTo(toX(0), height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(96,165,250,0.28)";
    ctx.lineWidth = 1;
    for (const tr of traces) {
      ctx.beginPath();
      tr.forEach((v, k) => (k ? ctx.lineTo(toX(t[k]), toY(v)) : ctx.moveTo(toX(t[k]), toY(v))));
      ctx.stroke();
    }
  }, [t, traces, width, height]);

  return (
    <div ref={box} className="plot-box">
      <canvas ref={canvas} />
    </div>
  );
}
