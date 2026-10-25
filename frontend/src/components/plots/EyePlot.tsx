import { useEffect, useRef } from "react";
import { axisLabel, siScale } from "../../units";
import { drawAxes, plotFrame } from "./axes";
import { PLOT_COLORS, setupCanvas, useSize } from "./useSize";

interface Props {
  t: number[];
  tunit: string;
  ui: number;
  traces: number[][];
  unit?: string;
}

/** Overlaid two-symbol traces: the vertical axis is centred on the mean and spans all traces. */
export function EyePlot({ t, tunit, ui, traces, unit }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || width < 40 || height < 40 || !t.length) return;
    const ctx = setupCanvas(canvas.current, width, height);
    if (!ctx) return;
    let sum = 0;
    let n = 0;
    for (const tr of traces) for (const v of tr) {
      sum += v;
      n += 1;
    }
    const mean = n ? sum / n : 0;
    let half = 0;
    for (const tr of traces) for (const v of tr) half = Math.max(half, Math.abs(v - mean));
    half = half > 0 ? half * 1.05 : Math.abs(mean) || 1;
    const ys = siScale(Math.max(Math.abs(mean - half), Math.abs(mean + half)), unit);
    const ts = siScale(Math.max(Math.abs(t[0]), Math.abs(t[t.length - 1])), tunit);

    ctx.clearRect(0, 0, width, height);
    const f = plotFrame(width, height);
    const t0 = t[0] * ts.factor;
    const t1 = t[t.length - 1] * ts.factor;
    const { toX, toY } = drawAxes(ctx, f, { lo: t0, hi: t1, label: axisLabel("Time", ts.unit) },
      { lo: (mean - half) * ys.factor, hi: (mean + half) * ys.factor, label: axisLabel("Amplitude", ys.unit) });

    ctx.save();
    ctx.beginPath();
    ctx.rect(f.left, f.top, f.width, f.height);
    ctx.clip();
    ctx.strokeStyle = "rgba(96,165,250,0.28)";
    ctx.lineWidth = 1;
    for (const tr of traces) {
      ctx.beginPath();
      tr.forEach((v, k) => (k ? ctx.lineTo(toX(t[k] * ts.factor), toY(v * ys.factor)) : ctx.moveTo(toX(t[k] * ts.factor), toY(v * ys.factor))));
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(248,113,113,0.65)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(0), f.top);
    ctx.lineTo(toX(0), f.top + f.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = PLOT_COLORS.text;
    ctx.font = "10.5px Inter, system-ui, sans-serif";
    const us = siScale(ui, tunit);
    ctx.fillText(tunit === "UI" ? "1 UI" : `UI = ${+(ui * us.factor).toPrecision(4)} ${us.unit}`, f.left + 6, f.top + 13);
    ctx.restore();
  }, [t, tunit, ui, traces, unit, width, height]);

  return (
    <div ref={box} className="plot-box">
      <canvas ref={canvas} />
    </div>
  );
}
