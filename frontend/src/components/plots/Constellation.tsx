import { useEffect, useRef } from "react";
import { axisLabel, siScale } from "../../units";
import { drawAxes, plotFrame } from "./axes";
import { setupCanvas, useSize } from "./useSize";

interface Props {
  i: number[];
  q: number[];
  unit?: string;
  refI?: number[];
  refQ?: number[];
  color?: string;
}

/** I/Q scatter plot with equal axes, scaled to the signal (SI-prefixed amplitude). */
export function Constellation({ i, q, unit, refI, refQ, color = "#34d399" }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || width < 40 || height < 40) return;
    const ctx = setupCanvas(canvas.current, width, height);
    if (!ctx) return;
    let lim = 0;
    for (let k = 0; k < i.length; k++) lim = Math.max(lim, Math.abs(i[k]), Math.abs(q[k]));
    for (let k = 0; k < (refI?.length ?? 0); k++) lim = Math.max(lim, Math.abs(refI![k]), Math.abs(refQ![k]));
    lim = lim > 0 ? lim * 1.1 : 1;
    const s = siScale(lim, unit);
    const L = lim * s.factor;

    ctx.clearRect(0, 0, width, height);
    const f = plotFrame(width, height, true);
    const { toX, toY } = drawAxes(ctx, f, { lo: -L, hi: L, label: axisLabel("In-phase", s.unit) },
      { lo: -L, hi: L, label: axisLabel("Quadrature", s.unit) });

    ctx.save();
    ctx.beginPath();
    ctx.rect(f.left, f.top, f.width, f.height);
    ctx.clip();
    ctx.fillStyle = color;
    ctx.globalAlpha = i.length > 1500 ? 0.35 : 0.6;
    const r = f.width > 260 ? 1.6 : 1.3;
    for (let k = 0; k < i.length; k++) {
      ctx.beginPath();
      ctx.arc(toX(i[k] * s.factor), toY(q[k] * s.factor), r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (refI && refQ) {
      ctx.strokeStyle = "#f87171";
      ctx.lineWidth = 1.5;
      for (let k = 0; k < refI.length; k++) {
        const x = toX(refI[k] * s.factor);
        const y = toY(refQ[k] * s.factor);
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4);
        ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x - 4, y + 4);
        ctx.lineTo(x + 4, y - 4);
        ctx.stroke();
      }
    }
    ctx.restore();
  }, [i, q, unit, refI, refQ, width, height, color]);

  return (
    <div ref={box} className="plot-box">
      <canvas ref={canvas} />
    </div>
  );
}
