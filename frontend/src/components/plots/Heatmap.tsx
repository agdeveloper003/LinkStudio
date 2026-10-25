import { useEffect, useRef } from "react";
import { drawAxes, plotFrame } from "./axes";
import { PLOT_COLORS, setupCanvas, useSize } from "./useSize";

function colormap(v: number): [number, number, number] {
  // Dark blue -> teal -> yellow.
  const stops: [number, number, number][] = [[15, 23, 42], [37, 99, 235], [45, 212, 191], [250, 204, 21]];
  const x = Math.min(0.999, Math.max(0, v)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return [0, 1, 2].map((c) => Math.round(stops[i][c] + f * (stops[i + 1][c] - stops[i][c]))) as [number, number, number];
}

interface Props {
  z: number[][];
  xlabel?: string;
  ylabel?: string;
  zunit?: string;
}

/** Matrix plot (rows = OFDM symbols, columns = subcarriers, centred on DC), values in dB. */
export function Heatmap({ z, xlabel = "Column", ylabel = "Row", zunit = "dB" }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || !z.length || width < 60 || height < 60) return;
    const ctx = setupCanvas(canvas.current, width, height);
    if (!ctx) return;
    const rows = z.length;
    const cols = z[0].length;
    const flat = z.flat();
    const max = Math.max(...flat);
    const min = Math.max(Math.min(...flat), max - 40);
    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [R, G, B] = colormap((z[r][c] - min) / (max - min || 1));
        const k = 4 * (r * cols + c);
        img.data[k] = R;
        img.data[k + 1] = G;
        img.data[k + 2] = B;
        img.data[k + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    off.getContext("2d")?.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const f = plotFrame(width - 30, height);
    ctx.imageSmoothingEnabled = false;
    ctx.save(); // first row at the bottom: the vertical axis counts symbols upwards
    ctx.translate(0, 2 * f.top + f.height);
    ctx.scale(1, -1);
    ctx.drawImage(off, f.left, f.top, f.width, f.height);
    ctx.restore();
    drawAxes(ctx, f, { lo: -cols / 2, hi: cols / 2, label: xlabel }, { lo: 0, hi: rows, label: ylabel });
    // Colour bar with its scale: top = maximum, bottom = maximum - range.
    const bx = f.left + f.width + 6;
    for (let k = 0; k < f.height; k++) {
      const [R, G, B] = colormap(1 - k / f.height);
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(bx, f.top + k, 8, 1);
    }
    ctx.save();
    ctx.fillStyle = PLOT_COLORS.text;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.translate(bx + 18, f.top + f.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${min.toFixed(0)} … ${max.toFixed(0)} ${zunit}`, 0, 0);
    ctx.restore();
  }, [z, xlabel, ylabel, zunit, width, height]);

  return (
    <div ref={box} className="plot-box">
      <canvas ref={canvas} />
    </div>
  );
}
