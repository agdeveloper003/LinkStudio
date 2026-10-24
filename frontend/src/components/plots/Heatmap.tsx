import { useEffect, useRef } from "react";
import { setupCanvas, useSize } from "./useSize";

function colormap(v: number): [number, number, number] {
  // Dark blue -> teal -> yellow.
  const stops: [number, number, number][] = [[15, 23, 42], [37, 99, 235], [45, 212, 191], [250, 204, 21]];
  const x = Math.min(0.999, Math.max(0, v)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return [0, 1, 2].map((c) => Math.round(stops[i][c] + f * (stops[i + 1][c] - stops[i][c]))) as [number, number, number];
}

/** Matrix plot (rows = OFDM symbols, columns = subcarriers), values in dB. */
export function Heatmap({ z }: { z: number[][] }) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width, height } = useSize(box);

  useEffect(() => {
    if (!canvas.current || !z.length || width < 10 || height < 10) return;
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
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, width, height);
  }, [z, width, height]);

  return (
    <div ref={box} className="plot-box">
      <canvas ref={canvas} />
    </div>
  );
}
