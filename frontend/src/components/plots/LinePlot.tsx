import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { SERIES_COLORS } from "../../theme";
import { useSize } from "./useSize";

export interface LineSeries {
  label: string;
  data: (number | null)[];
  color?: string;
  dashed?: boolean;
  points?: boolean;
  width?: number;
}

interface Props {
  x: number[];
  series: LineSeries[];
  xlabel?: string;
  ylabel?: string;
  logY?: boolean;
  bars?: boolean;
}

const AXIS = {
  stroke: "#8b93a7",
  grid: { stroke: "rgba(148,163,184,0.12)", width: 1 },
  ticks: { stroke: "rgba(148,163,184,0.25)", width: 1 },
  font: "11px Inter, system-ui, sans-serif",
  labelFont: "11px Inter, system-ui, sans-serif",
};

/** Thin wrapper around uPlot that recreates the chart when the series layout changes. */
export function LinePlot({ x, series, xlabel, ylabel, logY = false, bars = false }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const layoutKey = useRef("");
  const { width, height } = useSize(box);

  const data = [
    x,
    ...series.map((s) => (logY ? s.data.map((v) => (v !== null && v > 0 ? v : null)) : s.data)),
  ] as uPlot.AlignedData;

  useEffect(() => {
    if (!box.current || width < 20 || height < 20) return;
    const key = JSON.stringify([series.map((s) => [s.label, s.color, s.dashed, s.points]), xlabel, ylabel, logY, bars]);
    if (chart.current && layoutKey.current === key) {
      chart.current.setData(data);
      chart.current.setSize({ width, height: series.length > 1 ? Math.max(height - 26, 40) : height });
      return;
    }
    chart.current?.destroy();
    layoutKey.current = key;
    const showLegend = series.length > 1;
    const opts: uPlot.Options = {
      width,
      height: showLegend ? Math.max(height - 26, 40) : height,
      legend: { show: showLegend, live: false },
      cursor: { drag: { x: true, y: false } },
      scales: { x: { time: false }, y: logY ? { distr: 3 } : { auto: true } },
      axes: [
        { ...AXIS, label: xlabel, size: 34, labelSize: 14 },
        { ...AXIS, label: ylabel, size: 52, labelSize: 14, values: logY ? (_u, v) => v.map((n) => n.toExponential(0)) : undefined },
      ],
      series: [
        {},
        ...series.map((s, k) => ({
          label: s.label,
          stroke: s.color ?? SERIES_COLORS[k % SERIES_COLORS.length],
          width: s.width ?? 1.6,
          dash: s.dashed ? [6, 4] : undefined,
          spanGaps: false,
          points: { show: s.points ?? (bars || x.length < 40), size: bars ? 6 : 5 },
          paths: bars ? uPlot.paths.bars!({ size: [0.3, 12] }) : undefined,
        })),
      ],
    };
    chart.current = new uPlot(opts, data, box.current);
  }, [data, width, height]);

  useEffect(() => () => chart.current?.destroy(), []);

  return <div ref={box} className="plot-box" />;
}
