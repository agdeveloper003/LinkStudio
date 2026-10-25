import type { ProbeResult } from "../../api";
import { kindColor } from "../../theme";
import { Constellation } from "./Constellation";
import { EyePlot } from "./EyePlot";
import { Heatmap } from "./Heatmap";
import { LinePlot } from "./LinePlot";

interface Props {
  name: string;
  data?: ProbeResult;
  onView?: (view: string) => void;
  onClose?: () => void;
}

/** One scope card: title, view selector and the right plot for the data type. */
export function ProbeView({ name, data, onView, onClose }: Props) {
  const color = kindColor(data?.kind);
  return (
    <div className="scope-card">
      <div className="scope-header">
        <span className="scope-title" style={{ color }}>{name}</span>
        {data?.views && data.views.length > 1 && onView && (
          <select value={data.view} onChange={(e) => onView(e.target.value)}>
            {data.views.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        )}
        {onClose && (
          <button className="icon-btn" title="Remove probe" onClick={onClose}>×</button>
        )}
      </div>
      <div className="scope-body">{render(data, color)}</div>
    </div>
  );
}

/** The plot matching a probe result (constellation, line, eye or heatmap). */
export function ScopePlot({ data, empty = "Waiting for data…" }: { data?: ProbeResult; empty?: string }) {
  return render(data, kindColor(data?.kind), empty);
}

function render(data: ProbeResult | undefined, color: string, empty = "Waiting for data…") {
  if (!data) return <div className="scope-empty">{empty}</div>;
  switch (data.type) {
    case "scatter":
      return <Constellation i={data.i} q={data.q} unit={data.unit} refI={data.ref_i} refQ={data.ref_q} color={color} />;
    case "line":
      return (
        <LinePlot
          x={data.x}
          xlabel={data.xlabel}
          ylabel={data.ylabel}
          xunit={data.xunit}
          yunit={data.yunit}
          bars={data.bars}
          series={Object.entries(data.series).map(([label, values], k) => ({
            label,
            data: values,
            color: k === 0 ? color : undefined,
            width: 1.3,
          }))}
        />
      );
    case "eye":
      return <EyePlot t={data.t} tunit={data.tunit} ui={data.ui} traces={data.traces} unit={data.unit} />;
    case "heatmap":
      return <Heatmap z={data.z} xlabel={data.xlabel} ylabel={data.ylabel} zunit={data.zunit} />;
    default:
      return <div className="scope-empty">{data.type === "message" ? data.text : "Unsupported plot"}</div>;
  }
}
