import { useEffect, useState } from "react";
import { api, type BlockResponse, type EditorGraph } from "../api";
import { LinePlot } from "./plots/LinePlot";

type View = "magnitude" | "phase" | "delay";

interface Props {
  block: string;
  getGraph: () => EditorGraph;
  /** Changes whenever the result may change (parameters, connections). */
  refreshKey: string;
}

/** Transfer function of the selected block (model, or estimated for random channels). */
export function FrequencyResponse({ block, getGraph, refreshKey }: Props) {
  const [res, setRes] = useState<BlockResponse | null>(null);
  const [view, setView] = useState<View>("magnitude");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(async () => {
      setBusy(true);
      try {
        const r = await api.response(getGraph(), block);
        if (!cancelled) setRes(r);
      } catch {
        if (!cancelled) setRes(null);
      } finally {
        if (!cancelled) setBusy(false);
      }
    }, 400);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [block, refreshKey]);

  if (!res?.available) return null;
  const series =
    view === "magnitude"
      ? { label: "|H(f)|", data: res.magnitude_db, ylabel: "Magnitude", yunit: "dB" }
      : view === "phase"
        ? { label: "∠H(f)", data: res.phase_deg, ylabel: "Phase", yunit: "°" }
        : { label: "τg(f)", data: res.group_delay_s, ylabel: "Group delay", yunit: "s" };

  return (
    <div className="section">
      <div className="section-title response-title">
        Frequency response
        {busy && <span className="muted"> · updating</span>}
        <select value={view} onChange={(e) => setView(e.target.value as View)}>
          <option value="magnitude">magnitude</option>
          <option value="phase">phase</option>
          <option value="delay">group delay</option>
        </select>
      </div>
      <div className="response-plot">
        <LinePlot
          x={res.f}
          xlabel="Frequency"
          xunit="Hz"
          ylabel={series.ylabel}
          yunit={series.yunit}
          series={[{ label: series.label, data: series.data, color: "#e879f9", width: 1.5, points: false }]}
        />
      </div>
      <div className="param-help">{res.doc}</div>
    </div>
  );
}
