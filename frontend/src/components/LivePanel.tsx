import type { Probe } from "../api";
import type { LiveState } from "../useLive";
import { MetricsTable, ScopeGrid } from "./common";

interface Props {
  live: LiveState;
  probes: Probe[];
  onStart: () => void;
  onView: (key: string, view: string) => void;
  onRemoveProbe: (key: string) => void;
  blocked?: string;
}

export function LivePanel({ live, probes, onStart, onView, onRemoveProbe, blocked }: Props) {
  return (
    <div className="run-panel">
      <div className="run-controls">
        {live.running ? (
          <button className="danger" onClick={live.stop}>■ Stop</button>
        ) : (
          <button className="primary" onClick={onStart} disabled={!!blocked} title={blocked}>▶ Start live</button>
        )}
        {blocked && <span className="field-error blocked">{blocked}</span>}
        <label className="inline">
          rate
          <input
            type="range"
            min={1}
            max={30}
            value={live.rate}
            onChange={(e) => live.setRate(Number(e.target.value))}
          />
          <span className="mono">{live.rate} Hz</span>
        </label>
        <button onClick={live.resetMetrics} disabled={!live.running}>Reset metrics</button>
        <span className="muted small">
          {live.running
            ? `frame ${live.frame?.batch ?? 0} · ${live.frame?.step_ms ?? "–"} ms/frame`
            : "One frame per step. Parameters changed in the inspector are applied while running."}
        </span>
        {live.error && <span className="field-error">{live.error}</span>}
      </div>
      <div className="run-body">
        <div className="run-side">
          <div className="section-title">Metrics</div>
          <MetricsTable metrics={live.frame?.metrics} />
        </div>
        <ScopeGrid probes={probes} data={live.frame?.probes} onView={onView} onRemove={onRemoveProbe} />
      </div>
    </div>
  );
}
