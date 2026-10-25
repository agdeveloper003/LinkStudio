import { useState } from "react";
import type { LiveState } from "../useLive";
import { FloatingWindow, MetricsTable, ResultsToggle, SettingsMenu } from "./common";

interface Props {
  live: LiveState;
  onStart: () => void;
  blocked?: string;
  /** This mode is the one selected in the top bar. */
  active: boolean;
}

/** Live mode controls for the top bar, with the metrics in a floating window. */
export function LivePanel({ live, onStart, blocked, active }: Props) {
  const [showResults, setShowResults] = useState(false);
  if (!active) return null;

  return (
    <>
      {live.running ? (
        <button className="danger" onClick={live.stop}>■ Stop</button>
      ) : (
        <button
          className="primary"
          onClick={() => {
            onStart();
            setShowResults(true);
          }}
          disabled={!!blocked}
          title={blocked ?? "One frame per step. Parameters changed in the inspector are applied while running."}
        >
          ▶ Start
        </button>
      )}
      <SettingsMenu title="Live settings">
        <label className="inline">
          rate
          <input type="range" min={1} max={30} value={live.rate} onChange={(e) => live.setRate(Number(e.target.value))} />
          <span className="mono">{live.rate} Hz</span>
        </label>
        <button onClick={live.resetMetrics} disabled={!live.running}>Reset metrics</button>
        <p className="muted small">One frame per step. Parameters changed in the inspector are applied while running.</p>
      </SettingsMenu>
      <ResultsToggle open={showResults} onToggle={() => setShowResults(!showResults)} error={live.error ?? undefined} />
      {showResults && (
        <FloatingWindow id="live" title="Live metrics" initial={{ x: 260, y: 70, w: 340, h: 320 }} onClose={() => setShowResults(false)}>
          <div className="muted small mono float-status">
            {live.running ? `frame ${live.frame?.batch ?? 0} · ${live.frame?.step_ms ?? "–"} ms/frame` : "stopped"}
          </div>
          {live.error && <div className="field-error">{live.error}</div>}
          <MetricsTable metrics={live.frame?.metrics} />
        </FloatingWindow>
      )}
    </>
  );
}
