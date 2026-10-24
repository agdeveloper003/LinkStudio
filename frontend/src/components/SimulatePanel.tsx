import { useState } from "react";
import type { EditorGraph, Probe } from "../api";
import { useJob } from "../useJob";
import { DEFAULT_STOP, MetricsTable, ProgressBar, ScopeGrid, StopRuleEditor, fmt, stopConfig, type StopRule } from "./common";

interface Props {
  getGraph: () => EditorGraph;
  probes: Probe[];
  onView: (key: string, view: string) => void;
  onRemoveProbe: (key: string) => void;
  blocked?: string;
}

export function SimulatePanel({ getGraph, probes, onView, onRemoveProbe, blocked }: Props) {
  const [rule, setRule] = useState<StopRule>({ ...DEFAULT_STOP, n_batches: 20 });
  const { job, error, busy, run, cancel } = useJob();
  const result = job?.result;

  return (
    <div className="run-panel">
      <div className="run-controls">
        {busy ? (
          <button className="danger" onClick={cancel}>■ Cancel</button>
        ) : (
          <button className="primary" disabled={!!blocked} title={blocked} onClick={() => run("simulation", getGraph(), stopConfig(rule))}>▶ Run simulation</button>
        )}
        {blocked && <span className="field-error blocked">{blocked}</span>}
        <StopRuleEditor rule={rule} onChange={setRule} />
        {job && <ProgressBar value={job.progress} status={job.status} />}
        {job && (
          <span className="muted small">
            {job.status} · {job.message} · {fmt(job.elapsed_s)} s{result?.stop_reason ? ` · stop: ${result.stop_reason}` : ""}
          </span>
        )}
        {(error || job?.error) && <span className="field-error">{error ?? job?.error}</span>}
      </div>
      <div className="run-body">
        <div className="run-side">
          <div className="section-title">Metrics</div>
          <MetricsTable metrics={result?.metrics} />
        </div>
        <ScopeGrid
          probes={probes}
          data={result?.probes}
          onView={(k, v) => onView(k, v)}
          onRemove={onRemoveProbe}
        />
      </div>
    </div>
  );
}
