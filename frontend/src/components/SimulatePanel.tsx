import { useEffect, useState } from "react";
import type { EditorGraph, ProbeResult } from "../api";
import { useJob } from "../useJob";
import {
  DEFAULT_STOP, FloatingWindow, MetricsTable, ProgressBar, ResourceFields, ResultsToggle, SettingsMenu, StopRuleEditor, fmt, stopConfig,
  type Resources, type StopRule,
} from "./common";

interface Props {
  getGraph: () => EditorGraph;
  blocked?: string;
  resources: Resources;
  /** This mode is the one selected in the top bar. */
  active: boolean;
  /** Scope plots and metrics of the finished run, drawn inside the blocks. */
  onResult: (probes: Record<string, ProbeResult>, metrics: Record<string, number | null>) => void;
}

/** Monte-Carlo run controls for the top bar, with the metrics in a floating window. */
export function SimulatePanel({ getGraph, blocked, resources, active, onResult }: Props) {
  const [rule, setRule] = useState<StopRule>({ ...DEFAULT_STOP, n_batches: 20 });
  const [showResults, setShowResults] = useState(false);
  const { job, error, busy, run, cancel } = useJob();
  const result = job?.result;
  useEffect(() => {
    if (result?.probes || result?.metrics) onResult(result.probes ?? {}, result.metrics ?? {});
  }, [result, onResult]);
  if (!active) return null;

  const status = job ? `${job.status} · ${job.message} · ${fmt(job.elapsed_s)} s${result?.stop_reason ? ` · stop: ${result.stop_reason}` : ""}` : "";

  return (
    <>
      {busy ? (
        <button className="danger" onClick={cancel}>■ Cancel</button>
      ) : (
        <button
          className="primary"
          disabled={!!blocked}
          title={blocked}
          onClick={() => {
            run("simulation", getGraph(), { ...stopConfig(rule), workers: resources.workers });
            setShowResults(true);
          }}
        >
          ▶ Run
        </button>
      )}
      <SettingsMenu title="Simulation settings">
        <StopRuleEditor rule={rule} onChange={setRule} />
        <ResourceFields res={resources} />
      </SettingsMenu>
      <ResultsToggle open={showResults} onToggle={() => setShowResults(!showResults)} error={error ?? job?.error ?? undefined} />
      {job && <ProgressBar value={job.progress} status={job.status} />}
      {showResults && (
        <FloatingWindow id="simulate" title="Simulation results" initial={{ x: 260, y: 70, w: 360, h: 340 }} onClose={() => setShowResults(false)}>
          {job && <div className="muted small float-status">{status}</div>}
          {(error || job?.error) && <div className="field-error">{error ?? job?.error}</div>}
          <MetricsTable metrics={result?.metrics} />
        </FloatingWindow>
      )}
    </>
  );
}
