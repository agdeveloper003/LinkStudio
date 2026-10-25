import type { Probe, ProbeResult } from "../api";
import { ProbeView } from "./plots/ProbeView";

export function fmt(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return v.toPrecision(5).replace(/\.?0+$/, "");
}

export function MetricsTable({ metrics }: { metrics?: Record<string, number | null> }) {
  const all = metrics ?? {};
  const entries = Object.entries(all).filter(([k]) => !k.endsWith("_ci_low") && !k.endsWith("_ci_high"));
  if (!entries.length) return <div className="muted small">No metrics. Add a ber_meter, evm_meter or power_meter block.</div>;
  return (
    <table className="metrics">
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k} className={k.endsWith(".ber") || k.endsWith(".evm_pct") ? "highlight" : ""}>
            <td>{k}</td>
            <td>
              {fmt(v)}
              {all[`${k}_ci_low`] != null && all[`${k}_ci_high`] != null && (
                <div className="ci" title="95 % confidence interval (Wilson)">
                  [{fmt(all[`${k}_ci_low`])}, {fmt(all[`${k}_ci_high`])}]
                </div>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ScopeGrid({
  probes,
  data,
  onView,
  onRemove,
}: {
  probes: Probe[];
  data?: Record<string, ProbeResult>;
  onView: (key: string, view: string) => void;
  onRemove: (key: string) => void;
}) {
  if (!probes.length)
    return <div className="muted small scope-hint">No probes: select a block and tick one of its outputs in the inspector.</div>;
  return (
    <div className="scope-grid">
      {probes.map((p) => (
        <ProbeView
          key={p.key}
          name={p.key}
          data={data?.[p.key]}
          onView={(v) => onView(p.key, v)}
          onClose={() => onRemove(p.key)}
        />
      ))}
    </div>
  );
}

export interface StopRule {
  mode: "batches" | "metric";
  n_batches: number;
  metric: string;
  threshold: number;
  max_batches: number;
}

export const DEFAULT_STOP: StopRule = { mode: "batches", n_batches: 10, metric: "ber.bit_errors", threshold: 200, max_batches: 200 };

export function stopConfig(rule: StopRule): Record<string, unknown> {
  return rule.mode === "batches"
    ? { n_batches: rule.n_batches }
    : { metric: rule.metric, threshold: rule.threshold, max_batches: rule.max_batches };
}

export function StopRuleEditor({ rule, onChange }: { rule: StopRule; onChange: (r: StopRule) => void }) {
  return (
    <div className="stop-rule">
      <select value={rule.mode} onChange={(e) => onChange({ ...rule, mode: e.target.value as StopRule["mode"] })}>
        <option value="batches">Fixed batches</option>
        <option value="metric">Until metric ≥ threshold</option>
      </select>
      {rule.mode === "batches" ? (
        <label>
          batches
          <input type="number" min={1} value={rule.n_batches} onChange={(e) => onChange({ ...rule, n_batches: Number(e.target.value) })} />
        </label>
      ) : (
        <>
          <label>
            metric
            <input className="text" value={rule.metric} onChange={(e) => onChange({ ...rule, metric: e.target.value })} />
          </label>
          <label>
            ≥
            <input type="number" value={rule.threshold} onChange={(e) => onChange({ ...rule, threshold: Number(e.target.value) })} />
          </label>
          <label>
            max batches
            <input type="number" min={1} value={rule.max_batches} onChange={(e) => onChange({ ...rule, max_batches: Number(e.target.value) })} />
          </label>
        </>
      )}
    </div>
  );
}

/** Monte-Carlo resources: frames per batch (stored in the scenario) and CPU cores. */
export interface Resources {
  batch: number;
  workers: number;
  maxWorkers: number;
  setBatch: (n: number) => void;
  setWorkers: (n: number) => void;
}

export function ResourceFields({ res }: { res: Resources }) {
  return (
    <div className="stop-rule">
      <label title="Independent frames simulated together in every batch (vectorized). Stored in the scenario.">
        frames/batch
        <input type="number" min={1} value={res.batch} onChange={(e) => res.setBatch(Math.max(1, Math.round(Number(e.target.value))))} />
      </label>
      <label title={`Worker processes (CPU cores, up to ${res.maxWorkers}). Batches or sweep points run in parallel.`}>
        cores
        <input type="number" min={1} max={res.maxWorkers} value={res.workers}
               onChange={(e) => res.setWorkers(Math.min(res.maxWorkers, Math.max(1, Math.round(Number(e.target.value)))))} />
        <span className="muted small">/ {res.maxWorkers}</span>
      </label>
    </div>
  );
}

export function ProgressBar({ value, status }: { value: number; status?: string }) {
  return (
    <div className={`progress ${status ?? ""}`}>
      <div style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}
