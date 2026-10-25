import { useMemo, useRef, useState } from "react";
import type { EditorGraph } from "../api";
import { SERIES_COLORS } from "../theme";
import { useJob } from "../useJob";
import { DEFAULT_STOP, ProgressBar, ResourceFields, StopRuleEditor, fmt, stopConfig, type Resources, type StopRule } from "./common";
import { LinePlot, type LineSeries } from "./plots/LinePlot";

interface Props {
  getGraph: () => EditorGraph;
  numericParams: string[];
  theory: Record<string, string>;
  scenarioName: string;
  blocked?: string;
  resources: Resources;
}

interface KeptCurve {
  label: string;
  x: number[];
  y: (number | null)[];
}

interface Row {
  path: string;
  values: string;
}

/** "0:2:10" -> [0, 2, ..., 10];  "1, 2, 4" -> [1, 2, 4]. */
function parseValues(text: string): number[] {
  const t = text.trim();
  if (t.includes(":")) {
    const [a, b, c] = t.split(":").map(Number);
    const [start, step, stop] = c === undefined ? [a, 1, b] : [a, b, c];
    if (![start, step, stop].every(Number.isFinite) || step === 0) return [];
    const out: number[] = [];
    for (let v = start; step > 0 ? v <= stop + 1e-9 : v >= stop - 1e-9; v += step) out.push(+v.toFixed(10));
    return out.slice(0, 500);
  }
  return t.split(/[,\s]+/).filter(Boolean).map(Number).filter(Number.isFinite);
}

/** Put several (x, y) curves on one sorted x grid (uPlot needs aligned data). */
function alignCurves(curves: KeptCurve[]): { x: number[]; ys: (number | null)[][] } {
  const x = Array.from(new Set(curves.flatMap((c) => c.x))).sort((a, b) => a - b);
  const ys = curves.map((c) => {
    const m = new Map(c.x.map((xv, i) => [xv, c.y[i]]));
    return x.map((xv) => m.get(xv) ?? null);
  });
  return { x, ys };
}

export function SweepPanel({ getGraph, numericParams, theory, scenarioName, blocked, resources }: Props) {
  const [rows, setRows] = useState<Row[]>([{ path: "", values: "0:2:10" }]);
  const [rule, setRule] = useState<StopRule>({ ...DEFAULT_STOP, mode: "metric" });
  const [metric, setMetric] = useState("");
  const [logY, setLogY] = useState(true);
  const [theoryKey, setTheoryKey] = useState("");
  const [showTable, setShowTable] = useState(false);
  const [kept, setKept] = useState<KeptCurve[]>([]);
  const plotBox = useRef<HTMLDivElement>(null);
  const { job, error, busy, run, cancel } = useJob();

  const defaultPath = numericParams.find((p) => p.endsWith(".snr_db")) ?? numericParams[0] ?? "";
  const effRows = rows.map((r) => ({ ...r, path: r.path || defaultPath }));
  const records = job?.records ?? [];
  const paramNames = job?.result?.param_names ?? effRows.map((r) => r.path);

  const metricKeys = useMemo(() => {
    if (!records.length) return [];
    return Object.keys(records[0]).filter(
      (k) => !paramNames.includes(k) && k !== "elapsed_s" && !k.includes("_ci_") && typeof records[0][k] === "number",
    );
  }, [records, paramNames]);
  const yKey = metric && metricKeys.includes(metric) ? metric : metricKeys.find((k) => k.endsWith(".ber")) ?? metricKeys[0] ?? "";

  const current = useMemo((): KeptCurve[] => {
    if (!records.length || !yKey) return [];
    const xKey = paramNames[0];
    const groupKey = paramNames[1];
    const xs = Array.from(new Set(records.map((r) => Number(r[xKey])))).sort((a, b) => a - b);
    const groups = groupKey ? Array.from(new Set(records.map((r) => String(r[groupKey])))) : [""];
    return groups.map((g) => ({
      label: groupKey ? `${groupKey}=${g}` : yKey,
      x: xs,
      y: xs.map((x) => {
        const r = records.find((rec) => Number(rec[xKey]) === x && (!groupKey || String(rec[groupKey]) === g));
        const v = r?.[yKey];
        return typeof v === "number" ? v : null;
      }),
    }));
  }, [records, yKey, paramNames]);

  const plot = useMemo(() => {
    const th = job?.result?.theory;
    const curves: KeptCurve[] = [...current, ...kept];
    if (th) curves.push({ label: th.label, x: th.x, y: th.y });
    if (!curves.length) return null;
    const { x, ys } = alignCurves(curves);
    const series: LineSeries[] = curves.map((c, k) => {
      const isKept = k >= current.length && k < current.length + kept.length;
      const isTheory = th && k === curves.length - 1;
      return {
        label: c.label,
        data: ys[k],
        color: isTheory ? "#e2e8f0" : SERIES_COLORS[k % SERIES_COLORS.length],
        dashed: Boolean(isTheory) || isKept,
        points: !isTheory,
        width: isTheory ? 1.2 : 1.6,
      };
    });
    return { x, series, xKey: paramNames[0] ?? "" };
  }, [current, kept, paramNames, job?.result?.theory]);

  const keepCurve = () => {
    if (!current.length) return;
    const tag = window.prompt("Label for the kept curve:", scenarioName) ?? scenarioName;
    setKept((k) => [...k, ...current.map((c) => ({ ...c, label: `${tag}: ${c.label}` }))]);
  };

  const downloadPng = () => {
    const canvas = plotBox.current?.querySelector("canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = "sweep.png";
    a.click();
  };

  const start = () => {
    const params: Record<string, number[]> = {};
    for (const r of effRows) if (r.path) params[r.path] = parseValues(r.values);
    run("sweep", getGraph(), { params, theory: theoryKey || undefined, ...stopConfig(rule), workers: resources.workers });
  };

  const downloadCsv = () => {
    if (!records.length) return;
    const cols = Object.keys(records[0]);
    const csv = [cols.join(","), ...records.map((r) => cols.map((c) => r[c] ?? "").join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "sweep.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="run-panel">
      <div className="run-controls">
        {busy ? (
          <button className="danger" onClick={cancel}>■ Cancel</button>
        ) : (
          <button className="primary" onClick={start} disabled={!defaultPath || !!blocked} title={blocked}>▶ Run sweep</button>
        )}
        {blocked && <span className="field-error blocked">{blocked}</span>}
        <StopRuleEditor rule={rule} onChange={setRule} />
        <ResourceFields res={resources} />
        {job && <ProgressBar value={job.progress} status={job.status} />}
        {job && <span className="muted small">{job.status} · {job.message} · {fmt(job.elapsed_s)} s</span>}
        {(error || job?.error) && <span className="field-error">{error ?? job?.error}</span>}
      </div>
      <div className="run-body">
        <div className="run-side sweep-side">
          <div className="section-title">Parameters</div>
          {effRows.map((r, i) => (
            <div key={i} className="sweep-row">
              <select value={r.path} onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, path: e.target.value } : x)))}>
                {numericParams.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <input
                className="text"
                value={r.values}
                title="start:step:stop or comma-separated values"
                onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, values: e.target.value } : x)))}
              />
              {rows.length > 1 && (
                <button className="icon-btn" onClick={() => setRows(rows.filter((_, j) => j !== i))}>×</button>
              )}
            </div>
          ))}
          {rows.length < 2 && (
            <button className="link" onClick={() => setRows([...rows, { path: "", values: "4, 16" }])}>
              + second parameter (one curve per value)
            </button>
          )}
          <div className="section-title">Plot</div>
          <label className="inline">
            y
            <select value={yKey} onChange={(e) => setMetric(e.target.value)}>
              {metricKeys.length === 0 && <option value="">(run first)</option>}
              {metricKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
          <label className="inline">
            <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log scale
          </label>
          <label className="inline" title="Theory curves are BER vs Eb/N0: sweep an AWGN block in 'ebn0' mode.">
            theory
            <select value={theoryKey} onChange={(e) => setTheoryKey(e.target.value)}>
              <option value="">none</option>
              {Object.entries(theory).map(([k, label]) => (
                <option key={k} value={k}>{label}</option>
              ))}
            </select>
          </label>
          <div className="button-row">
            <button onClick={() => setShowTable(!showTable)} disabled={!records.length}>
              {showTable ? "Hide table" : "Show table"}
            </button>
            <button onClick={downloadCsv} disabled={!records.length}>CSV</button>
            <button onClick={downloadPng} disabled={!plot}>PNG</button>
          </div>
          <div className="button-row">
            <button onClick={keepCurve} disabled={!current.length} title="Keep this curve to compare with the next runs">Keep curve</button>
            <button onClick={() => setKept([])} disabled={!kept.length}>Clear kept ({kept.length})</button>
          </div>
        </div>
        <div className="sweep-main">
          <div className="sweep-plot" ref={plotBox}>
            {plot ? (
              <LinePlot x={plot.x} series={plot.series} xlabel={plot.xKey} ylabel={yKey} logY={logY} />
            ) : (
              <div className="scope-empty">Choose a parameter and value range, then run the sweep.</div>
            )}
          </div>
          {records.length > 0 && showTable && (
            <div className="sweep-table">
              <table className="metrics">
                <thead>
                  <tr>{Object.keys(records[0]).map((c) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {records.map((r, i) => (
                    <tr key={i}>{Object.keys(records[0]).map((c) => <td key={c}>{fmt(r[c] as number)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
