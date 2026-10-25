import { useEffect, useRef, useState, type ReactNode } from "react";

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
  if (!entries.length) return <div className="muted small">No metrics. Add a BER Meter, EVM Meter or Power Meter block (MEASUREMENTS).</div>;
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

/** Gear button in the top bar opening a dropdown with the settings of the current run mode. */
export function SettingsMenu({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div className="menu-anchor" ref={ref}>
      <button className={`icon-toggle${open ? " active" : ""}`} onClick={() => setOpen(!open)} title={title}>⚙</button>
      {open && (
        <div className="run-popover">
          <div className="section-title">{title}</div>
          {children}
        </div>
      )}
    </div>
  );
}

/** Top-bar button showing / hiding the floating results window of a run mode. */
export function ResultsToggle({ open, onToggle, error }: { open: boolean; onToggle: () => void; error?: string }) {
  return (
    <button
      className={`icon-toggle${open ? " active" : ""}${error ? " error" : ""}`}
      onClick={onToggle}
      title={error ? `Error: ${error}` : open ? "Hide results" : "Show results"}
    >
      <svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor">
        <rect x="2" y="8" width="2.6" height="6" rx="0.6" />
        <rect x="6.7" y="4" width="2.6" height="10" rx="0.6" />
        <rect x="11.4" y="1.5" width="2.6" height="12.5" rx="0.6" />
      </svg>
    </button>
  );
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function loadRect(id: string, fallback: Rect): Rect {
  try {
    const r = JSON.parse(window.localStorage.getItem(`linkstudio.window.${id}`) ?? "null") as Rect | null;
    if (r && [r.x, r.y, r.w, r.h].every(Number.isFinite)) return r;
  } catch {
    /* storage unavailable: default placement */
  }
  return fallback;
}

/** Movable (drag the title bar), resizable (bottom-right corner) and closable window over the canvas. */
export function FloatingWindow({
  id,
  title,
  initial,
  onClose,
  children,
}: {
  id: string;
  title: ReactNode;
  initial: Rect;
  onClose: () => void;
  children: ReactNode;
}) {
  const [rect, setRect] = useState(() => loadRect(id, initial));
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(`linkstudio.window.${id}`, JSON.stringify(rect));
    } catch {
      /* not remembered */
    }
  }, [id, rect]);

  // The user resizes through the CSS corner handle: mirror the new size into the state.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      const w = el.offsetWidth;
      const h = el.offsetHeight;
      setRect((r) => (r.w === w && r.h === h ? r : { ...r, w, h }));
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!drag.current) return;
      const x = Math.min(Math.max(e.clientX - drag.current.dx, 0), window.innerWidth - 80);
      const y = Math.min(Math.max(e.clientY - drag.current.dy, 48), window.innerHeight - 40);
      setRect((r) => ({ ...r, x, y }));
    };
    const up = () => (drag.current = null);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, []);

  // Keep the window reachable when the browser window shrinks.
  const x = Math.min(rect.x, Math.max(window.innerWidth - 120, 0));
  const y = Math.min(rect.y, Math.max(window.innerHeight - 60, 48));

  return (
    <div className="float-window" ref={box} style={{ left: x, top: y, width: rect.w, height: rect.h }}>
      <div
        className="float-header"
        onMouseDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          e.preventDefault();
          drag.current = { dx: e.clientX - x, dy: e.clientY - y };
        }}
      >
        <span className="float-title">{title}</span>
        <button className="icon-btn" onClick={onClose} title="Close">×</button>
      </div>
      <div className="float-body">{children}</div>
    </div>
  );
}

/** Header of a side panel with its close button. */
export function PanelTitle({ title, onClose, shortcut }: { title: string; onClose: () => void; shortcut: string }) {
  return (
    <div className="panel-title">
      {title}
      <button className="icon-btn panel-close" onClick={onClose} title={`Close panel (${shortcut})`}>×</button>
    </div>
  );
}
