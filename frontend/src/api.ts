// Types and HTTP helpers for the LinkStudio Python API.

export interface PortSpec {
  kinds: string;
  required?: boolean;
  doc?: string;
}

export interface BlockInfo {
  category: string;
  subcategory: string;
  /** Readable name for the UI (the registry key stays the identifier). */
  label: string;
  doc: string;
  inputs: Record<string, PortSpec>;
  outputs: Record<string, PortSpec>;
  params: JsonSchema;
}

export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  anyOf?: JsonSchema[];
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  [key: string]: unknown;
}

export interface Catalogue {
  blocks: Record<string, BlockInfo>;
  theory: Record<string, string>;
}

export interface Probe {
  key: string;
  view?: string;
}

export type Side = "left" | "right" | "top" | "bottom";

/** Where a port sits on the block border: side and position along it (0..1). */
export interface PortPlace {
  side: Side;
  offset: number;
}

export interface EditorBlock {
  type: string;
  params: Record<string, unknown>;
  position?: { x: number; y: number };
  ports?: Record<string, PortPlace>;
  spec?: Record<string, unknown>;
  /** Size of a resized block (scopes and display blocks). */
  size?: { w: number; h: number };
}

export interface EditorGraph {
  name: string;
  description?: string;
  category?: string;
  path?: string;
  seed: number;
  batch_size: number;
  blocks: Record<string, EditorBlock>;
  edges: { source: string; target: string }[];
  probes: Probe[];
  simulation?: Record<string, unknown>;
  sweep?: Record<string, unknown> | null;
  /** External ports of a subsystem file: port name -> "block.port". */
  inputs?: Record<string, string>;
  outputs?: Record<string, string>;
  validation?: Validation;
}

export interface InstancePort {
  name: string;
  kinds: string;
  required?: boolean;
  same_as?: string | null;
}

export interface Validation {
  ok: boolean;
  errors: string[];
  ports: Record<string, { inputs: InstancePort[]; outputs: InstancePort[] }>;
  kinds: Record<string, string>;
}

export interface ScenarioItem {
  path: string;
  name: string;
  category: string;
  description: string;
  subsystem: boolean;
  blocks: number;
}

// Physical values are in base SI units; the UI picks the prefix from the unit.
export type PlotData =
  | { type: "scatter"; i: number[]; q: number[]; unit?: string; ref_i?: number[]; ref_q?: number[] }
  | {
      type: "line"; x: number[]; series: Record<string, (number | null)[]>;
      xlabel?: string; xunit?: string; ylabel?: string; yunit?: string; bars?: boolean;
    }
  | { type: "eye"; t: number[]; tunit: string; ui: number; traces: number[][]; unit?: string }
  | { type: "heatmap"; z: number[][]; xlabel?: string; ylabel?: string; zunit?: string }
  | { type: "message"; text: string };

export type BlockResponse =
  | { available: false; reason: string }
  | {
      available: true; source: "model" | "estimate"; doc: string; f: number[];
      magnitude_db: (number | null)[]; phase_deg: (number | null)[]; group_delay_s: (number | null)[];
    };

/** Plot of a probe; ``alt`` holds the other views of the signal after a Simulate run. */
export type ProbeResult = PlotData & { view?: string; views?: string[]; kind?: string; alt?: Record<string, PlotData> };

export interface Job {
  id: string;
  mode: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  progress: number;
  message: string;
  records: Record<string, number | string | null>[];
  result: {
    metrics?: Record<string, number | null>;
    n_batches?: number;
    elapsed_s?: number;
    stop_reason?: string;
    probes?: Record<string, ProbeResult>;
    param_names?: string[];
    theory?: { label: string; x: number[]; y: number[] };
  } | null;
  error: string | null;
  elapsed_s: number;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string; cpu_count: number }>("/api/health"),
  catalogue: () => request<Catalogue>("/api/blocks"),
  scenarios: () => request<ScenarioItem[]>("/api/scenarios"),
  loadScenario: (path: string) => request<EditorGraph>(`/api/scenarios/${path}`),
  saveScenario: (path: string, graph: EditorGraph) =>
    request<{ path: string }>(`/api/scenarios/${path}`, { method: "PUT", body: JSON.stringify(graph) }),
  validate: (graph: EditorGraph) => request<Validation>("/api/validate", { method: "POST", body: JSON.stringify(graph) }),
  startJob: (body: { mode: string; graph: EditorGraph; config: Record<string, unknown> }) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify(body) }),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  exportPython: (graph: EditorGraph) =>
    request<{ code: string }>("/api/export/python", { method: "POST", body: JSON.stringify(graph) }),
  response: (graph: EditorGraph, block: string) =>
    request<BlockResponse>("/api/response", { method: "POST", body: JSON.stringify({ graph, block }) }),
  cancelJob: (id: string) => request<{ ok: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
};

export function liveSocketUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/api/live`;
}

export function downloadText(filename: string, text: string, type = "text/plain"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
