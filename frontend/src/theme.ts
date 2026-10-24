// Colors shared by nodes, edges and plots.

export const KIND_COLORS: Record<string, string> = {
  BITS: "#a78bfa",
  LLR: "#f472b6",
  SYMBOLS: "#34d399",
  SAMPLES: "#60a5fa",
  GRID: "#fbbf24",
  CSI: "#f87171",
  FLAGS: "#94a3b8",
  ANY: "#cbd5e1",
};

export const CATEGORY_COLORS: Record<string, string> = {
  source: "#a78bfa",
  coding: "#c084fc",
  modulation: "#34d399",
  waveform: "#60a5fa",
  channel: "#f87171",
  sync: "#fb923c",
  receiver: "#2dd4bf",
  mimo: "#facc15",
  wireline: "#d97706",
  optical: "#e879f9",
  metrics: "#94a3b8",
  core: "#e2e8f0",
};

export const CATEGORY_ORDER = [
  "source", "coding", "modulation", "waveform", "mimo", "channel", "sync", "receiver", "wireline", "optical", "metrics",
];

export const SERIES_COLORS = ["#6ea8fe", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171", "#2dd4bf", "#fb923c"];

/** Color of a (possibly multi) kind label like "BITS|LLR". */
export function kindColor(label: string | undefined): string {
  if (!label) return KIND_COLORS.ANY;
  const first = label.split("|")[0];
  return KIND_COLORS[first] ?? KIND_COLORS.ANY;
}

/** True when two kind labels share at least one kind. */
export function kindsCompatible(a: string, b: string): boolean {
  if (a === "ANY" || b === "ANY" || a.startsWith("same as") || b.startsWith("same as")) return true;
  const sa = new Set(a.split("|"));
  return b.split("|").some((k) => sa.has(k));
}
