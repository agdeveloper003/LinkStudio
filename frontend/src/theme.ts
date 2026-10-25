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

/**
 * Functional classes of the block library, in signal-chain order (TX, medium, RX).
 * Complementary classes (encoder / decoder, mapper / demapper...) share a color so
 * the pair stays visually associated while each class is listed on its own.
 */
export const CATEGORIES: { id: string; label: string; color: string }[] = [
  { id: "source", label: "SOURCES", color: "#a78bfa" },
  { id: "encoder", label: "ENCODERS", color: "#f472b6" },
  { id: "bit_processing", label: "BIT PROCESSING", color: "#67e8f9" },
  { id: "mapper", label: "MAPPERS", color: "#34d399" },
  { id: "multiplexer", label: "MULTIPLEXERS", color: "#facc15" },
  { id: "framing", label: "FRAMING", color: "#fb923c" },
  { id: "modulator", label: "MODULATORS", color: "#60a5fa" },
  { id: "filter", label: "FILTERS & RESAMPLERS", color: "#a3e635" },
  { id: "amplifier", label: "AMPLIFIERS & ATTENUATORS", color: "#e879f9" },
  { id: "medium", label: "TRANSMISSION MEDIA", color: "#f87171" },
  { id: "impairment", label: "IMPAIRMENTS", color: "#fda4af" },
  { id: "demodulator", label: "DEMODULATORS", color: "#60a5fa" },
  { id: "synchronization", label: "SYNCHRONIZATION", color: "#fb923c" },
  { id: "equalizer", label: "EQUALIZERS", color: "#2dd4bf" },
  { id: "detector", label: "DETECTORS & ESTIMATORS", color: "#d97706" },
  { id: "demultiplexer", label: "DEMULTIPLEXERS", color: "#facc15" },
  { id: "demapper", label: "DEMAPPERS", color: "#34d399" },
  { id: "decoder", label: "DECODERS", color: "#f472b6" },
  { id: "measurement", label: "MEASUREMENTS", color: "#94a3b8" },
  { id: "core", label: "SUBSYSTEMS", color: "#e2e8f0" },
];

export const CATEGORY_COLORS: Record<string, string> = Object.fromEntries(CATEGORIES.map((c) => [c.id, c.color]));
export const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(CATEGORIES.map((c) => [c.id, c.label]));
export const CATEGORY_ORDER = CATEGORIES.map((c) => c.id);

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
