// SI prefixes and axis ticks for physical quantities.

const PREFIXES: [number, string][] = [
  [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""],
  [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"], [1e-15, "f"],
];

/** Units that take SI prefixes (dB-like and dimensionless units do not). */
export function prefixable(unit: string | undefined): boolean {
  return !!unit && ["Hz", "s", "A", "V", "W", "√W", "m"].includes(unit);
}

export interface Scale {
  factor: number; // multiply raw values by this
  unit: string; // prefixed unit, e.g. "GHz"
}

/** Prefix that keeps |values| <= maxAbs between 1 and 1000 (for prefixable units). */
export function siScale(maxAbs: number, unit: string | undefined): Scale {
  if (!prefixable(unit) || !isFinite(maxAbs) || maxAbs <= 0) return { factor: 1, unit: unit ?? "" };
  for (const [p, name] of PREFIXES) {
    if (maxAbs >= p * 0.9995) return { factor: 1 / p, unit: name + unit };
  }
  const [p, name] = PREFIXES[PREFIXES.length - 1];
  return { factor: 1 / p, unit: name + unit };
}

export function maxAbs(values: Iterable<number | null>): number {
  let m = 0;
  for (const v of values) if (v !== null && isFinite(v)) m = Math.max(m, Math.abs(v));
  return m;
}

/** Axis label "Quantity [unit]" (unit omitted when empty). */
export function axisLabel(quantity: string | undefined, unit: string | undefined): string {
  if (!quantity) return unit ? `[${unit}]` : "";
  return unit ? `${quantity} [${unit}]` : quantity;
}

/** About ``count`` round tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Compact tick text: enough decimals to tell neighbouring ticks apart. */
export function tickText(v: number, ticks: number[]): string {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v) || 1;
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}
