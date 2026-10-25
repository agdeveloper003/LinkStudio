// Function-specific bodies of the blocks that are drawn with a display of their own:
// scopes (a digital screen), the sources (bit stream, shift register), the lasers, the
// photodiode receiver, the amplifiers and attenuators, the
// BER meter (status LED and counters), the optical power meter, the mapper (its
// constellation), the MIMO channel (live antenna links) and the fibers.

import { useStore } from "@xyflow/react";
import { useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { BlockActionsContext, RunDataContext } from "../context";
import type { ProbeResult } from "../api";
import type { BlockData } from "../graph";
import { isScopeType } from "../geometry";
import { kindColor } from "../theme";
import { ART, wavelengthColor } from "./BlockArt";
import { ScopePlot } from "./plots/ProbeView";

interface DisplayProps {
  id: string;
  data: BlockData;
  size: { w: number; h: number };
}

const num = (v: unknown, fallback: number) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);

/** Body of a display block, chosen by block type (undefined: generic block). */
export function displayFor(type: string): ((p: DisplayProps) => ReactNode) | undefined {
  if (isScopeType(type)) return undefined; // scopes draw their own frame (ScopeScreen)
  return DISPLAYS[type];
}

const DISPLAYS: Record<string, (p: DisplayProps) => ReactNode> = {
  ber_meter: (p) => <BerDisplay {...p} />,
  mapper: (p) => <MapperDisplay {...p} />,
  mimo_channel: (p) => <MimoDisplay {...p} />,
  fiber: (p) => <FiberDisplay {...p} />,
  multimode_fiber: (p) => <FiberDisplay {...p} />,
  optical_link: (p) => <LinkDisplay {...p} />,
  random_bits: (p) => <RandomBitsDisplay {...p} />,
  prbs: (p) => <PrbsDisplay {...p} />,
  dml: (p) => <DmlDisplay {...p} />,
  vcsel: (p) => <VcselDisplay {...p} />,
  optical_power_meter: (p) => <OpticalPowerMeterDisplay {...p} />,
  photodiode: (p) => <PhotodiodeDisplay {...p} />,
  optical_amplifier: (p) => <OpticalAmplifierDisplay {...p} />,
  optical_attenuator: (p) => <OpticalAttenuatorDisplay {...p} />,
  gain: (p) => <GainDisplay {...p} />,
  agc: (p) => <AgcDisplay {...p} />,
  ...ART,
};

// ------------------------------------------------------------------ scope

/** Sources feeding each input of a node, e.g. {in1: "tx.out"} (re-renders only when they change). */
function useInputSources(id: string): Record<string, string> {
  const key = useStore((s) =>
    s.edges
      .filter((e) => e.target === id)
      .map((e) => `${e.targetHandle}=${e.source}.${e.sourceHandle}`)
      .sort()
      .join("|"),
  );
  return useMemo(() => Object.fromEntries(key ? key.split("|").map((kv) => kv.split("=")) : []), [key]);
}

/** Which scope inputs have a handle: the connected ones plus the first free one. */
export function useScopePorts(id: string, inputs: { name: string }[]): Set<string> {
  const sources = useInputSources(id);
  const free = inputs.find((i) => !sources[i.name]);
  return new Set([...Object.keys(sources), ...(free ? [free.name] : [])]);
}

/** Signal selector of a scope (top-right of its header): shows the displayed signal. */
export function ScopeSelector({ id, data }: { id: string; data: BlockData }) {
  const sources = useInputSources(id);
  const { setParam } = useContext(BlockActionsContext);
  const run = useContext(RunDataContext);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const channel = num(data.params.channel, 1);
  const current = sources[`in${channel}`];
  const connected = data.inputs.filter((i) => sources[i.name]);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const dot = (port: string) => kindColor(run.probes[`${id}.${port}`]?.kind);
  return (
    <div className="scope-selector nodrag nopan" ref={ref}>
      <button className="scope-selector-button" onClick={() => setOpen(!open)} title="Signal on the screen">
        <span className="scope-selector-dot" style={{ background: current ? dot(`in${channel}`) : "#4b5563" }} />
        <span className="scope-selector-text">{current ?? "no signal"}</span>
        <span className="scope-selector-chev">▾</span>
      </button>
      {open && (
        <div className="scope-selector-menu">
          {connected.length === 0 && <div className="scope-selector-empty">Drag an input onto a link.</div>}
          {connected.map((i) => {
            const k = data.inputs.indexOf(i) + 1;
            return (
              <button
                key={i.name}
                className={k === channel ? "active" : ""}
                onClick={() => {
                  setParam(id, "channel", k);
                  setOpen(false);
                }}
              >
                <span className="scope-selector-dot" style={{ background: dot(i.name) }} />
                <span className="scope-ch">CH{k}</span>
                {sources[i.name]}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

const VIEW_LABELS: Record<string, string> = {
  time: "Time",
  spectrum: "Spectrum",
  constellation: "Constellation",
  eye: "Eye diagram",
  histogram: "LLR histogram",
  grid: "Resource grid",
  csi: "Channel response",
};
const ALL_VIEWS = Object.keys(VIEW_LABELS);

/**
 * Plot of the shown input in the chosen view. Live frames carry the chosen view; a
 * Simulate run carries every view of the signal (``alt``), so switching needs no new run.
 */
function useScopePlot(id: string, data: BlockData): { plot?: ProbeResult; connected: boolean } {
  const sources = useInputSources(id);
  const run = useContext(RunDataContext);
  const port = `in${num(data.params.channel, 1)}`;
  const probe = sources[port] ? run.probes[`${id}.${port}`] : undefined;
  const view = typeof data.params.view === "string" ? data.params.view : "auto";
  const alt = probe?.alt?.[view];
  return { plot: probe && alt ? { ...probe, ...alt, view } : probe, connected: !!sources[port] };
}

/**
 * Plot selector of a scope (under its name): the views that suit the shown signal (the
 * kind decides: constellation, spectrum, eye... for samples, histogram for LLRs, ...)
 * and Auto, the natural one.
 */
export function ScopeViewSelector({ id, data }: { id: string; data: BlockData }) {
  const { setParam } = useContext(BlockActionsContext);
  const { plot } = useScopePlot(id, data);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const chosen = typeof data.params.view === "string" ? data.params.view : "auto";
  const shown = plot?.view;

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const options = ["auto", ...(plot?.views ?? ALL_VIEWS)];
  const label = (v: string) => (v === "auto" ? `Auto${shown ? ` · ${VIEW_LABELS[shown] ?? shown}` : ""}` : VIEW_LABELS[v] ?? v);
  return (
    <div className="scope-view nodrag nopan" ref={ref}>
      <button className="scope-view-button" onClick={() => setOpen(!open)} title="Plot of the shown signal">
        {chosen === "auto" ? label("auto") : VIEW_LABELS[shown ?? chosen] ?? chosen}
        <span className="scope-selector-chev">▾</span>
      </button>
      {open && (
        <div className="scope-selector-menu">
          {options.map((v) => (
            <button
              key={v}
              className={v === chosen ? "active" : ""}
              onClick={() => {
                setParam(id, "view", v);
                setOpen(false);
              }}
            >
              {label(v)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** The screen of a scope: the selected input's plot, filling the whole glass. */
export function ScopeScreen({ id, data }: { id: string; data: BlockData }) {
  const { plot, connected } = useScopePlot(id, data);
  return (
    <div className="scope-bezel">
      <div className="scope-glass nowheel">
        <ScopePlot
          data={plot}
          empty={connected ? "Run Live or Simulate to see the signal." : "Drag an input onto a link to tap it."}
        />
      </div>
    </div>
  );
}

// ------------------------------------------------------------- BER meter

function sci(v: number): string {
  if (v === 0) return "0";
  const [m, e] = v.toExponential(2).split("e");
  return `${m}e${Number(e)}`;
}

/** Status LED (steady green, red for a moment whenever new errors arrive), live BER and counters. */
function BerDisplay({ id }: DisplayProps) {
  const { metrics } = useContext(RunDataContext);
  const bits = metrics[`${id}.bits`];
  const errors = metrics[`${id}.bit_errors`];
  const ber = metrics[`${id}.ber`];
  const has = typeof bits === "number" && bits > 0;
  const prev = useRef(0);
  const [hit, setHit] = useState(false);
  useEffect(() => {
    const e = typeof errors === "number" ? errors : 0;
    if (e > prev.current) {
      setHit(true);
      const t = window.setTimeout(() => setHit(false), 450);
      prev.current = e;
      return () => window.clearTimeout(t);
    }
    prev.current = e;
  }, [errors]);
  const state = hit ? "red" : "green";
  return (
    <div className="ber-display">
      <div className="ber-led-col">
        <span className={`ber-led ${state}`} />
        <span className="ber-led-label">{state === "red" ? "ERR" : "OK"}</span>
      </div>
      <div className="ber-readout">
        <div className="ber-main">
          <span className="ber-key">BER</span>
          <span className="ber-value">{has && typeof ber === "number" ? sci(ber) : "—"}</span>
        </div>
        <div className="ber-row">
          <span className="ber-key">bits</span>
          <span>{has ? (bits as number).toLocaleString("en-US") : "—"}</span>
        </div>
        <div className="ber-row">
          <span className="ber-key">errors</span>
          <span className={has && (errors ?? 0) > 0 ? "ber-bad" : ""}>{has ? (errors ?? 0).toLocaleString("en-US") : "—"}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- mapper

/** Ideal constellation points of the mapper parameters (same families as the backend). */
function constellation(p: Record<string, unknown>): { pts: [number, number][]; name: string } {
  const scheme = String(p.scheme ?? "qam");
  const order = num(p.order, 4);
  if (scheme === "psk") {
    const off = ((typeof p.phase_offset_deg === "number" ? p.phase_offset_deg : order === 4 ? 45 : 0) * Math.PI) / 180;
    return {
      pts: Array.from({ length: order }, (_, k) => [Math.cos((2 * Math.PI * k) / order + off), Math.sin((2 * Math.PI * k) / order + off)]),
      name: order === 2 ? "BPSK" : order === 4 ? "QPSK" : `${order}-PSK`,
    };
  }
  if (scheme === "pam") {
    return { pts: Array.from({ length: order }, (_, k) => [2 * k - (order - 1), 0]), name: `${order}-PAM` };
  }
  if (scheme === "custom") {
    const pts = Array.isArray(p.points) ? (p.points as [number, number][]) : [];
    return { pts, name: `custom (${pts.length})` };
  }
  const side = Math.round(Math.sqrt(order));
  const pts: [number, number][] = [];
  for (let a = 0; a < side; a++) for (let b = 0; b < side; b++) pts.push([2 * a - (side - 1), 2 * b - (side - 1)]);
  return { pts: side * side === order ? pts : [], name: `${order}-QAM` };
}

/** The mapper's constellation with a compact readout (bits per symbol, symbol rate). */
function MapperDisplay({ data }: DisplayProps) {
  const { pts, name } = constellation(data.params);
  const lim = Math.max(1e-9, ...pts.flatMap(([i, q]) => [Math.abs(i), Math.abs(q)])) * 1.25;
  const k = Math.log2(Math.max(pts.length, 2));
  const rate = num(data.params.symbol_rate, 0);
  const r = Math.max(0.018, 0.09 / Math.sqrt(pts.length)) * lim;
  return (
    <div className="mapper-display">
      <svg className="mapper-plot" viewBox={`${-lim} ${-lim} ${2 * lim} ${2 * lim}`} preserveAspectRatio="xMidYMid meet">
        <line x1={-lim} y1={0} x2={lim} y2={0} className="mapper-axis" />
        <line x1={0} y1={-lim} x2={0} y2={lim} className="mapper-axis" />
        {pts.map(([i, q], n) => (
          <circle key={n} cx={i} cy={-q} r={r} className="mapper-point" />
        ))}
      </svg>
      <div className="display-readout">
        <b>{name}</b> · {Number.isInteger(k) ? k : k.toFixed(1)} b/sym{rate ? ` · ${siRate(rate)}` : ""}
      </div>
    </div>
  );
}

function siRate(baud: number): string {
  if (baud >= 1e9) return `${+(baud / 1e9).toPrecision(3)} GBd`;
  if (baud >= 1e6) return `${+(baud / 1e6).toPrecision(3)} MBd`;
  if (baud >= 1e3) return `${+(baud / 1e3).toPrecision(3)} kBd`;
  return `${+baud.toPrecision(3)} Bd`;
}

// ---------------------------------------------------------- MIMO channel

/**
 * Wireless MIMO channel drawn as two mirrored antenna arrays and the rays between them,
 * with nothing in the middle: the air is the channel. Each element is a stylized horn
 * ``-<|`` fed from a common bus; live |h| sets the strength of every ray.
 */
function MimoDisplay({ id, data, size }: DisplayProps) {
  const { displays } = useContext(RunDataContext);
  const gains = (displays[id]?.gains as number[][] | undefined) ?? null;
  const nr = gains?.length ?? num(data.params.n_rx, 2);
  const nt = gains?.[0]?.length ?? 2;
  const max = gains ? Math.max(1e-9, ...gains.flat()) : 1;
  const { w, h } = size;
  const mid = 52; // height of the in / out ports (geometry.ts)
  const span = Math.min(h - 12, 36 * Math.max(nt, nr));
  const ys = (n: number) => Array.from({ length: n }, (_, k) => mid - span / 2 + ((k + 0.5) / n) * span);
  const yt = ys(nt);
  const yr = ys(nr);
  const BUS = 10; // feed bus, from the port
  const STEM = 9; // "-"
  const CONE = 11; // "<" length
  const HALF = 7; // half-aperture of the cone
  const faceT = BUS + STEM + CONE;
  const faceR = w - faceT;
  return (
    <svg className="mimo-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {yt.flatMap((ty, t) =>
        yr.map((ry, r) => {
          const g = gains ? gains[r][t] / max : 0.4;
          return (
            <line
              key={`${t}-${r}`}
              x1={faceT + 2}
              y1={ty}
              x2={faceR - 2}
              y2={ry}
              className={`mimo-ray${gains ? " live" : ""}`}
              style={{ strokeOpacity: gains ? 0.15 + 0.85 * g : 0.3, strokeWidth: gains ? 0.7 + 2.5 * g : 1 }}
            />
          );
        }),
      )}
      <AntennaArray x0={0} ys={yt} mid={mid} dir={1} bus={BUS} stem={STEM} cone={CONE} half={HALF} />
      <AntennaArray x0={w} ys={yr} mid={mid} dir={-1} bus={BUS} stem={STEM} cone={CONE} half={HALF} />
      {data.outputs.some((o) => o.name === "csi") && (
        <path className="mimo-csi" d={`M ${w - BUS} ${Math.max(...yr, mid)} V ${mid + 20} H ${w}`} />
      )}
    </svg>
  );
}

/** One antenna array: feed bus from the port, then one ``-<|`` element per antenna (mirrored for RX). */
function AntennaArray({ x0, ys, mid, dir, bus, stem, cone, half }: {
  x0: number; ys: number[]; mid: number; dir: 1 | -1; bus: number; stem: number; cone: number; half: number;
}) {
  const x = (d: number) => x0 + dir * d;
  return (
    <g className="mimo-array">
      <path d={`M ${x(0)} ${mid} H ${x(bus)} M ${x(bus)} ${Math.min(...ys, mid)} V ${Math.max(...ys, mid)}`} />
      {ys.map((y, k) => (
        <g key={k}>
          <path d={`M ${x(bus)} ${y} H ${x(bus + stem)}`} />
          <path className="mimo-horn" d={`M ${x(bus + stem + cone)} ${y - half} L ${x(bus + stem)} ${y} L ${x(bus + stem + cone)} ${y + half}`} />
          <path className="mimo-face" d={`M ${x(bus + stem + cone)} ${y - half - 1} V ${y + half + 1}`} />
        </g>
      ))}
    </g>
  );
}

// ---------------------------------------------------------------- fibers

/** Jacket colours of the fiber classes (TIA-598: yellow SMF, orange OM1/2, aqua OM3/4, lime OM5). */
const JACKET: Record<string, string> = {
  smf: "#facc15", nzdsf: "#facc15", pscf: "#facc15", dcf: "#facc15", custom: "#facc15",
  OM1: "#fb923c", OM2: "#fb923c", OM3: "#22d3ee", OM4: "#22d3ee", OM5: "#a3e635",
};

/** Three turns of fiber resting on the line (tangent at their bottom), close together like a dense winding. */
function Coil({ cx, line, r, pitch, color }: { cx: number; line: number; r: number; pitch: number; color: string }) {
  return (
    <>
      {[-1, 0, 1].map((k) => (
        <circle key={k} cx={cx + k * pitch} cy={line - r} r={r} className="fiber-turn" style={{ stroke: color }} />
      ))}
    </>
  );
}

/** A fiber: a straight line from in to out with its winding, and a light pulse while Live runs. */
function FiberDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const multimode = data.type === "multimode_fiber";
  const color = JACKET[String(data.params.fiber_type ?? (multimode ? "OM4" : "smf"))] ?? "#facc15";
  const { w, h } = size;
  const line = 38; // port height (geometry.ts)
  return (
    <svg className={`fiber-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="fiber-line" d={`M 0 ${line} H ${w}`} style={{ stroke: color }} />
      <Coil cx={w / 2} line={line} r={15} pitch={7} color={color} />
      <path className="fiber-pulse" d={`M 0 ${line} H ${w}`} pathLength={100} />
    </svg>
  );
}

/** Amplified link: per span a winding on the line and an amplifier (the first three spans drawn). */
function LinkDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const spans = Math.min(Math.max(1, num(data.params.n_spans, 10)), 3);
  const { w, h } = size;
  const line = 30; // port height (geometry.ts)
  const step = w / spans;
  return (
    <svg className={`fiber-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="fiber-line" d={`M 0 ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      {Array.from({ length: spans }).map((_, k) => {
        const x = k * step;
        return (
          <g key={k}>
            <Coil cx={x + step * 0.38} line={line} r={11} pitch={5} color="#facc15" />
            <path className="amp" d={`M ${x + step * 0.7} ${line - 8} L ${x + step * 0.7 + 14} ${line} L ${x + step * 0.7} ${line + 8} Z`} />
          </g>
        );
      })}
      <path className="fiber-pulse" d={`M 0 ${line} H ${w}`} pathLength={100} />
    </svg>
  );
}

// --------------------------------------------------------------- sources

/** Small seeded generator, so a picture changes only when the parameters change. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Bits of the same two-state Markov chain as the backend (P(0->1) = t / 2(1 - p1), P(1->0) = t / 2 p1). */
function markovBits(n: number, p1: number, t: number, seed: number): number[] {
  const rnd = mulberry32(seed);
  const up = p1 < 1 ? t / (2 * (1 - p1)) : 1;
  const down = p1 > 0 ? t / (2 * p1) : 1;
  const bits = [rnd() < p1 ? 1 : 0];
  for (let k = 1; k < n; k++) {
    const prev = bits[k - 1];
    bits.push(rnd() < (prev ? down : up) ? 1 - prev : prev);
  }
  return bits;
}

/** NRZ trace of ``bits`` (``pitch`` px per bit) with the digit above each bit. */
function NrzTrace({ bits, pitch, hi, lo }: { bits: number[]; pitch: number; hi: number; lo: number }) {
  let d = `M 0 ${bits[0] ? hi : lo}`;
  bits.forEach((b, k) => {
    d += ` V ${b ? hi : lo} H ${(k + 1) * pitch}`;
  });
  return (
    <>
      <path className="nrz-glow" d={d} />
      <path className="nrz-trace" d={d} />
      {bits.map((b, k) => (
        <text key={k} x={(k + 0.5) * pitch} y={hi - 5} className={`nrz-digit${b ? " one" : ""}`}>
          {b}
        </text>
      ))}
    </>
  );
}

/**
 * Random bits: a logic-analyzer strip of the stream (scrolling while Live runs) and the
 * transition probability, set with the slider or typed in. "i.i.d." clears it (independent bits).
 */
function RandomBitsDisplay({ id, data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const { setParam } = useContext(BlockActionsContext);
  const p1 = Math.min(Math.max(num(data.params.p_one, 0.5), 0), 1);
  const raw = data.params.transition_probability;
  const iid = typeof raw !== "number";
  const limit = 2 * Math.min(p1, 1 - p1);
  const t = iid ? 2 * p1 * (1 - p1) : Math.min(raw as number, limit);
  const [text, setText] = useState(t.toFixed(3));
  useEffect(() => setText(t.toFixed(3)), [t]);
  const n = 48;
  const pitch = 9;
  const bits = useMemo(() => markovBits(n, p1, t, 0x5eed + Math.round(t * 1000) * 7 + Math.round(p1 * 1000)), [p1, t]);
  const set = (v: number) => setParam(id, "transition_probability", Math.round(Math.min(Math.max(v, 0), limit) * 1000) / 1000);
  const commit = () => {
    const v = Number(text.replace(",", "."));
    if (text.trim() && Number.isFinite(v)) set(v);
    else setText(t.toFixed(3));
  };
  const width = size.w - 48;
  return (
    <div className="bits-display">
      <div className="bits-screen">
        <svg width={width} height={44} viewBox={`0 0 ${width} 44`}>
          <g className={`bits-scroll${running ? " running" : ""}`} style={{ ["--shift" as string]: `-${n * pitch}px` }}>
            <NrzTrace bits={[...bits, ...bits]} pitch={pitch} hi={18} lo={38} />
          </g>
        </svg>
      </div>
      <div className="bits-control nodrag nopan">
        <span className="bits-key" title="Transition probability: share of bits that differ from the previous one">
          P<sub>t</sub>
        </span>
        <input
          type="range"
          className="bits-slider"
          min={0}
          max={limit}
          step={0.001}
          value={t}
          style={{ ["--fill" as string]: `${limit ? (t / limit) * 100 : 0}%` }}
          onChange={(e) => set(Number(e.target.value))}
        />
        <input
          className="bits-number"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
        />
        <button
          className={`bits-iid${iid ? " active" : ""}`}
          title="Independent bits (no transition probability)"
          onClick={() => setParam(id, "transition_probability", null)}
        >
          i.i.d.
        </button>
      </div>
      <div className="display-readout">
        p₁ <b>{p1.toFixed(2)}</b> · mean run <b>{t > 0 ? (1 / t).toFixed(1) : "∞"}</b> b · <b>{num(data.params.n_bits, 1024)}</b> b/frame
      </div>
    </div>
  );
}

/** Feedback taps of the PRBS orders (as in the backend): s[k] = s[k - a] xor s[k - b]. */
const PRBS_TAPS: Record<number, [number, number]> = { 7: [7, 6], 9: [9, 5], 11: [11, 9], 15: [15, 14], 20: [20, 3], 23: [23, 18], 31: [31, 28] };

const seedRegister = (order: number) => Array.from({ length: order }, (_, k) => (k % 3 === 0 ? 1 : 0));

/**
 * PRBS: the shift register with its XOR feedback. While Live runs the register steps and
 * the bits shift through it. Registers longer than nine stages are drawn with a gap.
 */
function PrbsDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const order = num(data.params.order, 9);
  const [a, b] = PRBS_TAPS[order] ?? [order, 1];
  const [reg, setReg] = useState<number[]>(() => seedRegister(order));
  useEffect(() => setReg(seedRegister(order)), [order]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setReg((r) => [r[a - 1] ^ r[b - 1], ...r.slice(0, -1)]), 140);
    return () => window.clearInterval(timer);
  }, [running, a, b]);
  const long = order > 9;
  const cells = Math.min(order, 9);
  // Stage shown in each cell: all of them, or the first ones, the tap and the last one.
  const stage = (i: number) => (!long ? i + 1 : i < cells - 2 ? i + 1 : i === cells - 2 ? b : order);
  const width = size.w - 48;
  const cw = Math.min(18, (width - 16) / (cells + (long ? 0.7 : 0)));
  const gap = long ? cw * 0.7 : 0;
  const x = (i: number) => 12 + i * cw + (long && i >= cells - 2 ? gap : 0);
  const tapCell = Array.from({ length: cells }, (_, i) => i).find((i) => stage(i) === b) ?? cells - 2;
  const y0 = 22;
  const mid = y0 + (cw - 2) / 2;
  const xorX = x(0) - 4;
  return (
    <div className="prbs-display">
      <svg width={width} height={50} viewBox={`0 0 ${width} 50`}>
        <path className="prbs-feedback" d={`M ${x(cells - 1) + cw / 2 - 1} ${y0} V 8 H ${xorX + 5}`} />
        <path className="prbs-feedback" d={`M ${x(tapCell) + cw / 2 - 1} ${y0} V 8`} />
        <path className="prbs-feedback" d={`M ${xorX} 13 V ${mid} H ${x(0)}`} />
        <circle className="prbs-xor" cx={xorX} cy={8} r={5} />
        <path className="prbs-xor-cross" d={`M ${xorX - 3.5} 8 H ${xorX + 3.5} M ${xorX} 4.5 V 11.5`} />
        {Array.from({ length: cells }, (_, i) => {
          const bit = reg[stage(i) - 1] ?? 0;
          return (
            <g key={i}>
              <rect className={`prbs-cell${bit ? " one" : ""}`} x={x(i)} y={y0} width={cw - 2} height={cw - 2} rx={2} />
              <text className="prbs-bit" x={x(i) + (cw - 2) / 2} y={mid + 3}>{bit}</text>
            </g>
          );
        })}
        {long && <text className="prbs-gap" x={x(cells - 2) - gap / 2 - 1} y={mid + 3}>⋯</text>}
        <path className="prbs-out" d={`M ${x(cells - 1) + cw - 2} ${mid} H ${width}`} />
      </svg>
      <div className="display-readout">
        <b>PRBS-{order}</b> · x<sup>{a}</sup>+x<sup>{b}</sup>+1 · period <b>{(2 ** order - 1).toLocaleString("en-US")}</b>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- lasers

const fmtNm = (nm: number) => `${+nm.toFixed(1)} nm`;

/** Directly modulated laser: a butterfly package, RF drive on the left, fiber pigtail to the right. */
function DmlDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const nm = num(data.params.wavelength_nm, 1310);
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const [x0, x1] = [26, 118];
  const cx = (x0 + x1) / 2;
  return (
    <svg className={`laser-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ ["--beam" as string]: wavelengthColor(nm) }}>
      <defs>
        <linearGradient id="dml-lid" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f1d78f" />
          <stop offset="0.5" stopColor="#c49a45" />
          <stop offset="1" stopColor="#8f6c29" />
        </linearGradient>
      </defs>
      <path className="rf-lead" d={`M 0 ${line} H ${x0}`} />
      {Array.from({ length: 7 }, (_, k) => {
        const px = x0 + 10 + k * ((x1 - x0 - 20) / 6);
        return <path key={k} className="laser-pin" d={`M ${px} ${line - 18} V ${line - 26} M ${px} ${line + 18} V ${line + 26}`} />;
      })}
      <rect className="laser-body" x={x0} y={line - 18} width={x1 - x0} height={36} rx={3} />
      <rect x={x0 + 4} y={line - 14} width={x1 - x0 - 8} height={28} rx={2} fill="url(#dml-lid)" />
      <g transform={`translate(${cx - 18} ${line})`}>
        <circle className="laser-glow" cx={0} cy={0} r={9} />
        <path className="laser-diode" d="M -6 -6 L 5 0 L -6 6 Z M 5 -6 V 6 M -13 0 H -6 M 5 0 H 11" />
        <path className="laser-ray" d="M 1 -8 L 6 -13 M 4 -13 H 6 V 11 M 6 -7 L 11 -12" />
      </g>
      <text className="laser-etch" x={cx + 14} y={line - 1}>DML</text>
      <text className="laser-etch small" x={cx + 14} y={line + 9}>{fmtNm(nm)}</text>
      <path className="laser-boot" d={`M ${x1} ${line - 6} L ${x1 + 16} ${line - 3} V ${line + 3} L ${x1} ${line + 6} Z`} />
      <path className="fiber-line" d={`M ${x1 + 16} ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      <path className="laser-pulse" d={`M ${x1 + 16} ${line} H ${w}`} pathLength={100} />
    </svg>
  );
}

/**
 * VCSEL in cross-section, laid on its side: contact, bottom DBR, active layer, oxide aperture,
 * top DBR and contact ring; the beam leaves through the aperture into a coupling lens.
 */
function VcselDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const nm = num(data.params.wavelength_nm, 850);
  const color = wavelengthColor(nm);
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const top = line - 22;
  const bottom = line + 22;
  const stripes = (from: number, pairs: number) =>
    Array.from({ length: pairs * 2 }, (_, k) => (
      <rect key={k} className={k % 2 ? "dbr-low" : "dbr-high"} x={from + k * 3} y={top} width={3} height={bottom - top} />
    ));
  const bot = 26; // bottom DBR
  const act = bot + 30; // active layer
  const topDbr = act + 9;
  const ring = topDbr + 18;
  const lens = ring + 36;
  return (
    <svg className={`laser-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ ["--beam" as string]: color }}>
      <defs>
        <linearGradient id="vcsel-beam" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor={color} stopOpacity="0.9" />
          <stop offset="1" stopColor={color} stopOpacity="0.08" />
        </linearGradient>
      </defs>
      <path className="rf-lead" d={`M 0 ${line} H ${bot - 8}`} />
      <rect className="vcsel-contact" x={bot - 8} y={top - 3} width={6} height={bottom - top + 6} rx={1} />
      {stripes(bot - 2, 5)}
      <rect className="vcsel-active" x={act - 2} y={top} width={4} height={bottom - top} />
      <rect className="vcsel-oxide" x={act + 2} y={top} width={5} height={line - 5 - top} />
      <rect className="vcsel-oxide" x={act + 2} y={line + 5} width={5} height={bottom - line - 5} />
      {stripes(topDbr - 2, 3)}
      <rect className="vcsel-contact" x={ring - 2} y={top - 3} width={5} height={line - 6 - top + 3} rx={1} />
      <rect className="vcsel-contact" x={ring - 2} y={line + 6} width={5} height={bottom - line - 6 + 3} rx={1} />
      <path className="vcsel-beam" d={`M ${ring + 3} ${line - 4} L ${lens - 2} ${line - 13} V ${line + 13} L ${ring + 3} ${line + 4} Z`} fill="url(#vcsel-beam)" />
      <ellipse className="vcsel-lens" cx={lens} cy={line} rx={4} ry={15} />
      <path className="vcsel-focus" d={`M ${lens + 4} ${line - 9} L ${lens + 24} ${line} L ${lens + 4} ${line + 9} Z`} />
      <path className="fiber-line" d={`M ${lens + 24} ${line} H ${w}`} style={{ stroke: "#22d3ee" }} />
      <path className="laser-pulse" d={`M ${lens + 24} ${line} H ${w}`} pathLength={100} />
      <text className="laser-etch small vcsel-label" x={(bot + ring) / 2} y={bottom + 13}>VCSEL · {fmtNm(nm)}</text>
    </svg>
  );
}

// ---------------------------------------------------- optical power meter

/**
 * Optical power meter as a handheld instrument: rubber bumper, backlit LCD with the live
 * power in dBm and a level bar, and the connector at the bottom, where the fiber comes in.
 */
function OpticalPowerMeterDisplay({ id, size }: DisplayProps) {
  const { metrics, running } = useContext(RunDataContext);
  const dbm = metrics[`${id}.power_dbm`];
  const osnr = metrics[`${id}.osnr_db`];
  const has = typeof dbm === "number" && Number.isFinite(dbm);
  const { w, h } = size;
  const level = has ? Math.min(Math.max(((dbm as number) + 40) / 50, 0), 1) : 0; // -40 ... +10 dBm
  const segs = 14;
  const segW = (w - 42) / segs;
  const cx = w / 2;
  const casing = h - 26; // bottom of the casing; the connector hangs below it
  return (
    <svg className={`opm-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <defs>
        <linearGradient id="opm-bumper" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#fde047" />
          <stop offset="1" stopColor="#ca8a04" />
        </linearGradient>
        <linearGradient id="opm-lcd" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#bcd8b4" />
          <stop offset="1" stopColor="#8db089" />
        </linearGradient>
      </defs>
      <rect x={13} y={casing - 8} width={w - 26} height={10} rx={3} fill="#a16207" />
      <rect x={2} y={2} width={w - 4} height={casing - 4} rx={16} fill="url(#opm-bumper)" />
      <rect className="opm-face" x={9} y={9} width={w - 18} height={casing - 18} rx={10} />
      <rect className="opm-lcd" x={16} y={16} width={w - 32} height={56} rx={4} fill="url(#opm-lcd)" />
      <text className="opm-small" x={21} y={26}>
        {typeof osnr === "number" && Number.isFinite(osnr) ? `OSNR ${osnr.toFixed(1)}` : "OPM"}
      </text>
      <text className="opm-small" x={w - 21} y={26} textAnchor="end">{running ? "● LIVE" : "HOLD"}</text>
      <text className="opm-digits" x={w - 21} y={50} textAnchor="end">{has ? (dbm as number).toFixed(2) : "--.--"}</text>
      <text className="opm-small" x={w - 21} y={60} textAnchor="end">dBm</text>
      {Array.from({ length: segs }, (_, k) => (
        <rect key={k} className={`opm-seg${k < Math.round(level * segs) ? " on" : ""}`} x={21 + k * segW} y={63} width={segW - 1.5} height={4} />
      ))}
      {["λ", "dB", "⏻"].map((label, k) => (
        <g key={label}>
          <circle className="opm-button" cx={cx + (k - 1) * 28} cy={92} r={8} />
          <text className="opm-button-label" x={cx + (k - 1) * 28} y={95}>{label}</text>
        </g>
      ))}
      <rect className="opm-adapter" x={cx - 13} y={casing - 6} width={26} height={14} rx={2} />
      <rect className="opm-nut" x={cx - 9} y={casing + 8} width={18} height={9} rx={1.5} />
      <rect className="opm-ferrule" x={cx - 3} y={casing + 17} width={6} height={h - casing - 17} rx={1} />
    </svg>
  );
}

// ------------------------------------------------------------ photodiode

const fmtOhm = (r: number) => (r >= 1e3 ? `${+(r / 1e3).toPrecision(3)} kΩ` : `${+r.toPrecision(3)} Ω`);

/**
 * Direct-detection receiver: the fiber ends in a lens that focuses the light on the
 * photodiode (PIN, or APD with its avalanche gain), followed by the transimpedance
 * amplifier with its feedback resistor when one is set.
 */
function PhotodiodeDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const m = num(data.params.apd_gain, 1);
  const tia = typeof data.params.transimpedance_ohm === "number" ? (data.params.transimpedance_ohm as number) : null;
  const pd = 70; // photodiode axis
  const amp = 112; // TIA input
  return (
    <svg className={`rx-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="fiber-line" d={`M 0 ${line} H 26`} style={{ stroke: "#facc15" }} />
      <rect className="rx-ferrule" x={24} y={line - 4} width={10} height={8} rx={1.5} />
      <path className="rx-cone" d={`M 34 ${line - 3} L 44 ${line - 9} V ${line + 9} L 34 ${line + 3} Z`} />
      <ellipse className="vcsel-lens" cx={46} cy={line} rx={3} ry={11} />
      <path className="rx-cone focus" d={`M 49 ${line - 8} L ${pd - 9} ${line} L 49 ${line + 8} Z`} />
      <g className="rx-light">
        <path d={`M ${pd - 19} ${line - 15} L ${pd - 10} ${line - 7}`} />
        <path d={`M ${pd - 13} ${line - 8} L ${pd - 10} ${line - 7} L ${pd - 11} ${line - 10}`} />
        <path d={`M ${pd - 19} ${line + 15} L ${pd - 10} ${line + 7}`} />
        <path d={`M ${pd - 13} ${line + 8} L ${pd - 10} ${line + 7} L ${pd - 11} ${line + 10}`} />
      </g>
      <path className="rx-wire" d={`M ${pd} ${line - 22} V ${line - 7} M ${pd} ${line + 7} V ${line + 22} H ${pd + 8}`} />
      <path className="rx-diode" d={`M ${pd - 7} ${line + 6} L ${pd + 7} ${line + 6} L ${pd} ${line - 6} Z M ${pd - 7} ${line - 6} H ${pd + 7}`} />
      {m > 1 && <path className="rx-avalanche" d={`M ${pd + 10} ${line - 7} l 3 4 l -3 3 l 3 4 l -3 3`} />}
      <text className="rx-label" x={pd} y={line - 25}>{m > 1 ? `APD ×${+m.toPrecision(3)}` : "PIN"}</text>
      <path className="rx-wire" d={`M ${pd} ${line - 22} H ${pd + 22} V ${line} H ${amp}`} />
      <path className="rx-amp" d={`M ${amp} ${line - 12} L ${amp + 22} ${line} L ${amp} ${line + 12} Z`} />
      {tia !== null && (
        <>
          <path className="rx-wire" d={`M ${amp - 6} ${line} V ${line - 22} H ${amp + 4} M ${amp + 18} ${line - 22} H ${amp + 28} V ${line}`} />
          <path className="rx-resistor" d={`M ${amp + 4} ${line - 22} l 2 -3 l 3 6 l 3 -6 l 3 6 l 2 -3`} />
          <text className="rx-label" x={amp + 11} y={line + 23}>{fmtOhm(tia)}</text>
        </>
      )}
      <path className="rf-lead" d={`M ${amp + 22} ${line} H ${w}`} />
    </svg>
  );
}

// ------------------------------------------------- amplifiers / attenuators

/**
 * Optical amplifier: a doped-fiber coil inside the amplifier triangle, pumped through a
 * WDM coupler, with the gain (or the target output power) written on it.
 */
function OpticalAmplifierDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const power = data.params.mode === "power";
  const text = power ? `${num(data.params.target_power_dbm, 0)} dBm` : `+${num(data.params.gain_db, 20)} dB`;
  const [x0, x1] = [38, 112];
  return (
    <svg className={`amp-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="fiber-line" d={`M 0 ${line} H ${x0} M ${x1} ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      <path className="pump" d={`M 22 8 V ${line - 4}`} />
      <path className="pump-head" d={`M 18 ${line - 10} L 22 ${line - 4} L 26 ${line - 10}`} />
      <text className="amp-note" x={18} y={14} textAnchor="end">pump</text>
      <rect className="wdm" x={16} y={line - 4} width={12} height={8} rx={2} />
      <path className="amp-tri" d={`M ${x0} ${line - 28} L ${x1} ${line} L ${x0} ${line + 28} Z`} />
      {[0, 1, 2].map((k) => (
        <circle key={k} className="edf-turn" cx={x0 + 16 + k * 5} cy={line - 9} r={9} />
      ))}
      <text className="amp-gain" x={x0 + 21} y={line + 17}>{text}</text>
    </svg>
  );
}

/**
 * Optical attenuator: an inline barrel with a neutral-density wedge and the loss. As an
 * ideal 1:N splitter it fans out, with the kept branch drawn solid and the others fading.
 */
function OpticalAttenuatorDisplay({ data, size }: DisplayProps) {
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const ways = Math.max(1, Math.round(num(data.params.split_ways, 1)));
  const loss = num(data.params.loss_db, 0) + (ways > 1 ? 10 * Math.log10(ways) + num(data.params.excess_loss_db, 0) : 0);
  const cx = w / 2;
  const db = loss > 0 ? `−${+loss.toFixed(1)} dB` : "0 dB";
  if (ways > 1) {
    const shown = Math.min(ways, 5);
    const ys = Array.from({ length: shown }, (_, k) => line - 24 + ((k + 0.5) / shown) * 48);
    const kept = ys.reduce((best, y) => (Math.abs(y - line) < Math.abs(best - line) ? y : best), ys[0]);
    return (
      <svg className="att-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <path className="fiber-line" d={`M 0 ${line} H ${cx - 18}`} style={{ stroke: "#facc15" }} />
        <rect className="splitter" x={cx - 18} y={line - 27} width={36} height={54} rx={5} />
        {ys.map((y, k) => (
          <path
            key={k}
            className={y === kept ? "split-branch kept" : "split-branch"}
            d={`M ${cx - 12} ${line} C ${cx} ${line}, ${cx} ${y}, ${cx + 18} ${y} H ${y === kept ? w : cx + 34}`}
          />
        ))}
        <path className="fiber-line" d={`M ${cx + 18} ${kept} H ${w}`} style={{ stroke: "#facc15" }} />
        <text className="att-label" x={cx} y={line + 40}>1:{ways} · {db}</text>
      </svg>
    );
  }
  return (
    <svg className="att-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <defs>
        <linearGradient id="nd-wedge" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#e2e8f0" stopOpacity="0.15" />
          <stop offset="1" stopColor="#0f172a" stopOpacity="0.95" />
        </linearGradient>
      </defs>
      <path className="fiber-line" d={`M 0 ${line} H ${cx - 24} M ${cx + 24} ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      <rect className="att-barrel" x={cx - 24} y={line - 11} width={48} height={22} rx={4} />
      <path className="att-ring" d={`M ${cx - 17} ${line - 11} V ${line + 11} M ${cx + 17} ${line - 11} V ${line + 11}`} />
      <path d={`M ${cx - 8} ${line - 9} H ${cx + 8} L ${cx + 4} ${line + 9} H ${cx - 4} Z`} fill="url(#nd-wedge)" className="att-wedge" />
      <text className="att-label" x={cx} y={line + 25}>{db}</text>
    </svg>
  );
}

/** Electrical gain: the amplifier triangle with the gain in dB (a loss when negative). */
function GainDisplay({ data, size }: DisplayProps) {
  const { w, h } = size;
  const line = 32; // port height (geometry.ts)
  const g = num(data.params.gain_db, 0);
  return (
    <svg className="amp-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H 30 M 82 ${line} H ${w}`} />
      <path className={`amp-tri electrical${g < 0 ? " loss" : ""}`} d={`M 30 ${line - 24} L 82 ${line} L 30 ${line + 24} Z`} />
      <text className="amp-gain" x={46} y={line + 3.5}>{g > 0 ? "+" : ""}{+g.toFixed(1)} dB</text>
    </svg>
  );
}

/** Automatic gain control: an amplifier with its level detector feeding the gain back. */
function AgcDisplay({ data, size }: DisplayProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 40; // port height (geometry.ts)
  const [x0, x1] = [30, 82];
  return (
    <svg className={`amp-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${x0} M ${x1} ${line} H ${w}`} />
      <path className="amp-tri electrical" d={`M ${x0} ${line - 24} L ${x1} ${line} L ${x0} ${line + 24} Z`} />
      <text className="amp-gain" x={x0 + 17} y={line + 3.5}>AGC</text>
      <path className="agc-loop" d={`M ${x1 + 14} ${line} V 9 H ${x0 + 30} V ${line - 12}`} />
      <path className="pump-head agc" d={`M ${x0 + 26} ${line - 18} L ${x0 + 30} ${line - 12} L ${x0 + 34} ${line - 18}`} />
      <rect className="agc-detector" x={x1 + 4} y={9} width={20} height={11} rx={2} />
      <text className="amp-note" x={x1 + 14} y={17} textAnchor="middle">|x|²</text>
      <circle className="agc-dot" cx={x1 + 14} cy={line} r={2.4} />
      {data.params.remove_dc !== false && <text className="amp-note" x={x0 - 2} y={line + 34}>DC removed</text>}
    </svg>
  );
}
