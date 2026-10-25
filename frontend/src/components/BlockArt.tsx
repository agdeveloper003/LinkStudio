// Drawings of the signal-processing blocks: coding, mapping, framing, modulation,
// filtering, impairments, channels, synchronization, equalization and detection.
//
// Framed blocks (header + drawing) draw in a fixed 140 x 56 viewBox that scales with the
// block; the accent colour is the class colour (--c, set by BlockView). Frameless blocks
// (the physical ones: adders, mixers, modulators, cables, radio and free-space channels)
// draw in node pixels, with their ports on the line of the drawing (geometry.ts).

import { useContext, useMemo, type ReactNode } from "react";
import { RunDataContext } from "../context";
import type { BlockData } from "../graph";

interface ArtProps {
  id: string;
  data: BlockData;
  size: { w: number; h: number };
}

const num = (v: unknown, fallback: number) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);
const VW = 140;
const VH = 56;

type Pt = [number, number];
const poly = (pts: Pt[]) => pts.map(([x, y], k) => `${k ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
/** Path of y = f(t), t in [0, 1], mapped to x in [x0, x1]. */
const plot = (f: (t: number) => number, x0: number, x1: number, n = 64): string =>
  poly(Array.from({ length: n + 1 }, (_, k) => [x0 + ((x1 - x0) * k) / n, f(k / n)] as Pt));
const sinc = (x: number) => (Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

/** Raised-cosine pulse (the drawings of rrc / rc / matched shapes). */
function raisedCosine(t: number, beta: number): number {
  const d = 1 - (2 * beta * t) ** 2;
  return Math.abs(d) < 1e-6 ? (Math.PI / 4) * sinc(1 / (2 * beta)) : (sinc(t) * Math.cos(Math.PI * beta * t)) / d;
}

/** Colour of a wavelength: its visible hue, or a red-to-violet glow across the infrared bands. */
export function wavelengthColor(nm: number): string {
  if (nm >= 380 && nm <= 700) {
    const hue = nm < 440 ? 270 : nm < 490 ? 240 - ((nm - 440) / 50) * 60 : nm < 580 ? 180 - ((nm - 490) / 90) * 120 : 60 - ((nm - 580) / 120) * 60;
    return `hsl(${hue}, 95%, 60%)`;
  }
  const f = Math.min(Math.max((nm - 700) / 900, 0), 1); // 700 nm red ... 1600 nm violet
  return `hsl(${355 - f * 85}, 90%, 64%)`;
}

/** Body of a framed block: the drawing (scaled to fit) and an optional readout line. */
function Art({ children, caption, vh = VH }: { children: ReactNode; caption?: ReactNode; vh?: number }) {
  return (
    <div className="art">
      <svg className="art-svg" viewBox={`0 0 ${VW} ${vh}`} preserveAspectRatio="xMidYMid meet">
        {children}
      </svg>
      {caption != null && <div className="display-readout">{caption}</div>}
    </div>
  );
}

/** Arrow from (x0, y) to (x1, y). */
const Arrow = ({ x0, x1, y, cls = "a-mute" }: { x0: number; x1: number; y: number; cls?: string }) => (
  <path className={`${cls} a-arrow`} d={`M ${x0} ${y} H ${x1} M ${x1 - 3.5} ${y - 2.5} L ${x1} ${y} L ${x1 - 3.5} ${y + 2.5}`} />
);

/** Row of bit cells; ``kind(k)`` picks the style of cell k ("" data, "on" accent, "off" dropped, "bad" error). */
function Cells({ x, y, n, w, h = 9, kind = () => "", text }: {
  x: number; y: number; n: number; w: number; h?: number; kind?: (k: number) => string; text?: (k: number) => string;
}) {
  const cw = w / n;
  return (
    <g>
      {Array.from({ length: n }, (_, k) => (
        <g key={k}>
          <rect className={`a-cell ${kind(k)}`} x={x + k * cw + 0.4} y={y} width={cw - 0.8} height={h} rx={1} />
          {text && <text className="a-tiny" x={x + k * cw + cw / 2} y={y + h / 2 + 2}>{text(k)}</text>}
        </g>
      ))}
    </g>
  );
}

const XorNode = ({ x, y, r = 4 }: { x: number; y: number; r?: number }) => (
  <g>
    <circle className="a-node" cx={x} cy={y} r={r} />
    <path className="a-line thin" d={`M ${x - r * 0.65} ${y} H ${x + r * 0.65} M ${x} ${y - r * 0.65} V ${y + r * 0.65}`} />
  </g>
);

// ================================================================== coding

/** Codeword picture: k data cells become n cells with parity (encoder), or back (decoder). */
function CodewordArt({ k, n, decode, caption }: { k: number; n: number; decode?: boolean; caption: ReactNode }) {
  const scale = Math.min(1, 16 / n);
  const kn = Math.max(1, Math.round(k * scale));
  const nn = Math.max(kn + 1, Math.round(n * scale));
  const short = <Cells x={decode ? 102 : 2} y={22} n={kn} w={Math.min(36, kn * 9)} h={13} />;
  const long = (
    <Cells
      x={decode ? 2 : 52}
      y={22}
      n={nn}
      w={Math.min(86, nn * 8)}
      h={13}
      kind={(c) => (c >= kn ? "on" : decode && c === 1 ? "bad" : "")}
    />
  );
  return (
    <Art caption={caption}>
      {short}
      {long}
      {decode ? <Arrow x0={90} x1={99} y={28.5} /> : <Arrow x0={40} x1={49} y={28.5} />}
      <text className="a-small" x={decode ? 2 : 52} y={16}>{decode ? "received" : "codeword"}</text>
      <text className="a-small" x={decode ? 102 : 2} y={16}>data</text>
      {decode && <path className="a-ok" d="M 112 44 l 3 3 l 6 -7" />}
    </Art>
  );
}

const rate = (k: number, n: number) => {
  const g = (a: number, b: number): number => (b ? g(b, a % b) : a);
  const d = g(k, n) || 1;
  return `${k / d}/${n / d}`;
};

function HammingArt({ data }: ArtProps) {
  const m = num(data.params.m, 3);
  const n = 2 ** m - 1;
  const k = n - m;
  return <CodewordArt k={k} n={n} decode={data.type.endsWith("decoder")} caption={<><b>Hamming ({n},{k})</b> · r {rate(k, n)}</>} />;
}

function RepetitionArt({ data }: ArtProps) {
  const n = Math.max(1, Math.round(num(data.params.n, 3)));
  const decode = data.type.endsWith("decoder");
  const shown = Math.min(n, 8);
  const ys = Array.from({ length: shown }, (_, k) => 8 + ((k + 0.5) / shown) * 40);
  const x0 = decode ? 104 : 30;
  const x1 = decode ? 30 : 104;
  return (
    <Art caption={<><b>×{n}</b> · r 1/{n}{decode ? " · majority" : ""}</>}>
      <rect className="a-cell on" x={x0 - 5} y={23} width={10} height={10} rx={1.5} />
      {ys.map((y, k) => (
        <g key={k}>
          <path className="a-line thin" d={`M ${x0 + (decode ? -5 : 5)} 28 C 67 28, 67 ${y}, ${x1 + (decode ? 5 : -5)} ${y}`} />
          <rect className={`a-cell${decode && k === 1 ? " bad" : ""}`} x={x1 - 4} y={y - 4} width={8} height={8} rx={1.2} />
        </g>
      ))}
    </Art>
  );
}

/** Shift register of a convolutional encoder with its two modulo-2 adders. */
function ConvArt({ data }: ArtProps) {
  const K = Math.max(2, Math.round(num(data.params.constraint_length, 7)));
  const gens = (Array.isArray(data.params.generators) ? data.params.generators : [171, 133]) as number[];
  const cells = Math.min(K, 8);
  const cw = 92 / cells;
  const taps = gens.map((g) => parseInt(String(g), 8).toString(2).padStart(K, "0").slice(-K));
  const tapAt = (t: string, i: number) => t[Math.round((i * (K - 1)) / Math.max(cells - 1, 1))] === "1";
  return (
    <Art caption={<><b>K={K}</b> · ({gens.join(", ")})<sub>8</sub> · r 1/{gens.length}</>}>
      <Arrow x0={2} x1={20} y={28} />
      <Cells x={22} y={23} n={cells} w={92} h={10} />
      {taps.slice(0, 2).map((t, g) => {
        const y = g ? 50 : 6;
        return (
          <g key={g}>
            {Array.from({ length: cells }, (_, i) =>
              tapAt(t, i) ? <path key={i} className="a-line thin" d={`M ${22 + i * cw + cw / 2} ${g ? 33 : 23} V ${y}`} /> : null,
            )}
            <path className="a-line thin" d={`M ${22 + cw / 2} ${y} H 122`} />
            <XorNode x={124} y={y} />
            <Arrow x0={128} x1={138} y={y} cls="a-line" />
          </g>
        );
      })}
    </Art>
  );
}

/** Viterbi decoder: a trellis with the survivor path. */
function TrellisArt({ data }: ArtProps) {
  const K = Math.round(num(data.params.constraint_length, 7));
  const states = 4;
  const stages = 7;
  const x = (s: number) => 8 + s * 20;
  const y = (q: number) => 10 + q * 12;
  const survivor = [0, 1, 3, 2, 1, 2, 0, 0];
  return (
    <Art caption={<><b>Viterbi</b> · {2 ** (K - 1)} states · K={K}</>}>
      {Array.from({ length: stages - 1 }, (_, s) =>
        Array.from({ length: states }, (_, q) => [((q << 1) & 3), ((q << 1) & 3) | 1].map((r) => (
          <path key={`${s}-${q}-${r}`} className="a-mute thin" d={`M ${x(s)} ${y(q)} L ${x(s + 1)} ${y(r)}`} />
        ))),
      )}
      <path className="a-line" d={poly(survivor.slice(0, stages).map((q, s) => [x(s), y(q)] as Pt))} />
      {Array.from({ length: stages }, (_, s) =>
        Array.from({ length: states }, (_, q) => <circle key={`${s}${q}`} className="a-dot" cx={x(s)} cy={y(q)} r={1.6} />),
      )}
    </Art>
  );
}

/** LDPC: Tanner graph between variable nodes (circles) and check nodes (squares). */
function LdpcArt({ data }: ArtProps) {
  const kb = num(data.params.k_blocks, 8);
  const mb = num(data.params.m_blocks, 8);
  const z = num(data.params.z, 48);
  const wc = Math.max(1, Math.round(num(data.params.col_weight, 3)));
  const decode = data.type.endsWith("decoder");
  const nv = 12;
  const nc = 6;
  const vx = (i: number) => 10 + (i * 120) / (nv - 1);
  const cx = (j: number) => 20 + (j * 100) / (nc - 1);
  const edges = useMemo(() => {
    const out: [number, number][] = [];
    for (let i = 0; i < nv; i++) for (let w = 0; w < Math.min(wc, nc); w++) out.push([i, (i * 7 + w * 5 + (i >> 2)) % nc]);
    return out;
  }, [wc]);
  const n = (kb + mb) * z;
  const k = kb * z;
  return (
    <Art caption={decode
      ? <><b>LDPC</b> · {String(data.params.method ?? "min_sum").replace("_", "-")} · {num(data.params.max_iter, 30)} it</>
      : <><b>LDPC ({n},{k})</b> · r {rate(k, n)}</>}>
      {edges.map(([i, j], e) => (
        <path key={e} className={decode ? "a-line thin pulse" : "a-mute thin"} d={`M ${vx(i)} 44 L ${cx(j)} 12`} />
      ))}
      {Array.from({ length: nc }, (_, j) => <rect key={j} className="a-cell on" x={cx(j) - 3.5} y={7} width={7} height={7} rx={1} />)}
      {Array.from({ length: nv }, (_, i) => <circle key={i} className="a-node" cx={vx(i)} cy={47} r={3.4} />)}
    </Art>
  );
}

/** Polar code: the butterfly network of the transform, frozen inputs dimmed. */
function PolarArt({ data }: ArtProps) {
  const N = num(data.params.block_length, 256);
  const k = num(data.params.k, 128);
  const decode = data.type.endsWith("decoder");
  const n = 8;
  const y = (i: number) => 6 + i * 6.2;
  const frozen = new Set([0, 1, 2, 4]);
  const stageX = [30, 62, 94];
  return (
    <Art caption={<><b>Polar ({N},{k})</b> · {decode ? "SC decoding" : `r ${rate(k, N)}`}</>}>
      {Array.from({ length: n }, (_, i) => (
        <g key={i}>
          <path className="a-mute thin" d={`M 14 ${y(i)} H 126`} />
          <text className={`a-tiny${frozen.has(i) ? " dim" : ""}`} x={8} y={y(i) + 2}>{frozen.has(i) ? "0" : "u"}</text>
        </g>
      ))}
      {stageX.map((sx, s) => {
        const span = 1 << s;
        return Array.from({ length: n }, (_, i) =>
          (i & span) === 0 ? (
            <g key={`${s}-${i}`}>
              <path className="a-line thin" d={`M ${sx} ${y(i)} V ${y(i + span)}`} />
              <XorNode x={sx} y={y(i)} r={2.6} />
              <circle className="a-dot" cx={sx} cy={y(i + span)} r={1.3} />
            </g>
          ) : null,
        );
      })}
    </Art>
  );
}

const CRC_BITS: Record<string, number> = { crc8: 8, crc16: 16, crc24a: 24, crc32: 32 };

function CrcArt({ data }: ArtProps) {
  const poly = String(data.params.polynomial ?? "crc16");
  const bits = CRC_BITS[poly] ?? 16;
  const decode = data.type.endsWith("decoder");
  return (
    <Art caption={<><b>{poly.toUpperCase()}</b> · {bits}-bit check{decode ? " · valid flag" : ""}</>}>
      <Cells x={6} y={22} n={12} w={84} h={11} />
      <Cells x={90} y={22} n={4} w={34} h={11} kind={() => (decode ? "on ok" : "on")} />
      <text className="a-small" x={6} y={16}>payload</text>
      <text className="a-small" x={90} y={16}>CRC</text>
      {decode && <path className="a-ok" d="M 127 25 l 3 3 l 6 -7" />}
      <path className="a-mute thin" d="M 6 40 C 40 50, 80 50, 107 36" />
      <path className="a-mute thin" d="M 103 36.5 L 107 36 L 106 40" />
    </Art>
  );
}

/** 64b/66b, 4b/5b: the payload grows by the header or the extra symbol bits. */
function LineBlockArt({ data }: ArtProps) {
  const code = String(data.params.code ?? "64b66b");
  const [a, b] = (code.match(/\d+/g) ?? ["64", "66"]).map(Number);
  const decode = data.type.endsWith("decoder");
  return (
    <Art caption={<><b>{a}b/{b}b</b> · overhead {(((b - a) / a) * 100).toFixed(1)}%</>}>
      <text className="a-small" x={6} y={14}>{decode ? `${b}-bit block` : `${a} bits`}</text>
      <text className="a-small" x={6} y={52}>{decode ? `${a} bits` : `${b}-bit block`}</text>
      <rect className="a-cell" x={6} y={decode ? 32 : 17} width={110} height={9} rx={1.5} />
      <rect className="a-cell" x={6} y={decode ? 17 : 32} width={110} height={9} rx={1.5} />
      <rect className="a-cell on" x={decode ? 6 : 6} y={decode ? 17 : 32} width={Math.max(6, (110 * (b - a)) / b)} height={9} rx={1.5} />
      <text className="a-tiny" x={6 + Math.max(6, (110 * (b - a)) / b) / 2} y={(decode ? 17 : 32) + 6.5}>{b - a === 2 ? "01" : "+"}</text>
      <path className="a-mute a-arrow" d={`M 124 ${decode ? 21 : 36} V ${decode ? 36 : 21} M 121.5 ${decode ? 32.5 : 24.5} L 124 ${decode ? 36 : 21} L 126.5 ${decode ? 32.5 : 24.5}`} />
    </Art>
  );
}

function PunctureArt({ data }: ArtProps) {
  const pattern = (Array.isArray(data.params.pattern) ? data.params.pattern : [1, 1, 1, 0, 0, 1]) as number[];
  const decode = data.type === "depuncturer";
  const kept = pattern.filter(Boolean).length;
  const n = Math.min(pattern.length * 2, 16);
  const p = (k: number) => pattern[k % pattern.length];
  return (
    <Art caption={<>pattern <b>{pattern.join("")}</b> · keeps {kept}/{pattern.length}{decode ? " · erasures" : ""}</>}>
      <Cells x={8} y={14} n={n} w={124} h={11} kind={(k) => (p(k) ? "" : decode ? "off dashed" : "")} />
      <Cells x={8} y={34} n={n} w={124} h={11} kind={(k) => (p(k) ? "" : decode ? "on" : "off")} text={(k) => (p(k) ? "" : decode ? "0" : "×")} />
    </Art>
  );
}

/** Alamouti: the 2 x 2 space-time block over two antennas and two symbol times. */
function AlamoutiArt({ data }: ArtProps) {
  const decode = data.type.endsWith("decoder");
  return (
    <Art caption={decode ? <><b>Alamouti</b> · linear combining</> : <><b>Alamouti</b> · 2 TX · r 1</>}>
      <path className="a-mute" d="M 36 8 h -4 v 40 h 4 M 104 8 h 4 v 40 h -4" />
      <text className="a-math" x={52} y={24}>s₁</text>
      <text className="a-math" x={84} y={24}>−s₂*</text>
      <text className="a-math" x={52} y={42}>s₂</text>
      <text className="a-math" x={84} y={42}>s₁*</text>
      <text className="a-small" x={116} y={22}>ant 1</text>
      <text className="a-small" x={116} y={40}>ant 2</text>
      <text className="a-small" x={46} y={54}>t</text>
      <text className="a-small" x={78} y={54}>t+T</text>
    </Art>
  );
}

// ========================================================== bit processing

/** Small deterministic generator (mulberry32), to draw a permutation from a seed. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Interleaver. Random: the permutation (drawn from the seed) spreads a burst of three
 * adjacent bits apart, which is what the block is for. Block: the matrix written row by
 * row and read column by column (inverse: the other way round), numbered in write order.
 */
function InterleaverArt({ data }: ArtProps) {
  const random = data.params.kind !== "block";
  const inverse = data.params.inverse === true;
  const seed = num(data.params.seed, 1234);
  const n = 8;
  const perm = useMemo(() => {
    const rnd = seeded(seed);
    const p = Array.from({ length: n }, (_, k) => k);
    for (let k = n - 1; k > 0; k--) {
      const j = Math.floor(rnd() * (k + 1));
      [p[k], p[j]] = [p[j], p[k]];
    }
    return p; // output k takes input p[k]
  }, [seed]);
  const burst = (j: number) => j >= 3 && j <= 5;
  const cw = 120 / n;
  const cx = (k: number) => 10 + (k + 0.5) * cw;
  if (random) {
    return (
      <Art caption={<><b>random</b> · seed {seed}{inverse ? " · inverse" : ""}</>}>
        <Cells x={10} y={5} n={n} w={120} h={9} kind={(j) => (burst(j) ? "bad" : "")} />
        <Cells x={10} y={42} n={n} w={120} h={9} kind={(k) => (burst(perm[k]) ? "bad" : "on")} />
        {perm.map((j, k) => (
          <path key={k} className={burst(j) ? "a-warn-line" : "a-line thin"} d={`M ${cx(j)} 14 C ${cx(j)} 28, ${cx(k)} 28, ${cx(k)} 42`} />
        ))}
      </Art>
    );
  }
  const R = 4;
  const C = 5;
  const rows = num(data.params.rows, 16);
  const cell = 13;
  const x0 = 42;
  const y0 = 5;
  // Numbers in write order: rows (interleaver) or columns (inverse).
  const order = (r: number, c: number) => (inverse ? c * R + r : r * C + c) + 1;
  const first = (r: number, c: number) => (inverse ? r === 0 : c === 0); // read first
  return (
    <Art caption={<><b>block</b> · {rows} rows{inverse ? " · inverse" : ""}</>}>
      {Array.from({ length: R }, (_, r) =>
        Array.from({ length: C }, (_, c) => (
          <g key={`${r}-${c}`}>
            <rect className={`a-cell ${first(r, c) ? "on" : ""}`} x={x0 + c * cell + 0.4} y={y0 + r * 11} width={cell - 0.8} height={10} rx={1} />
            <text className="a-tiny" x={x0 + c * cell + cell / 2} y={y0 + r * 11 + 7}>{order(r, c)}</text>
          </g>
        )),
      )}
      {inverse ? (
        <>
          <path className="a-mute a-arrow" d={`M ${x0 - 6} ${y0} V ${y0 + R * 11 - 2} M ${x0 - 8.5} ${y0 + R * 11 - 5.5} L ${x0 - 6} ${y0 + R * 11 - 2} L ${x0 - 3.5} ${y0 + R * 11 - 5.5}`} />
          <text className="a-small" x={6} y={y0 + 22}>write</text>
          <Arrow x0={x0} x1={x0 + C * cell} y={y0 + R * 11 + 4} cls="a-line" />
          <text className="a-small" x={x0 + C * cell + 4} y={y0 + R * 11 + 6}>read</text>
        </>
      ) : (
        <>
          <Arrow x0={8} x1={x0 - 3} y={y0 + 5} />
          <text className="a-small" x={8} y={y0 + 16}>write</text>
          <path className="a-line a-arrow" d={`M ${x0 + C * cell + 6} ${y0} V ${y0 + R * 11 - 2} M ${x0 + C * cell + 3.5} ${y0 + R * 11 - 5.5} L ${x0 + C * cell + 6} ${y0 + R * 11 - 2} L ${x0 + C * cell + 8.5} ${y0 + R * 11 - 5.5}`} />
          <text className="a-small" x={x0 + C * cell + 11} y={y0 + 22}>read</text>
        </>
      )}
    </Art>
  );
}

/**
 * Additive scrambler: the LFSR (its real taps and seed) feeds back through an XOR and
 * its sequence is XOR-ed with the data; a run of zeros comes out as the scrambled bits.
 */
function ScramblerArt({ data }: ArtProps) {
  const taps = (Array.isArray(data.params.taps) && data.params.taps.length ? data.params.taps : [7, 4]).map(Number);
  const deg = Math.max(...taps, 1);
  const seed = num(data.params.seed, 0b1011101);
  const cells = Math.min(deg, 8);
  // State oldest bit first (MSB of the seed), as the backend; s[k] = XOR_t s[k - t].
  const { state, seq } = useMemo(() => {
    const s = Array.from({ length: deg }, (_, k) => (seed >> (deg - 1 - k)) & 1);
    const state = [...s].reverse(); // newest first, as drawn (left = newest)
    for (let k = 0; k < 8; k++) s.push(taps.reduce((acc, t) => acc ^ s[s.length - t], 0));
    return { state, seq: s.slice(deg) };
  }, [deg, seed, taps.join()]);
  const rx = 70;
  const rw = 64;
  const cw = rw / cells;
  const tapCell = (t: number) => Math.min(cells - 1, Math.round(((t - 1) / Math.max(deg - 1, 1)) * (cells - 1)));
  const poly = [...taps].sort((a, b) => b - a);
  return (
    <Art caption={<>{poly.map((t, k) => <span key={k}>x<sup>{t}</sup> + </span>)}1 · seed {seed}</>}>
      <Cells x={rx} y={3} n={cells} w={rw} h={11} text={(k) => String(state[k] ?? 0)} kind={(k) => (taps.some((t) => tapCell(t) === k) ? "on" : "")} />
      {taps.map((t) => (
        <path key={t} className="a-line thin" d={`M ${rx + (tapCell(t) + 0.5) * cw} 14 V 20 H 60`} />
      ))}
      <path className="a-line thin" d="M 60 20 V 12.5" />
      <XorNode x={60} y={8.5} />
      <Arrow x0={64} x1={rx} y={8.5} cls="a-line" />
      <path className="a-line thin" d="M 60 20 V 31.5" />
      <circle className="a-dot accent" cx={60} cy={20} r={1.4} />
      <Cells x={4} y={30} n={6} w={48} h={11} text={() => "0"} />
      <Arrow x0={52} x1={56} y={35.5} />
      <XorNode x={60} y={35.5} />
      <Arrow x0={64} x1={68} y={35.5} cls="a-line" />
      <Cells x={68} y={30} n={6} w={48} h={11} text={(k) => String(seq[k])} kind={(k) => (seq[k] ? "on" : "")} />
      <text className="a-small" x={4} y={51}>data</text>
      <text className="a-small" x={68} y={51}>scrambled</text>
    </Art>
  );
}

// ================================================================ mapping

function DemapperArt({ data }: ArtProps) {
  const order = num(data.params.order, 4);
  const side = Math.max(2, Math.round(Math.sqrt(order)));
  const method = String(data.params.method ?? "maxlog");
  const pts: Pt[] = [];
  for (let a = 0; a < side; a++) for (let b = 0; b < side; b++) pts.push([48 + ((a + 0.5) / side) * 44, 4 + ((b + 0.5) / side) * 48]);
  return (
    <Art caption={<><b>{method === "hard" ? "hard" : method === "exact" ? "exact LLR" : "max-log LLR"}</b>{data.params.scheme ? ` · ${order}-${String(data.params.scheme).toUpperCase()}` : " · upstream"}</>}>
      <rect className="a-screen" x={48} y={4} width={44} height={48} rx={3} />
      {Array.from({ length: side - 1 }, (_, k) => (
        <g key={k}>
          <path className="a-line dash" d={`M ${48 + ((k + 1) / side) * 44} 4 V 52`} />
          <path className="a-line dash" d={`M 48 ${4 + ((k + 1) / side) * 48} H 92`} />
        </g>
      ))}
      {pts.map(([x, y], k) => <circle key={k} className="a-dot accent" cx={x} cy={y} r={1.8} />)}
      <circle className="a-dot warn" cx={pts[0][0] + 6} cy={pts[0][1] + 4} r={1.6} />
      <path className="a-mute thin" d={`M ${pts[0][0] + 6} ${pts[0][1] + 4} L ${pts[0][0]} ${pts[0][1]}`} />
      <Arrow x0={98} x1={112} y={28} />
      <text className="a-math" x={124} y={31}>Λ</text>
    </Art>
  );
}

/** Line code levels of ``bits`` (nrz, manchester, ami, mlt3), as [t, level] steps. */
function lineCode(code: string, bits: number[]): Pt[] {
  const out: Pt[] = [];
  let ami = 1;
  let mlt = 0;
  const seq = [0, 1, 0, -1];
  let m = 0;
  bits.forEach((b, k) => {
    if (code === "manchester") {
      out.push([k, b ? 1 : -1], [k + 0.5, b ? -1 : 1]);
    } else if (code === "ami") {
      if (b) ami = -ami;
      out.push([k, b ? ami : 0]);
    } else if (code === "mlt3") {
      if (b) m = (m + 1) % 4;
      mlt = seq[m];
      out.push([k, mlt]);
    } else out.push([k, b ? 1 : -1]);
  });
  return out;
}

const BITS = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1];

function LineCodeArt({ data }: ArtProps) {
  const code = String(data.params.code ?? "nrz");
  const decode = data.type === "line_decoder";
  const steps = lineCode(code, BITS);
  const x = (t: number) => 6 + t * 12.8;
  const y = (v: number) => 30 - v * 14;
  let d = `M ${x(0)} ${y(steps[0][1])}`;
  steps.forEach(([, v], k) => {
    const next = steps[k + 1]?.[0] ?? BITS.length;
    d += ` V ${y(v)} H ${x(next)}`;
  });
  return (
    <Art caption={decode
      ? <><b>{code.toUpperCase()}</b> decoding · threshold {num(data.params.threshold, 0.5)}</>
      : <><b>{code.toUpperCase()}</b>{data.params.symbol_rate ? ` · ${+(num(data.params.symbol_rate, 1e9) / 1e9).toPrecision(3)} GBd` : ""}</>}>
      <path className="a-mute thin dash" d={`M 6 30 H 134`} />
      {decode && <path className="a-warn-line dash" d={`M 6 ${y(0.5)} H 134 M 6 ${y(-0.5)} H 134`} />}
      <path className="a-line" d={d} />
      {BITS.map((b, k) => (
        <text key={k} className="a-tiny" x={x(k + 0.5)} y={54}>{b}</text>
      ))}
      {decode && BITS.map((_, k) => <circle key={k} className="a-dot accent" cx={x(k + 0.5)} cy={y(steps.find(([t]) => t <= k + 0.5 && t >= k)?.[1] ?? 0)} r={1.4} />)}
    </Art>
  );
}

// ================================================================ framing

function PreambleArt({ data }: ArtProps) {
  const L = num(data.params.length, 64);
  const g = num(data.params.guard, 16);
  return (
    <Art caption={<>preamble <b>{L}</b> · guard <b>{g}</b></>}>
      <rect className="a-cell off dashed" x={6} y={20} width={16} height={16} rx={1.5} />
      <rect className="a-cell on" x={22} y={20} width={46} height={16} rx={1.5} />
      <path className="a-hatch" d={plot((t) => 28 + 5 * Math.sign(Math.sin(t * 37)), 24, 66, 80)} />
      <rect className="a-cell" x={68} y={20} width={66} height={16} rx={1.5} />
      <text className="a-small" x={8} y={14}>guard</text>
      <text className="a-small" x={30} y={14}>known</text>
      <text className="a-small" x={88} y={14}>payload</text>
    </Art>
  );
}

/** Correlation against the known preamble: the peak marks the frame start. */
function FrameSyncArt({ data }: ArtProps) {
  const ofdm = data.type === "ofdm_sync";
  const peak = 62;
  return (
    <Art caption={ofdm
      ? <><b>Schmidl-Cox</b>{data.params.correct_cfo !== false ? " · CFO corrected" : ""}</>
      : <>search <b>±{num(data.params.search, 32)}</b>{data.params.correct_cfo !== false ? " · CFO corrected" : ""}</>}>
      <path className="a-mute thin" d="M 6 48 H 134" />
      <path
        className="a-line"
        d={ofdm
          ? plot((t) => 48 - 34 * Math.max(0, Math.min(1, (t * 128 - 40) / 10, (90 - t * 128) / 10)) - 2 * Math.sin(t * 70), 6, 134)
          : plot((t) => 48 - 4 * Math.abs(Math.sin(t * 53)) - 36 * Math.exp(-(((t * 128 + 6 - peak) / 2.4) ** 2)), 6, 134, 160)}
      />
      <path className="a-warn-line" d={`M ${ofdm ? 52 : peak} 4 V 12 M ${(ofdm ? 52 : peak) - 3} 8 L ${ofdm ? 52 : peak} 12 L ${(ofdm ? 52 : peak) + 3} 8`} />
      <text className="a-small" x={(ofdm ? 52 : peak) + 6} y={10}>start</text>
    </Art>
  );
}

// =========================================================== pulse shaping

function shapeFn(shape: string, beta: number, bt: number): (t: number) => number {
  if (shape === "gaussian") {
    const s = Math.sqrt(Math.log(2)) / (2 * Math.PI * bt);
    return (t) => Math.exp(-(t ** 2) / (2 * (s * 2 * Math.PI) ** 2) * 4);
  }
  if (shape === "rect") return (t) => (Math.abs(t) <= 0.5 ? 1 : 0);
  if (shape === "none") return (t) => (Math.abs(t) < 0.05 ? 1 : 0);
  return (t) => raisedCosine(t, Math.max(beta, 0.01));
}

function PulseArt({ data }: ArtProps) {
  const matched = data.type === "matched_filter";
  const shape = String(data.params.shape ?? (matched ? "matched" : "rrc"));
  const beta = num(data.params.rolloff, 0.25);
  const sps = data.params.sps == null ? null : num(data.params.sps, 8);
  const f = shapeFn(shape, beta, num(data.params.bt, 0.3));
  const span = 4;
  const x = (t: number) => 70 + t * 14;
  const y = (v: number) => 44 - v * 34;
  const label = shape === "matched" ? "matched" : shape.toUpperCase();
  return (
    <Art caption={<><b>{label}</b>{["rrc", "rc", "matched"].includes(shape) ? ` · β ${beta}` : shape === "gaussian" ? ` · BT ${num(data.params.bt, 0.3)}` : ""}{sps ? ` · ${sps} sps` : ""}{matched && data.params.decimate !== false ? " · ↓" : ""}</>}>
      <path className="a-mute thin" d="M 8 44 H 132" />
      <path className="a-fill" d={`${plot((t) => y(f((t - 0.5) * 2 * span)), x(-span), x(span), 120)} V 44 H ${x(-span)} Z`} />
      <path className="a-line" d={plot((t) => y(f((t - 0.5) * 2 * span)), x(-span), x(span), 120)} />
      {Array.from({ length: 2 * span + 1 }, (_, k) => k - span).map((k) => (
        <g key={k}>
          <path className="a-mute thin" d={`M ${x(k)} 44 V ${y(f(k))}`} />
          <circle className="a-dot accent" cx={x(k)} cy={y(f(k))} r={1.5} />
        </g>
      ))}
      {matched && (
        <g>
          <path className="a-line thin" d="M 124 22 L 132 17" />
          <circle className="a-dot" cx={124} cy={22} r={1.3} />
          <circle className="a-dot" cx={133} cy={22} r={1.3} />
        </g>
      )}
    </Art>
  );
}

// ============================================================ multicarrier

/** Subcarrier comb: data (accent), pilots (warm) and the nulls (DC, guards). */
function CombArt({ data }: ArtProps) {
  const demod = data.type.endsWith("demodulator");
  const nfft = num(data.params.n_fft, 64);
  const active = num(data.params.n_active, 52);
  const pilot = num(data.params.pilot_spacing, 6);
  const cp = num(data.params.cp_len, 16);
  const bars = 32;
  const act = Math.max(2, Math.round((active / nfft) * bars));
  const lo = Math.floor((bars - act) / 2);
  return (
    <Art caption={<><b>{data.params.n_fft == null ? "auto" : nfft}-{demod ? "FFT" : "IFFT"}</b> · {active} active · CP {cp}</>}>
      {Array.from({ length: bars }, (_, k) => {
        const on = k >= lo && k < lo + act && k !== bars / 2;
        const isPilot = on && pilot > 0 && (k - lo) % Math.max(1, Math.round(pilot / 2)) === 0;
        const h = on ? 22 + 6 * Math.sin(k * 1.7) * (demod ? 1 : 0.2) : 3;
        return <rect key={k} className={`a-bar${on ? (isPilot ? " pilot" : " on") : ""}`} x={6 + k * 4} y={40 - h} width={2.6} height={h} rx={0.6} />;
      })}
      <path className="a-mute thin" d="M 4 40.5 H 136" />
      <text className="a-tiny" x={70} y={50}>f</text>
    </Art>
  );
}

function DmtArt({ data }: ArtProps) {
  const demod = data.type === "dmt_demodulator";
  const water = data.params.loading !== "uniform";
  const first = num(data.params.first_tone, 33);
  const last = num(data.params.last_tone, 255);
  const bars = 28;
  return (
    <Art caption={<><b>DMT</b> · tones {first}–{last}{demod ? " · per-tone FEQ" : water ? " · water-filling" : ` · ${num(data.params.uniform_bits, 4)} b/tone`}</>}>
      {Array.from({ length: bars }, (_, k) => {
        const b = water ? Math.max(1, Math.round(12 * Math.exp(-k / 11) + 1)) : num(data.params.uniform_bits, 4);
        return <rect key={k} className="a-bar on" x={8 + k * 4.5} y={44 - b * 3} width={3.2} height={b * 3} rx={0.5} />;
      })}
      {water && <path className="a-warn-line dash" d={plot((t) => 44 - 36 * Math.exp(-t * 2.6) - 3, 8, 134)} />}
      <path className="a-mute thin" d="M 6 44.5 H 136" />
      <text className="a-tiny" x={4} y={10}>bits</text>
    </Art>
  );
}

// ============================================================== filtering

function FirArt({ data }: ArtProps) {
  const taps = (Array.isArray(data.params.taps) ? data.params.taps : [1]) as number[];
  const shown = taps.slice(0, 24);
  const max = Math.max(1e-9, ...shown.map(Math.abs));
  const step = Math.min(14, 120 / Math.max(shown.length, 1));
  const x0 = 70 - ((shown.length - 1) * step) / 2;
  return (
    <Art caption={<><b>FIR</b> · {taps.length} tap{taps.length === 1 ? "" : "s"}</>}>
      <path className="a-mute thin" d="M 6 34 H 134" />
      {shown.map((v, k) => {
        const yy = 34 - (v / max) * 24;
        return (
          <g key={k}>
            <path className="a-line thin" d={`M ${x0 + k * step} 34 V ${yy}`} />
            <circle className="a-dot accent" cx={x0 + k * step} cy={yy} r={1.8} />
          </g>
        );
      })}
      <text className="a-tiny" x={130} y={44}>n</text>
    </Art>
  );
}

function QuantizerArt({ data }: ArtProps) {
  const bits = num(data.params.bits, 8);
  const steps = Math.min(2 ** bits, 10);
  let d = "M 30 50";
  for (let k = 0; k < steps; k++) {
    const x = 30 + ((k + 1) / steps) * 80;
    const y = 50 - ((k + 1) / steps) * 44;
    d += ` H ${x} V ${y}`;
  }
  return (
    <Art caption={<><b>{bits}-bit</b> ADC · {2 ** bits} levels</>}>
      <path className="a-mute thin dash" d="M 30 50 L 110 6" />
      <path className="a-line" d={d} />
      <path className="a-mute thin" d="M 26 50 H 116 M 30 54 V 4" />
    </Art>
  );
}

function IqImbalanceArt({ data }: ArtProps) {
  const a = num(data.params.amplitude_db, 0);
  const ph = num(data.params.phase_deg, 0);
  const g = 10 ** (a / 20);
  const phi = ((ph || 8) * Math.PI) / 180;
  const shown = a === 0 && ph === 0;
  const c: Pt = [70, 28];
  const map = ([i, q]: Pt): Pt => [c[0] + 18 * (i * g * Math.cos(phi / 2) + q * Math.sin(phi / 2)), c[1] - 18 * (q * Math.cos(phi / 2) + i * g * Math.sin(phi / 2) * 0.2)];
  const ideal: Pt[] = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
  return (
    <Art caption={<>ε <b>{a} dB</b> · φ <b>{ph}°</b>{data.params.dc_i || data.params.dc_q ? " · DC" : ""}</>}>
      <path className="a-mute thin" d="M 40 28 H 100 M 70 4 V 52" />
      <path className="a-mute thin dash" d={`${poly(ideal.map(([i, q]) => [c[0] + 18 * i, c[1] - 18 * q] as Pt))} Z`} />
      {!shown && <path className="a-line" d={`${poly(ideal.map(map))} Z`} />}
      {(shown ? ideal.map(([i, q]) => [c[0] + 18 * i, c[1] - 18 * q] as Pt) : ideal.map(map)).map(([x, y], k) => (
        <circle key={k} className="a-dot accent" cx={x} cy={y} r={2} />
      ))}
    </Art>
  );
}

function JitterArt({ data }: ArtProps) {
  const rj = num(data.params.rj_rms_ui, 0.02);
  const sj = num(data.params.sj_amplitude_ui, 0);
  const spread = Math.min(14, 4 + rj * 200 + sj * 40);
  return (
    <Art caption={<>RJ <b>{rj} UI</b>{sj ? <> · SJ <b>{sj} UI</b></> : ""}</>}>
      {[-1, -0.5, 0, 0.5, 1].map((o, k) => (
        <path key={k} className="a-line thin" style={{ opacity: 1 - Math.abs(o) * 0.6 }} d={`M 10 40 H ${50 + o * spread} V 10 H ${100 + o * spread} V 40 H 130`} />
      ))}
      <path className="a-fill" d={`${plot((t) => 54 - 8 * Math.exp(-(((t - 0.5) * 6) ** 2)), 50 - spread * 1.6, 50 + spread * 1.6)} Z`} />
      <path className="a-fill" d={`${plot((t) => 54 - 8 * Math.exp(-(((t - 0.5) * 6) ** 2)), 100 - spread * 1.6, 100 + spread * 1.6)} Z`} />
    </Art>
  );
}

// ========================================================= synchronization

/** Phase-locked loop: phase detector, loop filter and NCO. */
function PllArt({ caption, detector = "×", feedback = "NCO" }: { caption: ReactNode; detector?: string; feedback?: string }) {
  return (
    <Art caption={caption}>
      <Arrow x0={4} x1={34} y={16} />
      <circle className="a-node" cx={40} cy={16} r={6} />
      <text className="a-math" x={40} y={19}>{detector}</text>
      <Arrow x0={46} x1={136} y={16} cls="a-line" />
      <path className="a-line thin" d="M 90 16 V 40 H 84" />
      <rect className="a-box" x={58} y={33} width={26} height={14} rx={2} />
      <text className="a-tiny" x={71} y={42}>F(z)</text>
      <path className="a-line thin" d="M 58 40 H 50" />
      <rect className="a-box" x={22} y={33} width={28} height={14} rx={2} />
      <text className="a-tiny" x={36} y={42}>{feedback}</text>
      <path className="a-line thin a-arrow" d="M 40 33 V 23 M 37.5 26 L 40 22.5 L 42.5 26" />
    </Art>
  );
}

function CarrierRecoveryArt({ data }: ArtProps) {
  return <PllArt caption={<>B<sub>n</sub> <b>{num(data.params.loop_bandwidth, 0.005)}</b> · ζ <b>{num(data.params.damping, 0.707)}</b></>} />;
}

function TimingRecoveryArt({ data }: ArtProps) {
  return (
    <Art caption={<>B<sub>n</sub> <b>{num(data.params.loop_bandwidth, 0.01)}</b> · ζ <b>{num(data.params.damping, 0.707)}</b> · Gardner</>}>
      {[0, 1, 2, 3].map((k) => (
        <path key={k} className="a-line thin" style={{ opacity: 0.45 + k * 0.12 }} d={plot((t) => 28 - (k % 2 ? 1 : -1) * 18 * Math.cos(Math.PI * t) * (k > 1 ? 1 : 0.85), 10, 130)} />
      ))}
      <path className="a-warn-line" d="M 70 4 V 52" />
      <path className="a-warn-line a-arrow" d="M 60 52 L 70 46 L 80 52" />
      <circle className="a-dot warn" cx={70} cy={10} r={2} />
      <circle className="a-dot warn" cx={70} cy={46} r={2} />
    </Art>
  );
}

function FoCompensationArt({ data }: ArtProps) {
  const p = num(data.params.power, 4);
  return (
    <Art caption={<>(·)<sup>{p}</sup> spectrum · peak → Δf</>}>
      <path className="a-mute thin" d="M 6 46 H 134 M 70 46 V 6" />
      <path className="a-line" d={plot((t) => 46 - 3 * Math.abs(Math.sin(t * 61)) - 34 * Math.exp(-(((t - 0.68) * 40) ** 2)), 6, 134, 160)} />
      <path className="a-warn-line a-arrow" d="M 92 12 H 74 M 77.5 9.5 L 74 12 L 77.5 14.5" />
      <text className="a-small" x={98} y={14}>Δf</text>
    </Art>
  );
}

function PhaseRecoveryArt({ data }: ArtProps) {
  const bps = data.params.method !== "viterbi_viterbi";
  const rot = 0.35;
  const pts = [0, 1, 2, 3].map((k) => Math.PI / 4 + (k * Math.PI) / 2);
  return (
    <Art caption={bps ? <><b>BPS</b> · {num(data.params.n_test_phases, 32)} test phases</> : <><b>Viterbi-Viterbi</b> · 4th power</>}>
      <path className="a-mute thin" d="M 40 28 H 100 M 70 4 V 52" />
      <circle className="a-mute thin dash" cx={70} cy={28} r={20} fill="none" />
      {pts.map((a, k) => (
        <g key={k}>
          <circle className="a-dot dim" cx={70 + 20 * Math.cos(a + rot)} cy={28 - 20 * Math.sin(a + rot)} r={2} />
          <circle className="a-dot accent" cx={70 + 20 * Math.cos(a)} cy={28 - 20 * Math.sin(a)} r={2.2} />
        </g>
      ))}
      <path className="a-warn-line" d={`M ${70 + 26 * Math.cos(Math.PI / 4 + rot)} ${28 - 26 * Math.sin(Math.PI / 4 + rot)} A 26 26 0 0 1 ${70 + 26 * Math.cos(Math.PI / 4)} ${28 - 26 * Math.sin(Math.PI / 4)}`} />
    </Art>
  );
}

// ============================================================ equalization

/** Tapped delay line: z^-1 cells, tap weights, a sum and (adaptive) the update arrow. */
function TdlArt({ taps, caption, adaptive, feedback, extra }: {
  taps: number; caption: ReactNode; adaptive?: boolean; feedback?: number; extra?: ReactNode;
}) {
  const shown = Math.min(taps, feedback ? 3 : 5);
  const pitch = feedback ? 18 : 22;
  const xs = Array.from({ length: shown }, (_, k) => 10 + k * pitch);
  const sumX = feedback ? 82 : 124;
  return (
    <Art caption={caption}>
      <path className="a-line thin" d={`M 2 10 H ${xs[shown - 1] + 4}`} />
      {xs.slice(0, -1).map((x, k) => (
        <g key={k}>
          <rect className="a-box" x={x + 3} y={5} width={pitch - 6} height={10} rx={1.5} />
          <text className="a-tiny" x={x + pitch / 2} y={12.5}>z⁻¹</text>
        </g>
      ))}
      {xs.map((x, k) => (
        <g key={k}>
          <path className="a-line thin" d={`M ${x} 10 V 20`} />
          <path className="a-tri" d={`M ${x - 4} 20 L ${x + 4} 20 L ${x} 27 Z`} />
          <path className="a-line thin" d={`M ${x} 27 V 40 H ${sumX - 6}`} />
        </g>
      ))}
      {adaptive && <path className="a-warn-line a-arrow" d={`M ${xs[0] - 6} 31 L ${xs[shown - 1] + 8} 16 M ${xs[shown - 1] + 3.5} 16.4 L ${xs[shown - 1] + 8} 16 L ${xs[shown - 1] + 6.5} 20`} />}
      <circle className="a-node" cx={sumX} cy={40} r={6} />
      <text className="a-math" x={sumX} y={43}>Σ</text>
      {feedback ? (
        <g>
          <Arrow x0={sumX + 6} x1={100} y={40} cls="a-line" />
          <path className="a-box" d="M 100 33 h 14 v 14 h -14 Z" />
          <path className="a-line thin" d="M 103 44 H 107 V 36 H 111" />
          <Arrow x0={114} x1={136} y={40} cls="a-line" />
          <path className="a-line thin dash" d={`M 124 40 V 52 H ${sumX} V 46`} />
          <text className="a-tiny" x={104} y={55}>DFE ×{feedback}</text>
        </g>
      ) : (
        <Arrow x0={sumX + 6} x1={138} y={40} cls="a-line" />
      )}
      {extra}
    </Art>
  );
}

function LmsArt({ data }: ArtProps) {
  return (
    <TdlArt
      taps={num(data.params.n_taps, 11)}
      adaptive
      caption={<><b>{data.params.normalized === false ? "LMS" : "NLMS"}</b> · {num(data.params.n_taps, 11)} taps · μ {num(data.params.step_size, 0.1)}</>}
    />
  );
}

function DfeArt({ data }: ArtProps) {
  return (
    <TdlArt
      taps={num(data.params.ffe_taps, 7)}
      adaptive
      feedback={num(data.params.dfe_taps, 4)}
      caption={<><b>FFE {num(data.params.ffe_taps, 7)}</b> + <b>DFE {num(data.params.dfe_taps, 4)}</b> · μ {num(data.params.step_size, 0.05)}</>}
    />
  );
}

function LinearEqArt({ data }: ArtProps) {
  return <TdlArt taps={num(data.params.n_taps, 15)} caption={<><b>{String(data.params.method ?? "mmse").toUpperCase()}</b> · {num(data.params.n_taps, 15)} taps from CSI</>} />;
}

function EchoArt({ data }: ArtProps) {
  return (
    <TdlArt
      taps={num(data.params.n_taps, 16)}
      adaptive
      caption={<>echo replica · <b>{num(data.params.n_taps, 16)}</b> taps</>}
      extra={<text className="a-small" x={2} y={4}>tx</text>}
    />
  );
}

function OneTapArt({ data }: ArtProps) {
  const bars = 22;
  const H = (k: number) => 0.55 + 0.4 * Math.cos(k * 0.45) * Math.cos(k * 0.12);
  return (
    <Art caption={<><b>{String(data.params.method ?? "zf").toUpperCase()}</b> · one tap per subcarrier · ÷H</>}>
      {Array.from({ length: bars }, (_, k) => (
        <g key={k}>
          <rect className="a-bar" x={8 + k * 5.6} y={44 - H(k) * 34} width={2.2} height={H(k) * 34} />
          <rect className="a-bar on" x={10.6 + k * 5.6} y={44 - 20} width={2.2} height={20} />
        </g>
      ))}
      <path className="a-warn-line dash" d={plot((t) => 44 - 34 * H(t * (bars - 1)), 9, 9 + (bars - 1) * 5.6)} />
      <path className="a-mute thin" d="M 6 44.5 H 134" />
    </Art>
  );
}

function CtleArt({ data }: ArtProps) {
  const z = num(data.params.zero_hz, 1e9);
  const p1 = num(data.params.pole1_hz, 6e9);
  const p2 = num(data.params.pole2_hz, 25e9);
  const mag = (f: number) => Math.hypot(1, f / z) / (Math.hypot(1, f / p1) * Math.hypot(1, f / p2));
  const db = (t: number) => 20 * Math.log10(mag(10 ** (7 + t * 4)));
  const peak = Math.max(...Array.from({ length: 50 }, (_, k) => db(k / 49)));
  return (
    <Art caption={<>zero <b>{+(z / 1e9).toPrecision(3)} GHz</b> · peak <b>+{peak.toFixed(1)} dB</b></>}>
      <path className="a-mute thin" d="M 8 40 H 134 M 8 4 V 52" />
      <path className="a-line" d={plot((t) => 40 - db(t) * 1.4, 8, 134)} />
      <path className="a-mute thin dash" d={plot((t) => 40 + 14 * t * t, 8, 134)} />
      <text className="a-tiny" x={130} y={50}>f</text>
    </Art>
  );
}

function CdCompArt({ data }: ArtProps) {
  return (
    <Art caption={<>{data.params.dispersion_ps_nm == null ? <b>from link</b> : <><b>{num(data.params.dispersion_ps_nm, 0)} ps/nm</b></>} · all-pass</>}>
      <path className="a-line thin" d={plot((t) => 28 + 14 * Math.exp(-(((t - 0.5) * 3) ** 2)) * Math.sin((t - 0.5) * (t - 0.5) * 220), 6, 60, 120)} />
      <Arrow x0={66} x1={80} y={28} />
      <path className="a-fill" d={`${plot((t) => 28 - 18 * Math.exp(-(((t - 0.5) * 9) ** 2)), 86, 134)} Z`} />
      <path className="a-line" d={plot((t) => 28 - 18 * Math.exp(-(((t - 0.5) * 9) ** 2)), 86, 134)} />
      <text className="a-small" x={6} y={52}>dispersed</text>
      <text className="a-small" x={92} y={52}>compressed</text>
    </Art>
  );
}

/** 2 x 2 butterfly of adaptive filters (polarization demultiplexing). */
function ButterflyArt({ data }: ArtProps) {
  return (
    <Art caption={<><b>{String(data.params.algorithm ?? "lms").toUpperCase()}</b> · {num(data.params.n_taps, 15)} taps · 2×2</>}>
      {[[14, 14, "h_xx"], [14, 42, "h_yy"]].map(([, y]) => (
        <path key={String(y)} className="a-line thin" d={`M 6 ${y} H 120`} />
      ))}
      <path className="a-line thin" d="M 26 14 C 50 14, 50 42, 70 42 M 26 42 C 50 42, 50 14, 70 14" />
      {[[40, 7], [40, 35], [80, 7], [80, 35]].map(([x, y], k) => (
        <g key={k}>
          <rect className="a-box" x={x} y={y} width={22} height={14} rx={2} />
          <text className="a-tiny" x={x + 11} y={y + 9.5}>{["h_xx", "h_yy", "h_xy", "h_yx"][k]}</text>
        </g>
      ))}
      <circle className="a-node" cx={112} cy={14} r={5} />
      <circle className="a-node" cx={112} cy={42} r={5} />
      <text className="a-math" x={112} y={17}>Σ</text>
      <text className="a-math" x={112} y={45}>Σ</text>
      <Arrow x0={117} x1={136} y={14} cls="a-line" />
      <Arrow x0={117} x1={136} y={42} cls="a-line" />
    </Art>
  );
}

// =============================================================== detection

function ChannelEstArt({ data }: ArtProps) {
  const H = (t: number) => 30 - 14 * Math.cos(t * 9) * Math.cos(t * 3.2);
  return (
    <Art caption={<><b>{String(data.params.method ?? "ls").toUpperCase()}</b> at pilots · interpolated{data.params.average_symbols ? " · averaged" : ""}</>}>
      <path className="a-mute thin" d="M 6 50 H 134" />
      <path className="a-line" d={plot((t) => H(t), 8, 132)} />
      {Array.from({ length: 9 }, (_, k) => {
        const t = k / 8;
        return <circle key={k} className="a-dot warn" cx={8 + t * 124} cy={H(t) + (k % 2 ? 2 : -2)} r={2.2} />;
      })}
      <text className="a-tiny" x={6} y={8}>|Ĥ|</text>
    </Art>
  );
}

function MimoDetectorArt({ data }: ArtProps) {
  const method = String(data.params.method ?? "mmse");
  const formula = method === "zf" ? "(HᴴH)⁻¹Hᴴ" : method === "ml" ? "argmin‖y−Hx‖" : "(HᴴH+σ²I)⁻¹Hᴴ";
  return (
    <Art caption={<><b>{method.toUpperCase()}</b> detection</>}>
      {[12, 22, 34, 44].map((y) => <path key={y} className="a-line thin" d={`M 4 ${y} H 22 M 118 ${y} H 136`} />)}
      <rect className="a-box" x={22} y={6} width={96} height={44} rx={4} />
      <text className="a-math" x={70} y={32}>{formula}</text>
    </Art>
  );
}

function StreamArt({ data }: ArtProps) {
  const demux = data.type === "stream_demux";
  const n = Math.round(num(data.params.n_streams, 2));
  return (
    <Art caption={demux ? <>1 → <b>{n}</b> streams</> : <>streams → 1</>}>
      <path className="a-box" d={demux ? "M 50 18 L 90 6 V 50 L 50 38 Z" : "M 50 6 L 90 18 V 38 L 50 50 Z"} />
      {Array.from({ length: 4 }, (_, k) => 12 + k * 11).map((y) => (
        <path key={y} className="a-line thin" d={demux ? `M 90 ${y} H 130` : `M 10 ${y} H 50`} />
      ))}
      <path className="a-line" d={demux ? "M 10 28 H 50" : "M 90 28 H 130"} />
    </Art>
  );
}

// ============================================================ measurements

function EvmArt({ id }: ArtProps) {
  const { metrics } = useContext(RunDataContext);
  const evm = metrics[`${id}.evm_pct`];
  const mer = metrics[`${id}.mer_db`];
  const has = typeof evm === "number" && Number.isFinite(evm);
  const spread = has ? Math.min(10, 2 + (evm as number) * 0.4) : 4;
  return (
    <div className="evm-display">
      <svg viewBox="0 0 60 60" className="evm-plot">
        <rect className="a-screen" x={1} y={1} width={58} height={58} rx={4} />
        <path className="a-mute thin" d="M 30 4 V 56 M 4 30 H 56" />
        {[[18, 18], [42, 18], [18, 42], [42, 42]].map(([x, y], k) => (
          <g key={k}>
            <circle className="evm-cloud" cx={x} cy={y} r={spread} />
            <circle className="a-dot accent" cx={x} cy={y} r={1.6} />
          </g>
        ))}
        <path className="a-warn-line" d="M 42 18 L 47 13" />
        <circle className="a-dot warn" cx={47} cy={13} r={1.4} />
      </svg>
      <div className="evm-readout">
        <span className="ber-key">EVM</span>
        <span className="evm-value">{has ? `${(evm as number).toFixed(2)}%` : "—"}</span>
        <span className="ber-key">MER</span>
        <span className="evm-mer">{typeof mer === "number" && Number.isFinite(mer) ? `${mer.toFixed(1)} dB` : "—"}</span>
      </div>
    </div>
  );
}

/** Electrical power meter: an analog gauge with the live power, fed from the bottom. */
function PowerMeterArt({ id, size }: ArtProps) {
  const { metrics, running } = useContext(RunDataContext);
  const p = metrics[`${id}.power_db`];
  const papr = metrics[`${id}.papr_db_mean`];
  const has = typeof p === "number" && Number.isFinite(p);
  const { w, h } = size;
  const cx = w / 2;
  const cy = 64;
  const r = 40;
  const frac = has ? Math.min(Math.max(((p as number) + 30) / 40, 0), 1) : 0; // -30 ... +10 dB
  const ang = Math.PI * (1 - frac);
  const casing = h - 22;
  return (
    <svg className={`gauge-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <rect className="gauge-case" x={3} y={3} width={w - 6} height={casing - 3} rx={12} />
      <rect className="gauge-face" x={11} y={11} width={w - 22} height={68} rx={6} />
      {Array.from({ length: 11 }, (_, k) => {
        const a = Math.PI * (1 - k / 10);
        const long = k % 5 === 0;
        return (
          <path key={k} className="gauge-tick" d={`M ${cx + r * Math.cos(a)} ${cy - r * Math.sin(a)} L ${cx + (r - (long ? 7 : 4)) * Math.cos(a)} ${cy - (r - (long ? 7 : 4)) * Math.sin(a)}`} />
        );
      })}
      <path className="gauge-red" d={`M ${cx + r * Math.cos(Math.PI * 0.2)} ${cy - r * Math.sin(Math.PI * 0.2)} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} />
      <text className="gauge-scale" x={cx - r + 2} y={cy + 9}>−30</text>
      <text className="gauge-scale" x={cx + r - 2} y={cy + 9}>+10</text>
      <path className="gauge-needle" d={`M ${cx} ${cy} L ${cx + (r - 3) * Math.cos(ang)} ${cy - (r - 3) * Math.sin(ang)}`} />
      <circle className="gauge-hub" cx={cx} cy={cy} r={3} />
      <text className="gauge-value" x={cx} y={96}>{has ? `${(p as number).toFixed(2)} dB` : "— dB"}</text>
      <text className="gauge-scale" x={cx} y={108}>{typeof papr === "number" && Number.isFinite(papr) ? `PAPR ${papr.toFixed(1)} dB` : "POWER"}</text>
      <rect className="opm-adapter" x={cx - 7} y={casing - 2} width={14} height={8} rx={1.5} />
      <path className="rf-lead" d={`M ${cx} ${casing + 6} V ${h}`} />
    </svg>
  );
}

// ======================================================= frameless: physical

/** AWGN: the noise source feeding an adder on the line. */
function AwgnArt({ data, size }: ArtProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 44;
  const cx = w / 2;
  const mode = String(data.params.mode ?? "ebn0");
  const label = mode === "thermal" ? `NF ${num(data.params.noise_figure_db, 5)} dB` : `${{ snr: "SNR", esn0: "Es/N0", ebn0: "Eb/N0" }[mode] ?? mode} ${num(data.params.snr_db, 10)} dB`;
  const noise = useMemo(() => {
    let s = 7;
    return Array.from({ length: 40 }, () => {
      s = (s * 16807) % 2147483647;
      return (s / 2147483647 - 0.5) * 2;
    });
  }, []);
  return (
    <svg className={`awgn-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${cx - 9} M ${cx + 9} ${line} H ${w}`} />
      <rect className="noise-box" x={cx - 26} y={2} width={52} height={20} rx={4} />
      <path className="noise-trace" d={poly(noise.map((v, k) => [cx - 23 + k * 1.18, 12 + v * 6.5] as Pt))} />
      <path className="noise-feed a-arrow" d={`M ${cx} 22 V ${line - 10} M ${cx - 3} ${line - 13.5} L ${cx} ${line - 10} L ${cx + 3} ${line - 13.5}`} />
      <circle className="sum-node" cx={cx} cy={line} r={9} />
      <path className="sum-plus" d={`M ${cx - 5} ${line} H ${cx + 5} M ${cx} ${line - 5} V ${line + 5}`} />
      <text className="art-label" x={cx} y={line + 22}>{label}</text>
    </svg>
  );
}

/** Mixer with a local oscillator: carrier frequency offset, or phase noise (a jittery oscillator). */
function MixerArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 30;
  const cx = w / 2;
  const noisy = data.type === "phase_noise";
  const label = noisy
    ? `Δν ${num(data.params.linewidth_hz, 100)} Hz`
    : `Δf ${num(data.params.cfo_hz, 0)} Hz${num(data.params.phase_deg, 0) ? ` · ${num(data.params.phase_deg, 0)}°` : ""}`;
  const wave = plot((t) => 62 + 4.5 * Math.sin(t * 4 * Math.PI) + (noisy ? 1.6 * Math.sin(t * 47) : 0), cx - 8, cx + 8, 40);
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${cx - 9} M ${cx + 9} ${line} H ${w}`} />
      <circle className="sum-node" cx={cx} cy={line} r={9} />
      <path className="sum-plus" d={`M ${cx - 5} ${line - 5} L ${cx + 5} ${line + 5} M ${cx + 5} ${line - 5} L ${cx - 5} ${line + 5}`} />
      <path className="rf-lead thin" d={`M ${cx} ${line + 9} V ${62 - 12}`} />
      <circle className="lo-node" cx={cx} cy={62} r={12} />
      <path className="lo-wave" d={wave} />
      <text className="art-label" x={cx} y={line - 14}>{label}</text>
    </svg>
  );
}

/** Timing offset: a delay meander on the line. */
function DelayArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 30;
  const d = num(data.params.delay_samples, 0);
  const x0 = w / 2 - 22;
  let p = `M 0 ${line} H ${x0}`;
  for (let k = 0; k < 4; k++) p += ` V ${line - 12} h 5.5 V ${line + 12} h 5.5`;
  p += ` V ${line} H ${w}`;
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="delay-line" d={p} />
      <text className="art-label" x={w / 2} y={line + 26}>τ = {+d.toPrecision(3)} samples</text>
    </svg>
  );
}

/** Up / down sampler: the multirate symbol on the line. */
function RateArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 26;
  const up = data.type === "upsampler";
  const L = num(data.params.factor, 2);
  const cx = w / 2;
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${cx - 13} M ${cx + 13} ${line} H ${w}`} />
      <circle className="rate-node" cx={cx} cy={line} r={13} />
      <text className="rate-text" x={cx} y={line + 4.5}>{up ? "↑" : "↓"}{L}</text>
    </svg>
  );
}

/** Power amplifier: the triangle with its AM/AM compression curve. */
function PaArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 36;
  const model = String(data.params.model ?? "rapp");
  const p = num(data.params.smoothness, 2);
  const amam = (x: number) => (model === "clip" ? Math.min(x, 1) : model === "saleh" ? (2 * x) / (1 + x * x) : x / (1 + x ** (2 * p)) ** (1 / (2 * p)));
  const [x0, x1] = [26, 96];
  return (
    <svg className="amp-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${x0} M ${x1} ${line} H ${w}`} />
      <path className="amp-tri electrical" d={`M ${x0} ${line - 30} L ${x1} ${line} L ${x0} ${line + 30} Z`} />
      <path className="pa-axis" d={`M ${x0 + 6} ${line + 14} H ${x0 + 40} M ${x0 + 6} ${line + 14} V ${line - 14}`} />
      <path className="pa-curve" d={plot((t) => line + 14 - amam(t * 2.2) * 22, x0 + 6, x0 + 40)} />
      <text className="art-label" x={(x0 + x1) / 2} y={line + 44}>{model} · IBO {num(data.params.input_backoff_db, 6)} dB</text>
    </svg>
  );
}

/** Copper pair: two insulated wires twisting along the line. */
function TwistedPair({ x0, x1, y, pitch = 14, amp = 4 }: { x0: number; x1: number; y: number; pitch?: number; amp?: number }) {
  return (
    <g>
      <path className="pair-a" d={plot((t) => y + amp * Math.sin(((x1 - x0) * t * 2 * Math.PI) / pitch), x0, x1, 120)} />
      <path className="pair-b" d={plot((t) => y - amp * Math.sin(((x1 - x0) * t * 2 * Math.PI) / pitch), x0, x1, 120)} />
    </g>
  );
}

function CopperArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 40;
  const preset = String(data.params.preset ?? data.params.line_preset ?? "cat5e");
  const len = num(data.params.length_m, 100);
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H 14 M ${w - 14} ${line} H ${w}`} />
      <TwistedPair x0={14} x1={w - 14} y={line} />
      <path className="csi-lead" d={`M ${w - 24} ${line + 6} V 64 H ${w}`} />
      <text className="art-label" x={w / 2} y={line - 12}>{preset} · {len >= 1000 ? `${+(len / 1000).toPrecision(3)} km` : `${len} m`}</text>
    </svg>
  );
}

function CrosstalkArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const next = data.params.model !== "fext";
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 40 H 14 M ${w - 14} 40 H ${w} M 0 64 H 14`} />
      <TwistedPair x0={14} x1={w - 14} y={40} />
      <g opacity={0.55}>
        <TwistedPair x0={14} x1={w - 14} y={64} />
      </g>
      {[0.3, 0.5, 0.7].map((f) => (
        <path key={f} className="xt-arrow a-arrow" d={`M ${w * f} 58 V 47 M ${w * f - 2.5} 50 L ${w * f} 46.5 L ${w * f + 2.5} 50`} />
      ))}
      <text className="art-label" x={w / 2} y={26}>{next ? "NEXT" : "FEXT"} · {num(data.params.n_disturbers, 10)} disturbers</text>
    </svg>
  );
}

function HybridEchoArt({ data, size }: ArtProps) {
  const { w, h } = size;
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 40 H ${w / 2 - 9} M ${w / 2 + 9} 40 H ${w} M 0 64 H ${w / 2 - 16}`} />
      <path className="xt-arrow" d={`M ${w / 2 - 16} 64 C ${w / 2} 64, ${w / 2} 60, ${w / 2} 49`} />
      <path className="xt-arrow" d={`M ${w / 2 - 3} 52 L ${w / 2} 48.5 L ${w / 2 + 3} 52`} />
      <circle className="sum-node" cx={w / 2} cy={40} r={9} />
      <path className="sum-plus" d={`M ${w / 2 - 5} 40 H ${w / 2 + 5} M ${w / 2} 35 V 45`} />
      <text className="art-label" x={w / 2} y={24}>echo {num(data.params.echo_db, -10)} dB</text>
    </svg>
  );
}

function ImpulseNoiseArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 40;
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H ${w}`} />
      {[0.32, 0.62].map((f) => (
        <path key={f} className="bolt" d={`M ${w * f + 4} 6 L ${w * f - 4} 22 L ${w * f + 2} 22 L ${w * f - 5} ${line - 2} L ${w * f + 8} 17 L ${w * f + 1} 17 L ${w * f + 8} 6 Z`} />
      ))}
      <text className="art-label" x={w / 2} y={line + 18}>p {num(data.params.burst_probability, 0.001)} · +{num(data.params.power_db, 10)} dB</text>
    </svg>
  );
}

/** One antenna element ``-<|`` (as in the MIMO channel), mirrored with ``dir`` = -1. */
function Antenna({ x, y, dir }: { x: number; y: number; dir: 1 | -1 }) {
  const X = (d: number) => x + dir * d;
  return (
    <g className="mimo-array">
      <path d={`M ${X(0)} ${y} H ${X(9)}`} />
      <path className="mimo-horn" d={`M ${X(20)} ${y - 7} L ${X(9)} ${y} L ${X(20)} ${y + 7}`} />
      <path className="mimo-face" d={`M ${X(20)} ${y - 8} V ${y + 8}`} />
    </g>
  );
}

/** Flat fading: TX and RX antennas with the fading envelope of the channel between them. */
function FadingArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 40;
  const rician = data.params.model === "rician";
  const env = (t: number) => {
    const v = Math.abs(Math.sin(t * 9.3) + 0.8 * Math.sin(t * 23.1 + 1) + (rician ? 1.6 : 0.2));
    return line - 4 - Math.min(26, 16 * v);
  };
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <Antenna x={0} y={line} dir={1} />
      <Antenna x={w} y={line} dir={-1} />
      <path className="ray-dash" d={`M 24 ${line} H ${w - 24}`} />
      <path className="fade-env" d={plot(env, 28, w - 28, 90)} />
      <path className="csi-lead" d={`M ${w - 10} ${line + 8} V 64 H ${w}`} />
      <text className="art-label" x={w / 2} y={h - 2}>{rician ? `Rician K ${num(data.params.k_factor_db, 0)} dB` : "Rayleigh"} · {String(data.params.variation ?? "block")}</text>
    </svg>
  );
}

/** Multipath: the direct ray and rays reflected by scatterers, longer and weaker. */
function MultipathArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 40;
  const preset = data.params.preset ? String(data.params.preset) : null;
  const scat: Pt[] = [[w * 0.38, 8], [w * 0.62, 12], [w * 0.5, 66]];
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <Antenna x={0} y={line} dir={1} />
      <Antenna x={w} y={line} dir={-1} />
      <path className="mimo-ray live" d={`M 24 ${line} H ${w - 24}`} style={{ strokeWidth: 2 }} />
      {scat.map(([x, y], k) => (
        <g key={k}>
          <path className="mimo-ray live" style={{ strokeOpacity: 0.6 - k * 0.15, strokeWidth: 1.4 - k * 0.3 }} d={`M 24 ${line} L ${x} ${y} L ${w - 24} ${line}`} />
          <rect className="scatterer" x={x - 5} y={y - (y < line ? 6 : 0)} width={10} height={6} rx={1} />
        </g>
      ))}
      <path className="csi-lead" d={`M ${w - 10} ${line + 8} V 64 H ${w}`} />
      <text className="art-label" x={w / 2} y={h - 1}>{preset ?? `${num(data.params.delay_spread_ns, 100)} ns spread`}{data.params.fading !== false ? " · fading" : ""}</text>
    </svg>
  );
}

/** Path loss: a base-station mast and a terminal, with the wavefronts fading over the distance. */
function PathLossArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 50;
  const d = num(data.params.distance_m, 100);
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="rf-lead" d={`M 0 ${line} H 14 M ${w - 14} ${line} H ${w}`} />
      <path className="mast" d={`M 14 ${line} L 22 10 L 30 ${line} M 17 ${line - 12} H 27 M 19 ${line - 24} H 25 M 22 10 V 4`} />
      <circle className="a-dot warn" cx={22} cy={8} r={2.2} />
      {[0, 1, 2, 3, 4].map((k) => (
        <path key={k} className="wavefront" style={{ opacity: 0.85 - k * 0.16 }} d={`M ${40 + k * 18} 14 A 30 30 0 0 1 ${40 + k * 18} 42`} />
      ))}
      <rect className="ue" x={w - 26} y={line - 22} width={12} height={20} rx={2.5} />
      <path className="ue-screen" d={`M ${w - 24} ${line - 19} h 8 v 12 h -8 Z`} />
      <path className="dist" d={`M 22 ${line + 8} H ${w - 20} M 26 ${line + 5} L 22 ${line + 8} L 26 ${line + 11} M ${w - 24} ${line + 5} L ${w - 20} ${line + 8} L ${w - 24} ${line + 11}`} />
      <text className="art-label" x={w / 2} y={line + 22}>{String(data.params.model ?? "free_space").replace("_", " ")} · {d >= 1000 ? `${+(d / 1000).toPrecision(3)} km` : `${d} m`} · {num(data.params.frequency_ghz, 3.5)} GHz</text>
    </svg>
  );
}

/** Free-space optics: transmit telescope, diverging beam through the air, receive aperture. */
function FsoArt({ data, size }: ArtProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 40;
  const turb = data.type === "fso_turbulence" || data.params.turbulence !== "none";
  const channel = data.type === "fso_channel";
  const d = num(data.params.distance_m, 1000);
  const x0 = channel ? 30 : 8;
  const x1 = channel ? w - 30 : w - 8;
  return (
    <svg className={`laser-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ ["--beam" as string]: wavelengthColor(1550) }}>
      <path className="fiber-line" d={`M 0 ${line} H ${x0} M ${x1} ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      <path className="fso-beam" d={channel ? `M ${x0} ${line - 3} L ${x1} ${line - 14} V ${line + 14} L ${x0} ${line + 3} Z` : `M ${x0} ${line - 7} H ${x1} V ${line + 7} H ${x0} Z`} />
      {turb && [0.3, 0.5, 0.7].map((f, k) => (
        <path key={f} className="eddy" d={`M ${w * f - 8} ${line + (k % 2 ? 9 : -9)} c 4 -6, 10 -6, 12 0 s -4 8, -8 4`} />
      ))}
      {channel && (
        <>
          <path className="scope-tube" d={`M 10 ${line - 7} H 30 L 34 ${line - 10} V ${line + 10} L 30 ${line + 7} H 10 Z`} />
          <path className="scope-tube" d={`M ${w - 34} ${line - 15} L ${w - 30} ${line - 9} H ${w - 10} V ${line + 9} H ${w - 30} L ${w - 34} ${line + 15} Z`} />
        </>
      )}
      <path className="csi-lead" d={`M ${w - 6} ${line + 8} V 64 H ${w}`} />
      <text className="art-label" x={w / 2} y={h - 2}>
        {channel ? `${d >= 1000 ? `${+(d / 1000).toPrecision(3)} km` : `${d} m`} · ${String(data.params.turbulence ?? "gamma_gamma").replace("_", "-")}` : `${String(data.params.model ?? "lognormal").replace("_", "-")} turbulence`}
      </text>
    </svg>
  );
}

/** WDM multiplexer / demultiplexer: coloured channels merging into, or leaving, one fiber. */
function WdmArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 40;
  const demux = data.type === "wdm_demultiplexer";
  const n = Math.min(Math.max(Math.round(num(data.params.n_channels, 5)), 2), 7);
  const sel = num(data.params.channel, 0);
  const ys = Array.from({ length: n }, (_, k) => 10 + ((k + 0.5) / n) * 60);
  const hue = (k: number) => `hsl(${10 + (k * 260) / (n - 1)}, 85%, 62%)`;
  const mid = Math.floor(n / 2);
  const selK = demux ? Math.min(Math.max(mid + sel, 0), n - 1) : mid;
  const [bx0, bx1] = demux ? [w * 0.34, w * 0.5] : [w * 0.5, w * 0.66];
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path className="fiber-line" d={demux ? `M 0 ${line} H ${bx0}` : `M ${bx1} ${line} H ${w}`} style={{ stroke: "#e2e8f0" }} />
      {ys.map((y, k) => (
        <path
          key={k}
          className="wdm-ch"
          style={{ stroke: hue(k), opacity: k === selK ? 1 : 0.45 }}
          d={demux
            ? `M ${bx1} ${line + (y - line) * 0.25} C ${bx1 + 14} ${line + (y - line) * 0.25}, ${bx1 + 10} ${y}, ${bx1 + 26} ${y} H ${k === selK ? w : w - 18}`
            : `M ${k === selK ? 0 : 18} ${y} H ${bx0 - 26} C ${bx0 - 10} ${y}, ${bx0 - 14} ${line + (y - line) * 0.25}, ${bx0} ${line + (y - line) * 0.25}`}
        />
      ))}
      <path className="awg" d={`M ${bx0} ${line - (demux ? 8 : 22)} L ${bx1} ${line - (demux ? 22 : 8)} V ${line + (demux ? 22 : 8)} L ${bx0} ${line + (demux ? 8 : 22)} Z`} />
      <path className="fiber-line" d={demux ? `M ${bx1 + 26} ${line} H ${w}` : `M 0 ${line} H ${bx0 - 26}`} style={{ stroke: hue(selK) }} />
      <text className="art-label" x={w / 2} y={h - 1}>{demux ? `channel ${sel}` : `${num(data.params.n_channels, 5)} ch · ${num(data.params.spacing_ghz, 50)} GHz`}</text>
    </svg>
  );
}

/** Mach-Zehnder modulator: CW laser into the interferometer, electrodes driven by the input. */
function MzmArt({ data, size }: ArtProps) {
  const { running } = useContext(RunDataContext);
  const { w, h } = size;
  const line = 40;
  const nm = num(data.params.wavelength_nm, 1550);
  const iq = data.type === "iq_modulator";
  const eam = data.type === "intensity_modulator";
  const [a0, a1] = [48, w - 22];
  const arm = (y: number, x0: number, x1: number) => `M ${x0} ${line} C ${x0 + 10} ${line}, ${x0 + 6} ${y}, ${x0 + 16} ${y} H ${x1 - 16} C ${x1 - 6} ${y}, ${x1 - 10} ${line}, ${x1} ${line}`;
  return (
    <svg className={`laser-art${running ? " running" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ ["--beam" as string]: wavelengthColor(nm) }}>
      <path className="rf-lead" d={`M 0 ${line} H 6 V 6 H ${(a0 + a1) / 2 - 6} V ${iq ? 12 : 16}`} />
      <rect className="cw-laser" x={12} y={line - 8} width={22} height={16} rx={2} />
      <circle className="laser-glow on" cx={30} cy={line} r={5} />
      <text className="cw-text" x={21} y={line + 3}>CW</text>
      <path className="waveguide" d={`M 34 ${line} H ${a0}`} />
      {eam ? (
        <g>
          <path className="waveguide" d={`M ${a0} ${line} H ${a1}`} />
          <rect className="eam" x={(a0 + a1) / 2 - 18} y={line - 9} width={36} height={18} rx={2} />
          <path className="electrode" d={`M ${(a0 + a1) / 2 - 14} 16 H ${(a0 + a1) / 2 + 14}`} />
          <path className="rf-lead thin" d={`M ${(a0 + a1) / 2} 16 V ${line - 9}`} />
          <text className="cw-text dim" x={(a0 + a1) / 2} y={line + 3}>ER {num(data.params.extinction_ratio_db, 10)}</text>
        </g>
      ) : iq ? (
        <g>
          {[line - 18, line + 18].map((y, k) => (
            <g key={k}>
              <path className="waveguide" d={arm(y, a0, a1)} />
              <path className="waveguide thin" d={`M ${a0 + 26} ${y} l 6 -4 H ${a1 - 32} l 6 4 M ${a0 + 26} ${y} l 6 4 H ${a1 - 32} l 6 -4`} />
              <text className="cw-text dim" x={(a0 + a1) / 2} y={y + (k ? 11 : -6)}>{k ? "Q" : "I"}</text>
            </g>
          ))}
          <rect className="eam" x={a1 - 32} y={line + 12} width={14} height={11} rx={2} />
          <text className="cw-text" x={a1 - 25} y={line + 20}>π/2</text>
          <path className="electrode" d={`M ${(a0 + a1) / 2 - 16} 12 H ${(a0 + a1) / 2 + 16}`} />
        </g>
      ) : (
        <g>
          <path className="waveguide" d={arm(line - 13, a0, a1)} />
          <path className="waveguide" d={arm(line + 13, a0, a1)} />
          <path className="electrode" d={`M ${a0 + 20} 16 H ${a1 - 20} M ${a0 + 20} ${line + 20} H ${a1 - 20}`} />
          <text className="cw-text dim" x={(a0 + a1) / 2} y={line + 3}>V<tspan baselineShift="sub" fontSize="6">π</tspan> {num(data.params.v_pi, 4)} V</text>
        </g>
      )}
      <path className="fiber-line" d={`M ${a1} ${line} H ${w}`} style={{ stroke: "#facc15" }} />
      <path className="laser-pulse" d={`M ${a1} ${line} H ${w}`} pathLength={100} />
    </svg>
  );
}

/** Coherent receiver: signal and local oscillator into a 90° hybrid and balanced photodiodes. */
function CoherentRxArt({ data, size }: ArtProps) {
  const { w, h } = size;
  const line = 32;
  return (
    <svg className="mixer-art" width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ ["--beam" as string]: wavelengthColor(1550) }}>
      <path className="fiber-line" d={`M 0 ${line} H 50`} style={{ stroke: "#facc15" }} />
      <rect className="cw-laser" x={14} y={58} width={22} height={14} rx={2} />
      <text className="cw-text" x={25} y={68}>LO</text>
      <path className="waveguide" d={`M 36 65 H 44 V ${line + 14} H 50`} />
      <rect className="hybrid" x={50} y={line - 12} width={44} height={36} rx={3} />
      <text className="cw-text" x={72} y={line + 4}>90°</text>
      <text className="cw-text dim" x={72} y={line + 14}>hybrid</text>
      {[0, 1, 2, 3].map((k) => {
        const y = line - 8 + k * 9;
        return <path key={k} className="waveguide thin" d={`M 94 ${y} H 108`} />;
      })}
      {[0, 1].map((k) => {
        const y = line - 4 + k * 18;
        return (
          <g key={k}>
            <path className="rx-diode" d={`M 110 ${y - 8} l 5 0 l -2.5 -4 Z M 110 ${y + 3} l 5 0 l -2.5 -4 Z`} transform="translate(0 4)" />
            <path className="rx-wire" d={`M 118 ${y} H 128`} />
            <text className="cw-text dim" x={134} y={y + 3}>{k ? "Q" : "I"}</text>
          </g>
        );
      })}
      <path className="rf-lead" d={`M 128 ${line - 4} V ${line} H ${w} M 128 ${line + 14} V ${line}`} />
      <text className="art-label" x={w / 2} y={h - 2}>LO {num(data.params.lo_power_dbm, 13)} dBm</text>
    </svg>
  );
}

// ================================================================ registry

type Draw = (p: ArtProps) => ReactNode;

export const ART: Record<string, Draw> = {
  hamming_encoder: (p) => <HammingArt {...p} />,
  hamming_decoder: (p) => <HammingArt {...p} />,
  repetition_encoder: (p) => <RepetitionArt {...p} />,
  repetition_decoder: (p) => <RepetitionArt {...p} />,
  conv_encoder: (p) => <ConvArt {...p} />,
  viterbi_decoder: (p) => <TrellisArt {...p} />,
  ldpc_encoder: (p) => <LdpcArt {...p} />,
  ldpc_decoder: (p) => <LdpcArt {...p} />,
  polar_encoder: (p) => <PolarArt {...p} />,
  polar_decoder: (p) => <PolarArt {...p} />,
  crc_encoder: (p) => <CrcArt {...p} />,
  crc_decoder: (p) => <CrcArt {...p} />,
  line_block_encoder: (p) => <LineBlockArt {...p} />,
  line_block_decoder: (p) => <LineBlockArt {...p} />,
  puncturer: (p) => <PunctureArt {...p} />,
  depuncturer: (p) => <PunctureArt {...p} />,
  alamouti_encoder: (p) => <AlamoutiArt {...p} />,
  alamouti_decoder: (p) => <AlamoutiArt {...p} />,
  interleaver: (p) => <InterleaverArt {...p} />,
  scrambler: (p) => <ScramblerArt {...p} />,
  demapper: (p) => <DemapperArt {...p} />,
  line_encoder: (p) => <LineCodeArt {...p} />,
  line_decoder: (p) => <LineCodeArt {...p} />,
  preamble: (p) => <PreambleArt {...p} />,
  frame_sync: (p) => <FrameSyncArt {...p} />,
  ofdm_sync: (p) => <FrameSyncArt {...p} />,
  pulse_shaper: (p) => <PulseArt {...p} />,
  matched_filter: (p) => <PulseArt {...p} />,
  ofdm_modulator: (p) => <CombArt {...p} />,
  ofdm_demodulator: (p) => <CombArt {...p} />,
  dmt_modulator: (p) => <DmtArt {...p} />,
  dmt_demodulator: (p) => <DmtArt {...p} />,
  fir_filter: (p) => <FirArt {...p} />,
  quantizer: (p) => <QuantizerArt {...p} />,
  iq_imbalance: (p) => <IqImbalanceArt {...p} />,
  jitter: (p) => <JitterArt {...p} />,
  carrier_recovery: (p) => <CarrierRecoveryArt {...p} />,
  timing_recovery: (p) => <TimingRecoveryArt {...p} />,
  fo_compensation: (p) => <FoCompensationArt {...p} />,
  phase_recovery: (p) => <PhaseRecoveryArt {...p} />,
  lms_equalizer: (p) => <LmsArt {...p} />,
  dfe_equalizer: (p) => <DfeArt {...p} />,
  linear_equalizer: (p) => <LinearEqArt {...p} />,
  echo_canceller: (p) => <EchoArt {...p} />,
  one_tap_equalizer: (p) => <OneTapArt {...p} />,
  ofdm_equalizer: (p) => <OneTapArt {...p} />,
  ctle: (p) => <CtleArt {...p} />,
  cd_compensation: (p) => <CdCompArt {...p} />,
  adaptive_mimo_equalizer: (p) => <ButterflyArt {...p} />,
  ofdm_channel_estimator: (p) => <ChannelEstArt {...p} />,
  mimo_detector: (p) => <MimoDetectorArt {...p} />,
  stream_mux: (p) => <StreamArt {...p} />,
  stream_demux: (p) => <StreamArt {...p} />,
  evm_meter: (p) => <EvmArt {...p} />,
  // frameless
  power_meter: (p) => <PowerMeterArt {...p} />,
  awgn: (p) => <AwgnArt {...p} />,
  carrier_offset: (p) => <MixerArt {...p} />,
  phase_noise: (p) => <MixerArt {...p} />,
  timing_offset: (p) => <DelayArt {...p} />,
  upsampler: (p) => <RateArt {...p} />,
  downsampler: (p) => <RateArt {...p} />,
  power_amplifier: (p) => <PaArt {...p} />,
  copper_line: (p) => <CopperArt {...p} />,
  crosstalk: (p) => <CrosstalkArt {...p} />,
  hybrid_echo: (p) => <HybridEchoArt {...p} />,
  impulse_noise: (p) => <ImpulseNoiseArt {...p} />,
  flat_fading: (p) => <FadingArt {...p} />,
  multipath: (p) => <MultipathArt {...p} />,
  path_loss: (p) => <PathLossArt {...p} />,
  fso_channel: (p) => <FsoArt {...p} />,
  fso_turbulence: (p) => <FsoArt {...p} />,
  wdm_multiplexer: (p) => <WdmArt {...p} />,
  wdm_demultiplexer: (p) => <WdmArt {...p} />,
  mach_zehnder_modulator: (p) => <MzmArt {...p} />,
  intensity_modulator: (p) => <MzmArt {...p} />,
  iq_modulator: (p) => <MzmArt {...p} />,
  coherent_receiver: (p) => <CoherentRxArt {...p} />,
};
