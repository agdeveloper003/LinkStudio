// Block geometry: size of a block node and where each port sits on its border.
//
// Ports are keyed "in:<name>" / "out:<name>". A port on the left or right side has an
// offset along the body (below the header); on the top or bottom side, along the width.
// Sizes are computed here (not measured from the DOM) so that snapping and routing can
// use them while a block is being dragged.

import type { PortPlace, Side } from "./api";
import type { BlockData } from "./graph";

export const NODE_WIDTH = 184;
export const HEADER = 36; // instance name and block type
export const ROW = 22; // vertical pitch of side ports
const BOTTOM_PAD = 6;
const TOP_PITCH = 64; // horizontal pitch of top/bottom ports

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export const portKey = (dir: "in" | "out", name: string) => `${dir}:${name}`;

/** Scope blocks (``scope_*`` types) show their plot inside the block. */
export const isScopeType = (type: string) => type.startsWith("scope_");

/** Default size of a block drawn with a display of its own; resizable ones keep the user's size. */
export interface BodySpec {
  w: number;
  h: number;
  resizable?: boolean;
  minW?: number;
  minH?: number;
  /**
   * Drawn without the block frame and header (the link media: fibers, wireless channel).
   * ``ports`` is the vertical band [top, bottom] holding the side ports, in node pixels.
   */
  frameless?: { ports: [number, number] };
  /** Default places of some ports ("in:in" on the bottom side...); the user can still move them. */
  ports?: Record<string, PortPlace>;
}

/** Size of a block node; ``top`` / ``bottom`` bound the band of the side ports. */
export interface BlockSize {
  w: number;
  h: number;
  top?: number;
  bottom?: number;
}

const SOUTH = (offset = 0.5): PortPlace => ({ side: "bottom", offset });

/** Signal-processing blocks drawn with a picture under the header (BlockArt.tsx). */
const ART_BODY: BodySpec = { w: 196, h: HEADER + 88 };

const BODIES: Record<string, BodySpec> = {
  hamming_encoder: ART_BODY,
  hamming_decoder: ART_BODY,
  repetition_encoder: ART_BODY,
  repetition_decoder: ART_BODY,
  conv_encoder: ART_BODY,
  viterbi_decoder: ART_BODY,
  ldpc_encoder: ART_BODY,
  ldpc_decoder: ART_BODY,
  polar_encoder: ART_BODY,
  polar_decoder: ART_BODY,
  crc_encoder: ART_BODY,
  crc_decoder: ART_BODY,
  line_block_encoder: ART_BODY,
  line_block_decoder: ART_BODY,
  puncturer: ART_BODY,
  depuncturer: ART_BODY,
  alamouti_encoder: ART_BODY,
  alamouti_decoder: ART_BODY,
  interleaver: ART_BODY,
  scrambler: ART_BODY,
  demapper: ART_BODY,
  line_encoder: ART_BODY,
  line_decoder: ART_BODY,
  preamble: ART_BODY,
  frame_sync: ART_BODY,
  ofdm_sync: ART_BODY,
  pulse_shaper: ART_BODY,
  matched_filter: ART_BODY,
  ofdm_modulator: ART_BODY,
  ofdm_demodulator: ART_BODY,
  dmt_modulator: ART_BODY,
  dmt_demodulator: ART_BODY,
  fir_filter: ART_BODY,
  quantizer: ART_BODY,
  iq_imbalance: ART_BODY,
  jitter: ART_BODY,
  carrier_recovery: ART_BODY,
  timing_recovery: ART_BODY,
  fo_compensation: ART_BODY,
  phase_recovery: ART_BODY,
  lms_equalizer: ART_BODY,
  dfe_equalizer: ART_BODY,
  linear_equalizer: ART_BODY,
  echo_canceller: ART_BODY,
  one_tap_equalizer: ART_BODY,
  ofdm_equalizer: ART_BODY,
  ctle: ART_BODY,
  cd_compensation: ART_BODY,
  adaptive_mimo_equalizer: ART_BODY,
  ofdm_channel_estimator: ART_BODY,
  mimo_detector: ART_BODY,
  stream_mux: ART_BODY,
  stream_demux: ART_BODY,
  evm_meter: { w: 220, h: HEADER + 84 },
  power_meter: { w: 120, h: 136, frameless: { ports: [68, 68] }, ports: { "in:in": SOUTH() } },
  awgn: { w: 110, h: 76, frameless: { ports: [44, 44] } },
  carrier_offset: { w: 110, h: 80, frameless: { ports: [30, 30] } },
  phase_noise: { w: 110, h: 80, frameless: { ports: [30, 30] } },
  timing_offset: { w: 110, h: 64, frameless: { ports: [30, 30] } },
  upsampler: { w: 76, h: 52, frameless: { ports: [26, 26] } },
  downsampler: { w: 76, h: 52, frameless: { ports: [26, 26] } },
  power_amplifier: { w: 122, h: 84, frameless: { ports: [36, 36] } },
  copper_line: { w: 170, h: 80, frameless: { ports: [28, 76] } },
  crosstalk: { w: 170, h: 80, frameless: { ports: [28, 76] } },
  hybrid_echo: { w: 130, h: 80, frameless: { ports: [28, 76] } },
  impulse_noise: { w: 120, h: 64, frameless: { ports: [40, 40] } },
  flat_fading: { w: 170, h: 86, frameless: { ports: [28, 76] } },
  multipath: { w: 180, h: 86, frameless: { ports: [28, 76] } },
  path_loss: { w: 180, h: 80, frameless: { ports: [50, 50] } },
  fso_channel: { w: 190, h: 86, frameless: { ports: [28, 76] } },
  fso_turbulence: { w: 160, h: 86, frameless: { ports: [28, 76] } },
  wdm_multiplexer: { w: 150, h: 86, frameless: { ports: [40, 40] } },
  wdm_demultiplexer: { w: 150, h: 86, frameless: { ports: [40, 40] } },
  mach_zehnder_modulator: { w: 180, h: 70, frameless: { ports: [40, 40] } },
  intensity_modulator: { w: 170, h: 64, frameless: { ports: [40, 40] } },
  iq_modulator: { w: 190, h: 80, frameless: { ports: [40, 40] } },
  coherent_receiver: { w: 170, h: 86, frameless: { ports: [32, 32] } },
  random_bits: { w: 240, h: HEADER + 104 },
  prbs: { w: 240, h: HEADER + 84 },
  dml: { w: 176, h: 64, frameless: { ports: [40, 40] } },
  vcsel: { w: 180, h: 80, frameless: { ports: [40, 40] } },
  photodiode: { w: 160, h: 70, frameless: { ports: [40, 40] } },
  optical_amplifier: { w: 140, h: 72, frameless: { ports: [40, 40] } },
  optical_attenuator: { w: 120, h: 84, frameless: { ports: [40, 40] } },
  gain: { w: 112, h: 60, frameless: { ports: [32, 32] } },
  agc: { w: 124, h: 80, frameless: { ports: [40, 40] } },
  optical_power_meter: { w: 128, h: 158, frameless: { ports: [79, 79] }, ports: { "in:in": SOUTH() } },
  ber_meter: { w: 200, h: HEADER + 86 },
  mapper: { w: 180, h: HEADER + 156, resizable: true, minW: 150, minH: HEADER + 110 },
  mimo_channel: { w: 200, h: 104, frameless: { ports: [42, 82] } },
  fiber: { w: 150, h: 46, frameless: { ports: [38, 38] } },
  multimode_fiber: { w: 150, h: 46, frameless: { ports: [38, 38] } },
  optical_link: { w: 220, h: 42, frameless: { ports: [30, 30] } },
};
/** Scope inputs enter from the bottom, the first one in the middle and the others around it. */
const SCOPE_BODY: BodySpec = {
  w: 320, h: HEADER + 220, resizable: true, minW: 220, minH: HEADER + 150,
  ports: { "in:in1": SOUTH(0.5), "in:in2": SOUTH(0.3), "in:in3": SOUTH(0.7), "in:in4": SOUTH(0.1) },
};

export function bodySpec(type: string): BodySpec | undefined {
  return isScopeType(type) ? SCOPE_BODY : BODIES[type];
}

/** Every port with its place: stored overrides, defaults for the others. */
export function portPlaces(data: BlockData): Record<string, PortPlace> {
  const stored = { ...bodySpec(data.type)?.ports, ...data.ports };
  const keys = [...data.inputs.map((p) => portKey("in", p.name)), ...data.outputs.map((p) => portKey("out", p.name))];
  const out: Record<string, PortPlace> = {};
  // Default: inputs down the left side, outputs down the right side, one row each.
  const left = keys.filter((k) => !stored[k] && k.startsWith("in:"));
  const right = keys.filter((k) => !stored[k] && k.startsWith("out:"));
  const sideCount = (side: Side) => keys.filter((k) => stored[k]?.side === side).length;
  const rows = Math.max(left.length + sideCount("left"), right.length + sideCount("right"), 1);
  const used: Record<string, Set<number>> = { left: new Set(), right: new Set() };
  for (const k of keys) {
    const s = stored[k];
    if (s && (s.side === "left" || s.side === "right")) used[s.side].add(Math.round(s.offset * rows - 0.5));
  }
  const fill = (list: string[], side: "left" | "right") => {
    let row = 0;
    for (const k of list) {
      while (used[side].has(row)) row += 1;
      out[k] = { side, offset: (row + 0.5) / rows };
      row += 1;
    }
  };
  fill(left, "left");
  fill(right, "right");
  for (const k of keys) if (stored[k]) out[k] = stored[k];
  return out;
}

/** Width and height of a block node. */
export function blockSize(data: BlockData, places = portPlaces(data)): BlockSize {
  const count = (side: Side) => Object.values(places).filter((p) => p.side === side).length;
  const rows = Math.max(count("left"), count("right"), 1);
  const cols = Math.max(count("top"), count("bottom"));
  const minH = HEADER + rows * ROW + BOTTOM_PAD;
  const body = bodySpec(data.type);
  if (body?.frameless) {
    const [top, bottom] = body.frameless.ports;
    return { w: body.w, h: body.h, top, bottom: body.h - bottom };
  }
  if (body) {
    const user = body.resizable ? data.size : undefined;
    return {
      w: Math.max(user ? Math.max(user.w, body.minW ?? 0) : body.w, body.ports ? 0 : (cols + 1) * TOP_PITCH),
      h: Math.max(user ? Math.max(user.h, body.minH ?? 0) : body.h, minH),
    };
  }
  return { w: Math.max(NODE_WIDTH, (cols + 1) * TOP_PITCH), h: minH };
}

/** Port position relative to the block's top-left corner. */
export function placePoint(place: PortPlace, size: BlockSize): Point {
  const top = size.top ?? HEADER;
  const body = size.h - top - (size.bottom ?? BOTTOM_PAD);
  switch (place.side) {
    case "left":
      return { x: 0, y: top + place.offset * body };
    case "right":
      return { x: size.w, y: top + place.offset * body };
    case "top":
      return { x: place.offset * size.w, y: 0 };
    default:
      return { x: place.offset * size.w, y: size.h };
  }
}

/** Nearest side and snapped offset for a pointer at ``p`` (relative to the block). */
export function placeFromPoint(p: Point, size: BlockSize): PortPlace {
  const top = size.top ?? HEADER;
  const body = Math.max(size.h - top - (size.bottom ?? BOTTOM_PAD), 1);
  const dist: [Side, number][] = [
    ["left", Math.abs(p.x)],
    ["right", Math.abs(size.w - p.x)],
    ["top", Math.abs(p.y)],
    ["bottom", Math.abs(size.h - p.y)],
  ];
  const side = dist.reduce((a, b) => (b[1] < a[1] ? b : a))[0];
  if (side === "left" || side === "right") {
    const steps = Math.max(1, Math.round((2 * body) / ROW)); // half-row grid
    const k = Math.round(((p.y - top) / body) * steps);
    return { side, offset: Math.min(steps - 1, Math.max(1, k)) / steps };
  }
  const steps = 12;
  const k = Math.round((p.x / size.w) * steps);
  return { side, offset: Math.min(steps - 1, Math.max(1, k)) / steps };
}

export const horizontal = (side: Side) => side === "left" || side === "right";
