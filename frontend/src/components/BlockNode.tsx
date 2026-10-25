import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { memo, useContext, useEffect } from "react";
import type { InstancePort, PortPlace, Side } from "../api";
import { PortMoveContext, useCatalogue } from "../context";
import { blockSize, placeFromPoint, placePoint, portKey, portPlaces } from "../geometry";
import type { BlockNodeType } from "../graph";
import { CATEGORY_COLORS, kindColor } from "../theme";

function summary(params: Record<string, unknown>, keys: string[]): string {
  return keys
    .filter((k) => params[k] !== undefined && params[k] !== null)
    .slice(0, 2)
    .map((k) => `${k}=${typeof params[k] === "number" ? +(params[k] as number).toPrecision(4) : String(params[k])}`)
    .join("  ");
}

const HEADLINE: Record<string, string[]> = {
  mapper: ["scheme", "order"],
  awgn: ["snr_db", "mode"],
  random_bits: ["n_bits"],
  pulse_shaper: ["shape", "rolloff"],
  ofdm_modulator: ["n_fft", "n_active"],
  multipath: ["preset", "variation"],
  flat_fading: ["model", "variation"],
  demapper: ["method"],
  mimo_channel: ["n_rx", "variation"],
  mimo_detector: ["method"],
  conv_encoder: ["constraint_length"],
  carrier_offset: ["cfo_hz", "phase_deg"],
  power_amplifier: ["model", "input_backoff_db"],
  line_encoder: ["code"],
  line_decoder: ["code"],
  line_block_encoder: ["code"],
  copper_line: ["preset", "length_m"],
  crosstalk: ["model", "n_disturbers"],
  dmt_modulator: ["loading", "line_length_m"],
  ctle: ["zero_hz", "pole1_hz"],
  dfe_equalizer: ["ffe_taps", "dfe_taps"],
  intensity_modulator: ["power_dbm", "extinction_ratio_db"],
  mach_zehnder_modulator: ["laser_power_dbm", "drive"],
  dml: ["bias_ma", "modulation_ma"],
  vcsel: ["bias_ma", "ambient_c"],
  iq_modulator: ["power_dbm", "linewidth_hz"],
  fiber: ["length_km", "fiber_type"],
  optical_link: ["n_spans", "span_km"],
  optical_amplifier: ["mode", "noise_figure_db"],
  optical_attenuator: ["loss_db", "split_ways"],
  photodiode: ["responsivity", "apd_gain"],
  coherent_receiver: ["frequency_offset_hz", "lo_linewidth_hz"],
  adaptive_mimo_equalizer: ["algorithm", "n_taps"],
  phase_recovery: ["method"],
};

const POSITION: Record<Side, Position> = { left: Position.Left, right: Position.Right, top: Position.Top, bottom: Position.Bottom };

interface PortProps {
  node: string;
  dir: "in" | "out";
  port: InstancePort;
  place: PortPlace;
  size: { w: number; h: number };
  probed: boolean;
}

/** One handle on the block border with its label inside the block. Ctrl + drag moves it. */
function PortHandle({ node, dir, port, place, size, probed }: PortProps) {
  const movePort = useContext(PortMoveContext);
  const pt = placePoint(place, size);
  const key = portKey(dir, port.name);

  const startMove = (e: React.MouseEvent) => {
    if (!(e.ctrlKey || e.metaKey) || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const el = (e.currentTarget as HTMLElement).closest(".react-flow__node") as HTMLElement | null;
    if (!el) return;
    const move = (ev: MouseEvent) => {
      const r = el.getBoundingClientRect();
      const zoom = r.width / size.w || 1;
      movePort(node, key, placeFromPoint({ x: (ev.clientX - r.left) / zoom, y: (ev.clientY - r.top) / zoom }, size));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.classList.remove("moving-port");
    };
    document.body.classList.add("moving-port");
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const optional = dir === "in" && port.required === false;
  return (
    <>
      <Handle
        type={dir === "in" ? "target" : "source"}
        position={POSITION[place.side]}
        id={port.name}
        className="handle"
        style={{
          left: pt.x,
          top: pt.y,
          right: "auto",
          bottom: "auto",
          background: kindColor(port.kinds),
          borderStyle: optional ? "dashed" : "solid",
        }}
        title={`${port.name}: ${port.kinds}${optional ? " (optional)" : ""} - Ctrl + drag to move`}
        onMouseDownCapture={startMove}
      />
      <span className={`port-label side-${place.side}${optional ? " optional" : ""}`} style={{ left: pt.x, top: pt.y }}>
        {probed && <span className="probe-dot" title="probed">●</span>}
        {port.name}
      </span>
    </>
  );
}

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockNodeType>) {
  const catalogue = useCatalogue();
  const updateInternals = useUpdateNodeInternals();
  const info = catalogue?.blocks[data.type];
  const category = data.type === "subsystem" ? "core" : info?.category ?? "core";
  const color = CATEGORY_COLORS[category] ?? "#94a3b8";
  const headline = summary(data.params, HEADLINE[data.type] ?? Object.keys(data.params).slice(0, 2));
  const places = portPlaces(data);
  const size = blockSize(data, places);
  const placesKey = JSON.stringify(places);

  // Handles moved without a size change: React Flow must re-measure them for the links.
  useEffect(() => updateInternals(id), [id, placesKey, updateInternals]);

  return (
    <div
      className={`block-node${selected ? " selected" : ""}${data.error ? " error" : ""}`}
      style={{ width: size.w, height: size.h }}
    >
      <div className="block-node-header" style={{ borderTopColor: color }}>
        <span className="block-node-name">{id}</span>
        <span className="block-node-type" style={{ color }}>{data.type}</span>
      </div>
      <div className="block-node-summary">{headline}</div>
      {data.inputs.map((p) => (
        <PortHandle key={`in:${p.name}`} node={id} dir="in" port={p} place={places[portKey("in", p.name)]} size={size} probed={false} />
      ))}
      {data.outputs.map((p) => (
        <PortHandle
          key={`out:${p.name}`}
          node={id}
          dir="out"
          port={p}
          place={places[portKey("out", p.name)]}
          size={size}
          probed={data.probed.includes(p.name)}
        />
      ))}
    </div>
  );
}

export const BlockNode = memo(BlockNodeImpl);
