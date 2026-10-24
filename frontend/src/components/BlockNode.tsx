import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";
import type { BlockNodeType } from "../graph";
import { CATEGORY_COLORS, kindColor } from "../theme";
import { useCatalogue } from "../context";

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
  iq_modulator: ["power_dbm", "linewidth_hz"],
  fiber: ["length_km", "nonlinear"],
  optical_link: ["n_spans", "span_km"],
  optical_amplifier: ["mode", "noise_figure_db"],
  photodiode: ["responsivity", "apd_gain"],
  coherent_receiver: ["frequency_offset_hz", "lo_linewidth_hz"],
  adaptive_mimo_equalizer: ["algorithm", "n_taps"],
  phase_recovery: ["method"],
};

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockNodeType>) {
  const catalogue = useCatalogue();
  const info = catalogue?.blocks[data.type];
  const category = data.type === "subsystem" ? "core" : info?.category ?? "core";
  const color = CATEGORY_COLORS[category] ?? "#94a3b8";
  const rows = Math.max(data.inputs.length, data.outputs.length);
  const headline = summary(data.params, HEADLINE[data.type] ?? Object.keys(data.params).slice(0, 2));

  return (
    <div className={`block-node${selected ? " selected" : ""}${data.error ? " error" : ""}`}>
      <div className="block-node-header" style={{ borderTopColor: color }}>
        <span className="block-node-name">{id}</span>
        <span className="block-node-type" style={{ color }}>{data.type}</span>
      </div>
      {headline && <div className="block-node-summary">{headline}</div>}
      <div className="block-node-ports">
        {Array.from({ length: rows }).map((_, i) => {
          const inp = data.inputs[i];
          const out = data.outputs[i];
          return (
            <div className="port-row" key={i}>
              <div className="port port-in">
                {inp && (
                  <>
                    <Handle
                      type="target"
                      position={Position.Left}
                      id={inp.name}
                      className="handle"
                      style={{ background: kindColor(inp.kinds), borderStyle: inp.required === false ? "dashed" : "solid" }}
                      title={`${inp.name}: ${inp.kinds}${inp.required === false ? " (optional)" : ""}`}
                    />
                    <span className={inp.required === false ? "optional" : ""}>{inp.name}</span>
                  </>
                )}
              </div>
              <div className="port port-out">
                {out && (
                  <>
                    <span>
                      {data.probed.includes(out.name) && <span className="probe-dot" title="probed">●</span>}
                      {out.name}
                    </span>
                    <Handle
                      type="source"
                      position={Position.Right}
                      id={out.name}
                      className="handle"
                      style={{ background: kindColor(out.kinds) }}
                      title={`${out.name}: ${out.kinds}`}
                    />
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export const BlockNode = memo(BlockNodeImpl);
