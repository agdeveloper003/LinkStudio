import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { createContext, memo, useContext, useEffect } from "react";
import type { InstancePort, PortPlace, Side } from "../api";
import { BlockActionsContext, PortMoveContext, useCatalogue } from "../context";
import { blockSize, bodySpec, isScopeType, placeFromPoint, placePoint, portKey, portPlaces, type BlockSize } from "../geometry";
import type { BlockData, BlockNodeType } from "../graph";
import { CATEGORY_COLORS, kindColor } from "../theme";
import { ScopeScreen, ScopeSelector, displayFor, useScopePorts } from "./BlockDisplays";

/** Set by the library: blocks are drawn as pictures, without React Flow handles. */
export const PreviewContext = createContext(false);

const POSITION: Record<Side, Position> = { left: Position.Left, right: Position.Right, top: Position.Top, bottom: Position.Bottom };

interface PortProps {
  node: string;
  dir: "in" | "out";
  port: InstancePort;
  place: PortPlace;
  size: BlockSize;
  /** Hide the port name (scope blocks, drawn link media). */
  bare?: boolean;
}

/** One handle on the block border with its label inside the block. Ctrl + drag moves it. */
function PortHandle({ node, dir, port, place, size, bare }: PortProps) {
  const movePort = useContext(PortMoveContext);
  const preview = useContext(PreviewContext);
  const pt = placePoint(place, size);
  const key = portKey(dir, port.name);
  const optional = dir === "in" && port.required === false;
  if (preview) {
    const style = { left: pt.x, top: pt.y, background: kindColor(port.kinds), borderStyle: optional ? "dashed" : "solid" };
    return <span className="handle preview-handle" style={style} />;
  }

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
      {!bare && (
        <span className={`port-label side-${place.side}${optional ? " optional" : ""}`} style={{ left: pt.x, top: pt.y }}>
          {port.name}
        </span>
      )}
    </>
  );
}

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockNodeType>) {
  return <BlockView id={id} data={data} selected={selected} />;
}

/**
 * A block as drawn on the canvas. The library draws the same component inside a
 * PreviewContext, so every design shows up there as it is in the sandbox.
 */
export function BlockView({ id, name = id, data, selected }: { id: string; name?: string; data: BlockData; selected?: boolean }) {
  const catalogue = useCatalogue();
  const preview = useContext(PreviewContext);
  const updateInternals = useUpdateNodeInternals();
  const info = catalogue?.blocks[data.type];
  const category = data.type === "subsystem" ? "core" : info?.category ?? "core";
  const color = CATEGORY_COLORS[category] ?? "#94a3b8";
  const scope = isScopeType(data.type);
  const body = bodySpec(data.type);
  const Display = displayFor(data.type);
  const places = portPlaces(data);
  const size = blockSize(data, places);
  const placesKey = JSON.stringify(places);
  const scopePorts = useScopePorts(id, scope ? data.inputs : []);
  const inputs = scope ? data.inputs.filter((p) => scopePorts.has(p.name)) : data.inputs;

  // Handles moved, added or removed without a size change: React Flow must re-measure them.
  useEffect(() => {
    if (!preview) updateInternals(id);
  }, [id, placesKey, inputs.length, updateInternals, preview]);

  if (body?.frameless && Display) {
    // Link media: only the drawing and its ports, no frame, header or readouts.
    const bare = (name: string) => name === "in" || name === "out";
    return (
      <div
        className={`frameless-node display-${data.type}${selected ? " selected" : ""}${data.error ? " error" : ""}`}
        style={{ width: size.w, height: size.h, ["--c" as string]: color }}
        title={`${id} - ${info?.label ?? data.type}`}
      >
        <span className="frameless-name">{name}</span>
        {Display({ id, data, size })}
        {data.inputs.map((p) => (
          <PortHandle key={`in:${p.name}`} node={id} dir="in" port={p} place={places[portKey("in", p.name)]} size={size} bare={bare(p.name)} />
        ))}
        {data.outputs.map((p) => (
          <PortHandle key={`out:${p.name}`} node={id} dir="out" port={p} place={places[portKey("out", p.name)]} size={size} bare={bare(p.name)} />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`block-node${scope ? " scope-node" : ""}${Display ? ` display-node display-${data.type}` : ""}${selected ? " selected" : ""}${data.error ? " error" : ""}`}
      style={{ width: size.w, height: size.h, ["--c" as string]: color }}
    >
      <div className="block-node-header" style={{ borderTopColor: color }}>
        <span className="block-node-name">{name}</span>
        <span className="block-node-type" style={{ color }} title={data.type}>{info?.label ?? data.type}</span>
        {scope && <ScopeSelector id={id} data={data} />}
      </div>
      {scope && <ScopeScreen id={id} data={data} />}
      {Display && <div className="display-body">{Display({ id, data, size })}</div>}
      {inputs.map((p) => (
        <PortHandle key={`in:${p.name}`} node={id} dir="in" port={p} place={places[portKey("in", p.name)]} size={size} bare={scope} />
      ))}
      {data.outputs.map((p) => (
        <PortHandle key={`out:${p.name}`} node={id} dir="out" port={p} place={places[portKey("out", p.name)]} size={size} />
      ))}
      {body?.resizable && !preview && <ResizeCorner id={id} size={size} />}
    </div>
  );
}

export const BlockNode = memo(BlockNodeImpl);

/** Bottom-right grip of a resizable block (scopes, displays): drag to change its size. */
function ResizeCorner({ id, size }: { id: string; size: BlockSize }) {
  const { resize } = useContext(BlockActionsContext);
  const start = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const el = (e.currentTarget as HTMLElement).closest(".react-flow__node") as HTMLElement | null;
    const zoom = el ? el.getBoundingClientRect().width / size.w || 1 : 1;
    const x0 = e.clientX;
    const y0 = e.clientY;
    const move = (ev: MouseEvent) =>
      resize(id, { w: Math.round(size.w + (ev.clientX - x0) / zoom), h: Math.round(size.h + (ev.clientY - y0) / zoom) });
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.classList.remove("resizing-block");
    };
    document.body.classList.add("resizing-block");
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  return <div className="resize-corner nodrag nopan" onMouseDown={start} title="Drag to resize" />;
}
