// Editor graph state helpers: conversion between React Flow and the API format.

import type { Edge, Node } from "@xyflow/react";
import type { BlockInfo, Catalogue, EditorGraph, InstancePort, JsonSchema, PortPlace, Probe, Validation } from "./api";

export interface BlockData extends Record<string, unknown> {
  type: string;
  params: Record<string, unknown>;
  spec?: Record<string, unknown>;
  inputs: InstancePort[];
  outputs: InstancePort[];
  probed: string[];
  /** Ports moved by the user ("in:name" / "out:name"); the others use the default place. */
  ports?: Record<string, PortPlace>;
  error?: boolean;
}

export type BlockNodeType = Node<BlockData, "block">;

export function portsFromCatalogue(info: BlockInfo | undefined): { inputs: InstancePort[]; outputs: InstancePort[] } {
  if (!info) return { inputs: [], outputs: [] };
  return {
    inputs: Object.entries(info.inputs).map(([name, p]) => ({ name, kinds: p.kinds, required: p.required })),
    outputs: Object.entries(info.outputs).map(([name, p]) => ({ name, kinds: p.kinds })),
  };
}

export function defaultParams(schema: JsonSchema): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, prop] of Object.entries(schema.properties ?? {})) {
    if ("default" in prop) out[key] = prop.default;
  }
  return out;
}

export function uniqueName(base: string, taken: Set<string>): string {
  const clean = base.replace(/[^A-Za-z0-9_]/g, "_");
  if (!taken.has(clean)) return clean;
  let i = 2;
  while (taken.has(`${clean}_${i}`)) i += 1;
  return `${clean}_${i}`;
}

export function splitRef(ref: string): [string, string] {
  const dot = ref.indexOf(".");
  return [ref.slice(0, dot), ref.slice(dot + 1)];
}

export function edgeId(source: string, sourceHandle: string, target: string, targetHandle: string): string {
  return `${source}.${sourceHandle}->${target}.${targetHandle}`;
}

export function toEditorGraph(
  nodes: BlockNodeType[],
  edges: Edge[],
  meta: { name: string; description?: string; category?: string; seed: number; batch_size: number; probes: Probe[] },
): EditorGraph {
  const blocks: EditorGraph["blocks"] = {};
  for (const n of nodes) {
    blocks[n.id] = {
      type: n.data.type,
      params: n.data.params,
      position: { x: n.position.x, y: n.position.y },
      ...(n.data.ports && Object.keys(n.data.ports).length ? { ports: n.data.ports } : {}),
      ...(n.data.spec ? { spec: n.data.spec } : {}),
    };
  }
  return {
    name: meta.name,
    description: meta.description,
    category: meta.category,
    seed: meta.seed,
    batch_size: meta.batch_size,
    blocks,
    edges: edges.map((e) => ({ source: `${e.source}.${e.sourceHandle}`, target: `${e.target}.${e.targetHandle}` })),
    probes: meta.probes.filter((p) => blocks[splitRef(p.key)[0]]),
  };
}

export function fromEditorGraph(graph: EditorGraph, catalogue: Catalogue): { nodes: BlockNodeType[]; edges: Edge[] } {
  const ports = graph.validation?.ports ?? {};
  const probedBy: Record<string, string[]> = {};
  for (const p of graph.probes ?? []) {
    const [b, port] = splitRef(p.key);
    (probedBy[b] ??= []).push(port);
  }
  const nodes: BlockNodeType[] = Object.entries(graph.blocks).map(([name, b]) => {
    const fallback = portsFromCatalogue(catalogue.blocks[b.type]);
    const inst = ports[name];
    return {
      id: name,
      type: "block",
      position: b.position ?? { x: 0, y: 0 },
      data: {
        type: b.type,
        params: b.params ?? {},
        spec: b.spec,
        inputs: inst?.inputs ?? fallback.inputs,
        outputs: inst?.outputs ?? fallback.outputs,
        probed: probedBy[name] ?? [],
        ports: b.ports,
      },
    };
  });
  const edges: Edge[] = graph.edges.map((e) => {
    const [s, sh] = splitRef(e.source);
    const [t, th] = splitRef(e.target);
    return { id: edgeId(s, sh, t, th), source: s, sourceHandle: sh, target: t, targetHandle: th };
  });
  return { nodes, edges };
}

/** Apply validation results: instance ports (subsystems) and edge kinds. */
export function applyValidation(nodes: BlockNodeType[], validation: Validation): BlockNodeType[] {
  const failing = new Set<string>();
  for (const err of validation.errors) {
    for (const n of nodes) if (err.includes(`'${n.id}'`) || err.includes(`'${n.id}.`)) failing.add(n.id);
  }
  return nodes.map((n) => {
    const inst = validation.ports[n.id];
    const outputs = inst
      ? inst.outputs.map((o) => ({ ...o, kinds: validation.kinds[`${n.id}.${o.name}`] ?? o.kinds }))
      : n.data.outputs;
    return { ...n, data: { ...n.data, inputs: inst?.inputs ?? n.data.inputs, outputs, error: failing.has(n.id) } };
  });
}

/** Numeric parameters of every block, as "block.param" paths (for sweeps). */
export function numericParams(nodes: BlockNodeType[], catalogue: Catalogue): string[] {
  const out: string[] = [];
  for (const n of nodes) {
    const props = catalogue.blocks[n.data.type]?.params.properties ?? {};
    for (const [key, schema] of Object.entries(props)) {
      const types = [schema.type, ...(schema.anyOf ?? []).map((s) => s.type)];
      if (types.includes("number") || types.includes("integer")) out.push(`${n.id}.${key}`);
    }
  }
  return out;
}
