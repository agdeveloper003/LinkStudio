import { createContext, useContext } from "react";
import type { Catalogue, PortPlace, ProbeResult } from "./api";

export const CatalogueContext = createContext<Catalogue | null>(null);

export function useCatalogue(): Catalogue | null {
  return useContext(CatalogueContext);
}

/** Move a port of a block to a new place on its border (Ctrl + drag on the port). */
/** Data of the last live frame or simulation, drawn inside the blocks. */
export interface RunData {
  /** Scope plots by scope input ("scope.in1"). */
  probes: Record<string, ProbeResult>;
  /** Block metrics ("ber.bit_errors"). */
  metrics: Record<string, number | null>;
  /** Extra live data of blocks with a custom look (mimo_channel gains...). */
  displays: Record<string, Record<string, unknown>>;
  /** A live session is streaming (animations in the displays). */
  running: boolean;
}

export const EMPTY_RUN: RunData = { probes: {}, metrics: {}, displays: {}, running: false };
export const RunDataContext = createContext<RunData>(EMPTY_RUN);

/** Block edits made from inside a block (scope channel, display size). */
export interface BlockActions {
  setParam: (node: string, param: string, value: unknown) => void;
  resize: (node: string, size: { w: number; h: number }) => void;
}

export const BlockActionsContext = createContext<BlockActions>({ setParam: () => {}, resize: () => {} });

export const PortMoveContext = createContext<(node: string, key: string, place: PortPlace) => void>(() => {});
