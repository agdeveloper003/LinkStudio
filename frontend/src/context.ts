import { createContext, useContext } from "react";
import type { Catalogue, PortPlace } from "./api";

export const CatalogueContext = createContext<Catalogue | null>(null);

export function useCatalogue(): Catalogue | null {
  return useContext(CatalogueContext);
}

/** Move a port of a block to a new place on its border (Ctrl + drag on the port). */
export const PortMoveContext = createContext<(node: string, key: string, place: PortPlace) => void>(() => {});
