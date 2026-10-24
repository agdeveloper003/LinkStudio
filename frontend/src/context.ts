import { createContext, useContext } from "react";
import type { Catalogue } from "./api";

export const CatalogueContext = createContext<Catalogue | null>(null);

export function useCatalogue(): Catalogue | null {
  return useContext(CatalogueContext);
}
