import { useEffect, useMemo, useRef, useState } from "react";
import type { BlockInfo, Catalogue } from "../api";
import { blockSize, bodySpec } from "../geometry";
import { defaultParams, portsFromCatalogue, type BlockData } from "../graph";
import { CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_ORDER } from "../theme";
import { BlockView, PreviewContext } from "./BlockNode";
import { PanelTitle } from "./common";

interface Props {
  catalogue: Catalogue;
  onAdd: (type: string) => void;
  onClose: () => void;
}

export const DRAG_MIME = "application/linkstudio-block";
const VIEW_KEY = "linkstudio.paletteView";
const ZOOM_KEY = "linkstudio.paletteZoom";
const PAD = 7; // room for the handles around a block picture

type View = "tree" | "grid";

function loadView(): View {
  try {
    return window.localStorage.getItem(VIEW_KEY) === "grid" ? "grid" : "tree";
  } catch {
    return "tree";
  }
}

function loadZoom(): number {
  try {
    const z = Number(window.localStorage.getItem(ZOOM_KEY));
    return z >= 0.35 && z <= 1 ? z : 0.7;
  } catch {
    return 0.7;
  }
}

/** Width of an element, kept up to date. */
function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

/** A block of the library drawn exactly as on the canvas (default parameters), scaled down. */
function BlockPicture({ type, info, zoom, maxWidth }: { type: string; info: BlockInfo; zoom: number; maxWidth: number }) {
  const data = useMemo<BlockData>(() => ({ type, params: defaultParams(info.params), ...portsFromCatalogue(info), probed: [] }), [type, info]);
  const size = blockSize(data);
  const above = bodySpec(type)?.frameless ? 16 : 0; // name label over a frameless drawing
  const scale = Math.max(0.2, Math.min(zoom, maxWidth > 0 ? (maxWidth - 2 * PAD) / size.w : zoom));
  return (
    <div className="block-picture" style={{ width: size.w * scale + 2 * PAD, height: (size.h + above) * scale + 2 * PAD }}>
      <div className="block-picture-inner" style={{ width: size.w, height: size.h, transform: `translate(${PAD}px, ${PAD + above * scale}px) scale(${scale})` }}>
        <PreviewContext.Provider value={true}>
          <BlockView id={`palette:${type}`} name={type} data={data} />
        </PreviewContext.Provider>
      </div>
    </div>
  );
}

interface Group {
  category: string;
  count: number;
  subgroups: { subcategory: string; blocks: string[] }[];
}

/**
 * Block library: searchable, grouped by class and subclass, drag onto the canvas or
 * double-click to add. The tree view lists the names; the grid view shows the blocks.
 */
export function Palette({ catalogue, onAdd, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [view, setViewState] = useState<View>(loadView);
  const [zoom, setZoomState] = useState(loadZoom);
  const [listRef, listWidth] = useWidth<HTMLDivElement>();

  const setZoom = (z: number) => {
    setZoomState(z);
    try {
      window.localStorage.setItem(ZOOM_KEY, String(z));
    } catch {
      /* private mode: not remembered */
    }
  };

  const setView = (v: View) => {
    setViewState(v);
    try {
      window.localStorage.setItem(VIEW_KEY, v);
    } catch {
      /* private mode: not remembered */
    }
  };

  const groups = useMemo<Group[]>(() => {
    const q = query.trim().toLowerCase();
    const byCat: Record<string, Record<string, string[]>> = {};
    for (const [name, info] of Object.entries(catalogue.blocks)) {
      const haystack = [name, info.label, info.subcategory, CATEGORY_LABELS[info.category] ?? info.category, info.doc];
      if (q && !haystack.some((s) => s?.toLowerCase().includes(q))) continue;
      ((byCat[info.category] ??= {})[info.subcategory] ??= []).push(name);
    }
    const label = (n: string) => catalogue.blocks[n].label;
    const cats = Object.keys(byCat).sort((a, b) => {
      const ia = CATEGORY_ORDER.indexOf(a);
      const ib = CATEGORY_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    return cats.map((c) => {
      const subgroups = Object.keys(byCat[c])
        .sort()
        .map((s) => ({ subcategory: s, blocks: byCat[c][s].sort((a, b) => label(a).localeCompare(label(b))) }));
      return { category: c, count: subgroups.reduce((n, s) => n + s.blocks.length, 0), subgroups };
    });
  }, [catalogue, query]);

  const item = (name: string) => {
    const info = catalogue.blocks[name];
    return (
      <div
        key={name}
        className={view === "grid" ? "palette-card" : "palette-item"}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData(DRAG_MIME, name);
          e.dataTransfer.effectAllowed = "move";
        }}
        onDoubleClick={() => onAdd(name)}
        title={`${info.label} (${name})\n\n${info.doc}\n\nDrag onto the canvas or double-click to add.`}
      >
        {view === "grid" ? <BlockPicture type={name} info={info} zoom={zoom} maxWidth={listWidth - 40} /> : info.label}
      </div>
    );
  };

  return (
    <aside className="palette">
      <PanelTitle title="Blocks" onClose={onClose} shortcut="Ctrl+B" />
      <div className="palette-tools">
        <input className="search" placeholder="Search blocks…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="view-switch">
          <button className={view === "tree" ? "active" : ""} onClick={() => setView("tree")} title="Tree view: block names by class">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">
              <rect x="1" y="2" width="3" height="2.4" rx="0.6" />
              <rect x="5.5" y="2" width="9.5" height="2.4" rx="0.6" />
              <rect x="3" y="6.8" width="2.4" height="2.4" rx="0.6" />
              <rect x="7" y="6.8" width="8" height="2.4" rx="0.6" />
              <rect x="3" y="11.6" width="2.4" height="2.4" rx="0.6" />
              <rect x="7" y="11.6" width="8" height="2.4" rx="0.6" />
            </svg>
          </button>
          <button className={view === "grid" ? "active" : ""} onClick={() => setView("grid")} title="Grid view: the blocks themselves">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">
              <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" />
              <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" />
              <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" />
              <rect x="9" y="9" width="5.5" height="5.5" rx="1" />
            </svg>
          </button>
        </div>
      </div>
      {view === "grid" && (
        <label className="palette-zoom" title="Size of the block pictures">
          <span>Zoom</span>
          <input type="range" min={0.35} max={1} step={0.05} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} />
          <span className="palette-zoom-value">{Math.round(zoom * 100)}%</span>
        </label>
      )}
      <div className="palette-list" ref={listRef}>
        {groups.map(({ category, count, subgroups }) => (
          <div key={category} className="palette-group">
            <button
              className="palette-cat"
              onClick={() => setCollapsed((c) => ({ ...c, [category]: !c[category] }))}
            >
              <span className="cat-dot" style={{ background: CATEGORY_COLORS[category] ?? "#94a3b8" }} />
              {CATEGORY_LABELS[category] ?? category.toUpperCase()}
              <span className="count">{count}</span>
              <span className="chev">{collapsed[category] && !query ? "▸" : "▾"}</span>
            </button>
            {(!collapsed[category] || query) &&
              subgroups.map(({ subcategory, blocks }) => (
                <div key={subcategory} className="palette-subgroup">
                  {subcategory && <div className="palette-subcat">{subcategory}</div>}
                  <div className={view === "grid" ? "palette-grid" : undefined}>{blocks.map((name) => item(name))}</div>
                </div>
              ))}
          </div>
        ))}
      </div>
    </aside>
  );
}
