import { useMemo, useState } from "react";
import type { Catalogue } from "../api";
import { CATEGORY_COLORS, CATEGORY_ORDER } from "../theme";

interface Props {
  catalogue: Catalogue;
  onAdd: (type: string) => void;
}

export const DRAG_MIME = "application/linkstudio-block";

/** Block library: searchable, grouped by category, drag onto the canvas or click to add. */
export function Palette({ catalogue, onAdd }: Props) {
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const byCat: Record<string, string[]> = {};
    for (const [name, info] of Object.entries(catalogue.blocks)) {
      if (q && !name.includes(q) && !info.doc.toLowerCase().includes(q)) continue;
      (byCat[info.category] ??= []).push(name);
    }
    const cats = Object.keys(byCat).sort((a, b) => {
      const ia = CATEGORY_ORDER.indexOf(a);
      const ib = CATEGORY_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    return cats.map((c) => ({ category: c, blocks: byCat[c].sort() }));
  }, [catalogue, query]);

  return (
    <aside className="palette">
      <div className="panel-title">Blocks</div>
      <input
        className="search"
        placeholder="Search blocks…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="palette-list">
        {groups.map(({ category, blocks }) => (
          <div key={category} className="palette-group">
            <button
              className="palette-cat"
              onClick={() => setCollapsed((c) => ({ ...c, [category]: !c[category] }))}
            >
              <span className="cat-dot" style={{ background: CATEGORY_COLORS[category] ?? "#94a3b8" }} />
              {category}
              <span className="count">{blocks.length}</span>
              <span className="chev">{collapsed[category] && !query ? "▸" : "▾"}</span>
            </button>
            {(!collapsed[category] || query) &&
              blocks.map((name) => (
                <div
                  key={name}
                  className="palette-item"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(DRAG_MIME, name);
                    e.dataTransfer.effectAllowed = "move";
                  }}
                  onDoubleClick={() => onAdd(name)}
                  title={`${catalogue.blocks[name].doc}\n\nDrag onto the canvas or double-click to add.`}
                >
                  {name}
                </div>
              ))}
          </div>
        ))}
      </div>
    </aside>
  );
}
