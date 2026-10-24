import { useEffect, useMemo, useRef, useState } from "react";
import type { ScenarioItem } from "../api";

interface Props {
  scenarios: ScenarioItem[];
  current: string;
  onOpen: (path: string) => void;
  onBlank: () => void;
  onClose: () => void;
}

const ORDER = ["Tutorials", "Wireless", "Coding", "Wireline", "Optical", "Examples (CLI)", "Subsystems"];

/** Modal scenario gallery: categories, descriptions and search. */
export function ScenarioBrowser({ scenarios, current, onOpen, onBlank, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<string>("All");
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    search.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const categories = useMemo(() => {
    const cats = Array.from(new Set(scenarios.map((s) => s.category)));
    return cats.sort((a, b) => (ORDER.indexOf(a) + 100 * +(ORDER.indexOf(a) < 0)) - (ORDER.indexOf(b) + 100 * +(ORDER.indexOf(b) < 0)));
  }, [scenarios]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return scenarios.filter(
      (s) =>
        (category === "All" || s.category === category) &&
        (!q || `${s.name} ${s.description} ${s.path}`.toLowerCase().includes(q)),
    );
  }, [scenarios, query, category]);

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Open a scenario</h2>
          <input ref={search} className="search" placeholder="Search scenarios…" value={query} onChange={(e) => setQuery(e.target.value)} />
          <button className="icon-btn" onClick={onClose} title="Close (Esc)">×</button>
        </div>
        <div className="modal-body">
          <nav className="modal-cats">
            {["All", ...categories].map((c) => (
              <button key={c} className={c === category ? "active" : ""} onClick={() => setCategory(c)}>
                {c}
                <span className="count">{c === "All" ? scenarios.length : scenarios.filter((s) => s.category === c).length}</span>
              </button>
            ))}
            <button className="blank" onClick={onBlank}>+ Blank scenario</button>
          </nav>
          <div className="modal-grid">
            {categories
              .filter((c) => category === "All" || c === category)
              .map((c) => {
                const items = filtered.filter((s) => s.category === c);
                if (!items.length) return null;
                return (
                  <section key={c}>
                    <h3>{c}</h3>
                    <div className="cards">
                      {items.map((s) => (
                        <button
                          key={s.path}
                          className={`card${s.path === current ? " current" : ""}`}
                          onClick={() => onOpen(s.path)}
                        >
                          <div className="card-title">{s.name}</div>
                          <div className="card-desc">{s.description || s.path}</div>
                          <div className="card-meta">{s.blocks} blocks · {s.path}</div>
                        </button>
                      ))}
                    </div>
                  </section>
                );
              })}
            {!filtered.length && <div className="muted">No scenario matches “{query}”.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
