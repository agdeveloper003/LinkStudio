import { useEffect, useState } from "react";
import type { Catalogue, Probe, Validation } from "../api";
import type { BlockNodeType } from "../graph";
import { CATEGORY_COLORS, kindColor } from "../theme";
import { ParamField } from "./ParamField";

interface Props {
  catalogue: Catalogue;
  node: BlockNodeType | null;
  probes: Probe[];
  validation: Validation | null;
  scenario: { name: string; description: string; category: string; categories: string[]; blocks: number; edges: number };
  onParam: (node: string, param: string, value: unknown) => void;
  onRename: (oldName: string, newName: string) => string | null;
  onToggleProbe: (key: string) => void;
  onDelete: (node: string) => void;
  onDuplicate: (node: string) => void;
  onScenarioChange: (patch: { name?: string; description?: string; category?: string }) => void;
}

export function Inspector(props: Props) {
  const { catalogue, node } = props;
  if (!node) return <ScenarioInfo {...props} />;
  const info = catalogue.blocks[node.data.type];
  const category = node.data.type === "subsystem" ? "core" : info?.category ?? "core";
  const properties = info?.params.properties ?? {};
  const doc = node.data.type === "subsystem" ? "Hierarchical block loaded from a scenario file." : info?.doc ?? "";

  return (
    <aside className="inspector">
      <div className="panel-title">Inspector</div>
      <RenameField name={node.id} onRename={(n) => props.onRename(node.id, n)} />
      <div className="block-type">
        <span className="cat-dot" style={{ background: CATEGORY_COLORS[category] }} />
        {node.data.type}
        <span className="muted">· {category}</span>
      </div>
      <p className="doc">{doc}</p>

      {node.data.type === "subsystem" && node.data.spec && (
        <div className="section">
          <div className="section-title">Subsystem</div>
          <pre className="spec">{JSON.stringify(node.data.spec, null, 2)}</pre>
        </div>
      )}

      {Object.keys(properties).length > 0 && (
        <div className="section">
          <div className="section-title">Parameters</div>
          {Object.entries(properties).map(([key, schema]) => (
            <ParamField
              key={key}
              name={key}
              schema={schema}
              value={node.data.params[key]}
              onChange={(v) => props.onParam(node.id, key, v)}
            />
          ))}
        </div>
      )}

      {node.data.outputs.length > 0 && (
        <div className="section">
          <div className="section-title">Probes</div>
          {node.data.outputs.map((o) => {
            const key = `${node.id}.${o.name}`;
            const on = props.probes.some((p) => p.key === key);
            return (
              <label key={o.name} className="probe-toggle">
                <input type="checkbox" checked={on} onChange={() => props.onToggleProbe(key)} />
                <span className="kind-chip" style={{ borderColor: kindColor(o.kinds), color: kindColor(o.kinds) }}>
                  {o.kinds}
                </span>
                {o.name}
              </label>
            );
          })}
        </div>
      )}

      <div className="block-actions">
        <button onClick={() => props.onDuplicate(node.id)} title="Duplicate (Ctrl+D)">Duplicate</button>
        <button className="danger" onClick={() => props.onDelete(node.id)} title="Delete (Del)">Delete</button>
      </div>
    </aside>
  );
}

function RenameField({ name, onRename }: { name: string; onRename: (n: string) => string | null }) {
  const [text, setText] = useState(name);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setText(name);
    setError(null);
  }, [name]);
  const commit = () => {
    if (text === name) return;
    const err = onRename(text);
    setError(err);
    if (err) setText(name);
  };
  return (
    <div className="rename">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
      />
      {error && <div className="field-error">{error}</div>}
    </div>
  );
}

function ScenarioInfo({ scenario, validation, onScenarioChange }: Props) {
  return (
    <aside className="inspector">
      <div className="panel-title">Scenario</div>
      <div className="param">
        <label className="param-label">name</label>
        <input className="text" value={scenario.name} onChange={(e) => onScenarioChange({ name: e.target.value })} />
      </div>
      <div className="param">
        <label className="param-label">category</label>
        <input className="text" list="scenario-categories" value={scenario.category}
               onChange={(e) => onScenarioChange({ category: e.target.value })} />
        <datalist id="scenario-categories">
          {scenario.categories.filter((c) => c !== "Subsystems").map((c) => <option key={c} value={c} />)}
        </datalist>
      </div>
      <div className="param">
        <label className="param-label">description</label>
        <textarea
          className="text"
          rows={3}
          value={scenario.description}
          onChange={(e) => onScenarioChange({ description: e.target.value })}
        />
      </div>
      <div className="stats">
        <div><b>{scenario.blocks}</b> blocks</div>
        <div><b>{scenario.edges}</b> connections</div>
      </div>
      <div className="section">
        <div className="section-title">Validation</div>
        {!validation && <div className="muted">Not validated yet.</div>}
        {validation?.ok && <div className="ok-text">Graph is valid.</div>}
        {validation?.errors.map((e, i) => (
          <div key={i} className="field-error">{e}</div>
        ))}
      </div>
      <div className="section help">
        <div className="section-title">How to</div>
        <ul>
          <li>Drag blocks from the library onto the canvas.</li>
          <li>Connect an output (right) to an input (left); colors show the signal kind.</li>
          <li>Select a block to edit its parameters and add probes on its outputs.</li>
          <li><b>Live</b> streams scopes and reacts to parameter changes instantly.</li>
          <li><b>Simulate</b> runs a Monte-Carlo job; <b>Sweep</b> draws curves vs a parameter.</li>
          <li>Shift / Ctrl + click or drag a box to select several blocks.</li>
        </ul>
      </div>
      <div className="section help">
        <div className="section-title">Shortcuts</div>
        <table className="shortcuts">
          <tbody>
            <tr><td>Ctrl+O</td><td>open a scenario</td></tr>
            <tr><td>Ctrl+S / Ctrl+Shift+S</td><td>save / save as</td></tr>
            <tr><td>Ctrl+Z / Ctrl+Y</td><td>undo / redo</td></tr>
            <tr><td>Ctrl+C / Ctrl+V</td><td>copy / paste blocks</td></tr>
            <tr><td>Ctrl+D</td><td>duplicate selection</td></tr>
            <tr><td>Ctrl+L</td><td>automatic layout</td></tr>
            <tr><td>Del</td><td>delete selection</td></tr>
          </tbody>
        </table>
      </div>
    </aside>
  );
}
