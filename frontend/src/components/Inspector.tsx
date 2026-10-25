import { useEffect, useState } from "react";
import type { Catalogue, EditorGraph, Validation } from "../api";
import type { BlockNodeType } from "../graph";
import { CATEGORY_COLORS, CATEGORY_LABELS } from "../theme";
import { FrequencyResponse } from "./FrequencyResponse";
import { ParamField } from "./ParamField";
import { PanelTitle } from "./common";

interface Props {
  catalogue: Catalogue;
  node: BlockNodeType | null;
  validation: Validation | null;
  scenario: { name: string; description: string; category: string; categories: string[]; blocks: number; edges: number };
  onParam: (node: string, param: string, value: unknown) => void;
  onRename: (oldName: string, newName: string) => string | null;
  onDelete: (node: string) => void;
  onDuplicate: (node: string) => void;
  onFlip: (node: string) => void;
  onScenarioChange: (patch: { name?: string; description?: string; category?: string }) => void;
  onClose: () => void;
  getGraph: () => EditorGraph;
  /** Key of the graph structure and parameters (refreshes the frequency response). */
  structureKey: string;
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
      <PanelTitle title="Inspector" onClose={props.onClose} shortcut="Ctrl+I" />
      <RenameField name={node.id} onRename={(n) => props.onRename(node.id, n)} />
      <div className="block-type">
        <span className="cat-dot" style={{ background: CATEGORY_COLORS[category] }} />
        <span className="block-type-label">{info?.label ?? node.data.type}</span>
        <code className="muted">{node.data.type}</code>
      </div>
      <div className="block-class muted">
        {CATEGORY_LABELS[category] ?? category}
        {info?.subcategory ? ` › ${info.subcategory}` : ""}
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

      <FrequencyResponse block={node.id} getGraph={props.getGraph} refreshKey={props.structureKey} />

      <div className="block-actions">
        <button onClick={() => props.onFlip(node.id)} title="Turn the block around: inputs and outputs swap sides (F)">Flip</button>
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

function ScenarioInfo({ scenario, validation, onScenarioChange, onClose }: Props) {
  return (
    <aside className="inspector">
      <PanelTitle title="Scenario" onClose={onClose} shortcut="Ctrl+I" />
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
          <li>Connect an output to an input; colors show the signal kind.</li>
          <li>Ctrl + drag a port to move it around the block border.</li>
          <li>Blocks snap so that connected ports line up; links route around blocks.</li>
          <li>Select a block to edit its parameters.</li>
          <li>Pick <b>Live</b>, <b>Simulate</b> or <b>Sweep</b> in the top bar, set it up with ⚙ and press run; results open in a floating window.</li>
          <li><b>Live</b> reacts to parameter changes instantly; <b>Sweep</b> draws curves vs a parameter.</li>
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
            <tr><td>Ctrl+B / Ctrl+I</td><td>show / hide blocks / inspector</td></tr>
            <tr><td>Del</td><td>delete selection</td></tr>
          </tbody>
        </table>
      </div>
    </aside>
  );
}
