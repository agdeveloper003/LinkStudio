import { useEffect, useState, type ReactNode } from "react";
import type { JsonSchema } from "../api";

interface Props {
  name: string;
  schema: JsonSchema;
  value: unknown;
  onChange: (value: unknown) => void;
}

/** Split ``anyOf: [X, null]`` into the base schema and a nullable flag. */
function resolve(schema: JsonSchema): { base: JsonSchema; nullable: boolean } {
  if (schema.anyOf) {
    const nonNull = schema.anyOf.filter((s) => s.type !== "null");
    const nullable = nonNull.length < schema.anyOf.length;
    if (nonNull.length === 1) return { base: { ...schema, ...nonNull[0], anyOf: undefined }, nullable };
    return { base: { ...schema, type: "json", anyOf: undefined }, nullable };
  }
  return { base: schema, nullable: false };
}

function sliderRange(name: string, s: JsonSchema): [number, number] | null {
  const lo = s.minimum ?? s.exclusiveMinimum;
  const hi = s.maximum ?? s.exclusiveMaximum;
  if (lo !== undefined && hi !== undefined && hi - lo <= 1000) return [lo, hi];
  if (name.endsWith("_db")) return [-10, 40];
  return null;
}

function NumberInput({ value, integer, onCommit }: { value: number; integer: boolean; onCommit: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v)) onCommit(integer ? Math.round(v) : v);
    else setText(String(value));
  };
  return (
    <input
      className="num"
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

function JsonInput({ value, onCommit }: { value: unknown; onCommit: (v: unknown) => void }) {
  const [text, setText] = useState(JSON.stringify(value));
  const [bad, setBad] = useState(false);
  useEffect(() => setText(JSON.stringify(value)), [value]);
  const commit = () => {
    try {
      onCommit(JSON.parse(text));
      setBad(false);
    } catch {
      setBad(true);
    }
  };
  return (
    <input
      className={`text${bad ? " invalid" : ""}`}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

/** Form control generated from a parameter's JSON Schema. */
export function ParamField({ name, schema, value, onChange }: Props) {
  const { base, nullable } = resolve(schema);
  const isNull = value === null || value === undefined;
  const label = (
    <label className="param-label" title={schema.description}>
      {name}
    </label>
  );

  let control: ReactNode;
  if (base.enum) {
    control = (
      <select
        value={isNull ? "__null__" : String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "__null__") return onChange(null);
          const match = base.enum!.find((v) => String(v) === raw);
          onChange(match);
        }}
      >
        {nullable && <option value="__null__">auto</option>}
        {base.enum.map((v) => (
          <option key={String(v)} value={String(v)}>{String(v)}</option>
        ))}
      </select>
    );
  } else if (base.type === "boolean") {
    control = (
      <label className="switch">
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        <span />
      </label>
    );
  } else if (base.type === "number" || base.type === "integer") {
    const integer = base.type === "integer";
    const range = sliderRange(name, base);
    const fallback = typeof base.default === "number" ? base.default : range ? range[0] : 0;
    control = (
      <div className="num-row">
        {nullable && (
          <label className="auto-toggle" title="Automatic (from upstream metadata)">
            <input type="checkbox" checked={isNull} onChange={(e) => onChange(e.target.checked ? null : fallback)} />
            auto
          </label>
        )}
        {!isNull && (
          <>
            {range && (
              <input
                type="range"
                min={range[0]}
                max={range[1]}
                step={integer ? 1 : (range[1] - range[0]) / 200}
                value={Number(value)}
                onChange={(e) => onChange(integer ? Math.round(Number(e.target.value)) : Number(e.target.value))}
              />
            )}
            <NumberInput value={Number(value)} integer={integer} onCommit={onChange} />
          </>
        )}
      </div>
    );
  } else if (base.type === "string") {
    control = <input className="text" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />;
  } else {
    control = (
      <div className="num-row">
        {nullable && (
          <label className="auto-toggle">
            <input type="checkbox" checked={isNull} onChange={(e) => onChange(e.target.checked ? null : [])} />
            none
          </label>
        )}
        {!isNull && <JsonInput value={value} onCommit={onChange} />}
      </div>
    );
  }

  return (
    <div className="param">
      {label}
      {control}
      {schema.description && <div className="param-help">{schema.description}</div>}
    </div>
  );
}
