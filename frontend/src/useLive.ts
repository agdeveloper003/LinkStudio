import { useCallback, useEffect, useRef, useState } from "react";
import { liveSocketUrl, type EditorGraph, type Probe, type ProbeResult } from "./api";

export interface LiveFrame {
  batch: number;
  metrics: Record<string, number | null>;
  probes: Record<string, ProbeResult>;
  errors: string[];
  step_ms: number;
}

export interface LiveState {
  running: boolean;
  frame: LiveFrame | null;
  error: string | null;
  rate: number;
  start: (graph: EditorGraph) => void;
  stop: () => void;
  setParam: (path: string, value: unknown) => void;
  setProbes: (probes: Probe[]) => void;
  setRate: (rate: number) => void;
  resetMetrics: () => void;
}

/** WebSocket session with the live simulation endpoint. */
export function useLive(): LiveState {
  const ws = useRef<WebSocket | null>(null);
  const [running, setRunning] = useState(false);
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rate, setRateState] = useState(8);

  const send = useCallback((msg: unknown) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(msg));
  }, []);

  const stop = useCallback(() => {
    send({ type: "stop" });
    ws.current?.close();
    ws.current = null;
    setRunning(false);
  }, [send]);

  const start = useCallback(
    (graph: EditorGraph) => {
      ws.current?.close();
      setError(null);
      setFrame(null);
      const socket = new WebSocket(liveSocketUrl());
      ws.current = socket;
      socket.onopen = () => socket.send(JSON.stringify({ type: "start", graph, rate }));
      socket.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "started") setRunning(true);
        else if (msg.type === "frame") {
          setFrame(msg as LiveFrame);
          if (msg.errors?.length) setError(msg.errors.join("; "));
        } else if (msg.type === "error") {
          setError(msg.error);
          setRunning(false);
        } else if (msg.type === "stopped") setRunning(false);
      };
      socket.onerror = () => setError("Connection to the simulation server failed.");
      socket.onclose = () => {
        if (ws.current === socket) {
          ws.current = null;
          setRunning(false);
        }
      };
    },
    [rate],
  );

  useEffect(() => () => ws.current?.close(), []);

  return {
    running,
    frame,
    error,
    rate,
    start,
    stop,
    setParam: (path, value) => send({ type: "set_param", path, value }),
    setProbes: (probes) => send({ type: "probes", probes }),
    setRate: (r) => {
      setRateState(r);
      send({ type: "rate", rate: r });
    },
    resetMetrics: () => send({ type: "reset" }),
  };
}
