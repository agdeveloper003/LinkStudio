import { useCallback, useEffect, useRef, useState } from "react";
import { api, type EditorGraph, type Job } from "./api";

/** Start a background job and poll it until it finishes. */
export function useJob() {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const clear = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  };

  const run = useCallback(async (mode: string, graph: EditorGraph, config: Record<string, unknown>) => {
    clear();
    setError(null);
    try {
      const started = await api.startJob({ mode, graph, config });
      setJob(started);
      timer.current = window.setInterval(async () => {
        try {
          const j = await api.job(started.id);
          setJob(j);
          if (j.status !== "running" && j.status !== "queued") clear();
        } catch (e) {
          setError(String(e));
          clear();
        }
      }, 300);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const cancel = useCallback(async () => {
    if (job) await api.cancelJob(job.id);
  }, [job]);

  useEffect(() => clear, []);

  const busy = job?.status === "running" || job?.status === "queued";
  return { job, error, busy, run, cancel };
}
