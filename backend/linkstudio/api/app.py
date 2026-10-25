"""FastAPI application: block catalogue, scenarios, validation, jobs and live streaming.

Run with ``python -m linkstudio serve`` (serves the built UI too) or, for UI development,
``uvicorn linkstudio.api.app:app --reload`` next to ``npm run dev`` in ``frontend/``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import linkstudio
from linkstudio.api.editor import (
    auto_layout_editor, build_graph, editor_to_python, editor_to_yaml_dict, spec_to_editor, validate_editor,
)
from linkstudio.api.jobs import THEORY_CURVES, JobManager, _clean
from linkstudio.api.probes import block_displays, compute_probes, scope_probes
from linkstudio.api.response import block_response
from linkstudio.core import Simulation, block_catalogue
from linkstudio.core.parallel import cpu_count

BACKEND_DIR = Path(linkstudio.__file__).resolve().parents[1]
SCENARIO_DIR = BACKEND_DIR / "scenarios"
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


def _scenario_path(rel: str) -> Path:
    path = (SCENARIO_DIR / rel).resolve()
    if not path.is_relative_to(SCENARIO_DIR.resolve()):
        raise HTTPException(400, "Path outside the scenario folder.")
    if path.suffix not in (".yaml", ".yml"):
        path = path.with_suffix(".yaml")
    return path


def create_app() -> FastAPI:
    app = FastAPI(title="LinkStudio", version=linkstudio.__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    jobs = JobManager(SCENARIO_DIR)

    # ------------------------------------------------------------ catalogue
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": linkstudio.__version__, "cpu_count": cpu_count()}

    @app.get("/api/blocks")
    def blocks() -> dict[str, Any]:
        return {"blocks": block_catalogue(), "theory": {k: v[0] for k, v in THEORY_CURVES.items()}}

    # ------------------------------------------------------------ scenarios
    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict[str, Any]]:
        items = []
        for path in sorted(SCENARIO_DIR.rglob("*.yaml")):
            rel = path.relative_to(SCENARIO_DIR).as_posix()
            try:
                spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            subsystem = rel.startswith("subsystems/") or "outputs" in spec
            category = "Subsystems" if subsystem else spec.get("category") or "Examples"
            items.append({"path": rel, "name": spec.get("name", path.stem), "category": category,
                          "description": spec.get("description", ""), "subsystem": subsystem,
                          "blocks": len(spec.get("blocks") or {})})
        return items

    @app.post("/api/layout")
    def layout(editor: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"positions": auto_layout_editor(editor, SCENARIO_DIR)}
        except Exception as exc:
            raise HTTPException(400, f"Cannot lay out: {exc}") from exc

    @app.post("/api/export/python")
    def export_python(editor: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"code": editor_to_python(editor)}
        except Exception as exc:
            raise HTTPException(400, f"Cannot export: {exc}") from exc

    @app.get("/api/scenarios/{rel:path}")
    def get_scenario(rel: str) -> dict[str, Any]:
        path = _scenario_path(rel)
        if not path.exists():
            raise HTTPException(404, f"Scenario '{rel}' not found.")
        spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            editor = spec_to_editor(spec, path.parent)
        except Exception as exc:
            raise HTTPException(400, f"Cannot load scenario: {exc}") from exc
        editor["path"] = path.relative_to(SCENARIO_DIR).as_posix()
        return editor

    @app.put("/api/scenarios/{rel:path}")
    def save_scenario(rel: str, editor: dict[str, Any]) -> dict[str, Any]:
        path = _scenario_path(rel)
        try:
            data = editor_to_yaml_dict(editor)
        except Exception as exc:
            raise HTTPException(400, f"Cannot save: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return {"path": path.relative_to(SCENARIO_DIR).as_posix()}

    # ----------------------------------------------------------- validation
    @app.post("/api/validate")
    def validate(editor: dict[str, Any]) -> dict[str, Any]:
        return validate_editor(editor, SCENARIO_DIR)

    @app.post("/api/response")
    def response(request: dict[str, Any]) -> dict[str, Any]:
        editor = request["graph"]
        try:
            graph = build_graph(editor, SCENARIO_DIR)
            return _clean(block_response(graph, request["block"], int(editor.get("seed", 0))))
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    # ----------------------------------------------------------------- jobs
    @app.post("/api/jobs")
    def start_job(request: dict[str, Any]) -> dict[str, Any]:
        return jobs.submit(request).to_dict()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        jobs.cancel(job_id)
        return {"ok": True}

    # ----------------------------------------------------------------- live
    @app.websocket("/api/live")
    async def live(ws: WebSocket) -> None:
        """Interactive mode: run one frame per step and stream probe data.

        Live mode always simulates a single frame per step (batch size 1): the scopes show
        one frame and the metrics accumulate across steps. Client messages: ``start`` {graph, rate}, ``set_param`` {path, value},
        ``reset``, ``stop``. The scope blocks of the graph select the streamed plots.
        """
        await ws.accept()
        state: dict[str, Any] = {"sim": None, "probes": [], "rate": 8.0, "pending": [], "reset": False}

        async def receiver() -> None:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "start":
                    editor = msg["graph"]
                    try:
                        graph = build_graph(editor, SCENARIO_DIR)
                        sim = Simulation(graph, batch_size=1, seed=int(editor.get("seed", 0)))
                        sim.reset()
                    except Exception as exc:
                        await ws.send_json({"type": "error", "error": str(exc)})
                        continue
                    state.update(sim=sim, probes=scope_probes(editor), rate=float(msg.get("rate", 8.0)),
                                 pending=[])
                    await ws.send_json({"type": "started"})
                elif kind == "set_param":
                    state["pending"].append((msg["path"], msg["value"]))
                elif kind == "rate":
                    state["rate"] = float(msg.get("rate", 8.0))
                elif kind == "reset":
                    state["reset"] = True
                elif kind == "stop":
                    state["sim"] = None
                    await ws.send_json({"type": "stopped"})

        def step(sim: Simulation, probes: list, pending: list, reset: bool) -> dict[str, Any]:
            errors = []
            for path, value in pending:
                try:
                    sim.graph.set_param(path, value)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
            if pending or reset:  # restart the measurements after a change
                for block in sim.graph.blocks.values():
                    if block.metrics():
                        block.on_reset()
            t0 = time.perf_counter()
            values = sim.step()
            return {"type": "frame", "batch": sim._batch_index, "metrics": sim.graph.metrics(),
                    "probes": compute_probes(values, probes), "displays": block_displays(sim.graph, values),
                    "errors": errors,
                    "step_ms": round(1000 * (time.perf_counter() - t0), 1)}

        recv_task = asyncio.create_task(receiver())
        try:
            while not recv_task.done():
                sim = state["sim"]
                if sim is None:
                    await asyncio.sleep(0.05)
                    continue
                pending, state["pending"] = state["pending"], []
                reset, state["reset"] = state["reset"], False
                t0 = time.perf_counter()
                try:
                    frame = await asyncio.to_thread(step, sim, list(state["probes"]), pending, reset)
                except Exception as exc:
                    state["sim"] = None
                    await ws.send_json({"type": "error", "error": str(exc)})
                    continue
                await ws.send_json(_clean(frame))
                await asyncio.sleep(max(0.0, 1.0 / max(state["rate"], 0.5) - (time.perf_counter() - t0)))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            recv_task.cancel()

    # ------------------------------------------------------------ static UI
    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
