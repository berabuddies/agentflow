from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from agentflow.defaults import bundled_template_path
from agentflow.loader import load_pipeline_from_data, load_pipeline_from_path, load_pipeline_from_text
from agentflow.orchestrator import Orchestrator
from agentflow.specs import PipelineSpec
from agentflow.store import RunStore


_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def _api_pipeline_path_enabled() -> bool:
    return os.getenv("AGENTFLOW_API_ALLOW_PIPELINE_PATH", "").strip().lower() in {"1", "true", "yes", "on"}


def _require_json_request(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise HTTPException(status_code=415, detail="application/json content type required")


@lru_cache(maxsize=1)
def _load_default_web_example() -> str:
    example_path = bundled_template_path("pipeline")
    project_root = os.path.dirname(os.path.dirname(__file__))
    pythonpath = project_root
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath = f"{project_root}{os.pathsep}{existing_pythonpath}"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    if result.returncode != 0:
        raise RuntimeError(f"default pipeline example failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def _parse_pipeline_payload(payload: dict[str, Any], *, allow_pipeline_path: bool = True) -> PipelineSpec:
    try:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")

        pipeline_path = payload.get("pipeline_path")
        if isinstance(pipeline_path, str) and pipeline_path.strip():
            if not allow_pipeline_path:
                raise HTTPException(status_code=403, detail="pipeline_path is disabled for the web API by default")
            return load_pipeline_from_path(pipeline_path)

        base_dir = payload.get("base_dir")
        if base_dir is not None and not isinstance(base_dir, (str, os.PathLike)):
            raise ValueError("base_dir must be a string path")
        if "pipeline_text" in payload:
            pipeline_text = payload["pipeline_text"]
            if not isinstance(pipeline_text, str):
                raise ValueError("pipeline_text must be a string")
            return load_pipeline_from_text(pipeline_text, base_dir=base_dir)

        pipeline_data = payload["pipeline"] if "pipeline" in payload else dict(payload)
        if isinstance(pipeline_data, dict):
            pipeline_data = dict(pipeline_data)
            pipeline_data.pop("base_dir", None)
            pipeline_data.pop("pipeline_path", None)
        return load_pipeline_from_data(pipeline_data, base_dir=base_dir)
    except (ValueError, ValidationError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(*, store: RunStore | None = None, orchestrator: Orchestrator | None = None) -> FastAPI:
    store = store or RunStore(os.getenv("AGENTFLOW_RUNS_DIR", ".agentflow/runs"))
    orchestrator = orchestrator or Orchestrator(
        store=store,
        max_concurrent_runs=int(os.getenv("AGENTFLOW_MAX_CONCURRENT_RUNS", "2")),
    )
    app = FastAPI(title="AgentFlow", version="0.1.0")
    app.state.store = store
    app.state.orchestrator = orchestrator

    base_dir = os.path.join(os.path.dirname(__file__), "web")
    templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
    app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            name="index.html",
            request=request,
            context={"example": _load_default_web_example(), "base_dir": os.getcwd()},
        )

    @app.get("/api/examples/default")
    async def default_example() -> JSONResponse:
        return JSONResponse({"example": _load_default_web_example(), "base_dir": os.getcwd()})

    @app.post("/api/runs/validate")
    async def validate_run(request: Request) -> JSONResponse:
        _require_json_request(request)
        payload = await request.json()
        pipeline = _parse_pipeline_payload(payload, allow_pipeline_path=_api_pipeline_path_enabled())
        return JSONResponse({"ok": True, "pipeline": pipeline.model_dump(mode="json")})

    @app.post("/api/runs")
    async def create_run(request: Request) -> JSONResponse:
        _require_json_request(request)
        payload = await request.json()
        pipeline = _parse_pipeline_payload(payload, allow_pipeline_path=_api_pipeline_path_enabled())
        run = await app.state.orchestrator.submit(pipeline)
        return JSONResponse(run.model_dump(mode="json"))

    @app.get("/api/runs")
    async def list_runs() -> JSONResponse:
        return JSONResponse([run.model_dump(mode="json") for run in app.state.store.list_runs()])

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        try:
            run = app.state.store.get_run(run_id)
        except KeyError as exc:  # pragma: no cover - exercised by API callers only
            raise HTTPException(status_code=404, detail="run not found") from exc
        return JSONResponse(run.model_dump(mode="json"))

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> JSONResponse:
        try:
            run = await app.state.orchestrator.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return JSONResponse(run.model_dump(mode="json"))

    @app.post("/api/runs/{run_id}/rerun")
    async def rerun(run_id: str) -> JSONResponse:
        try:
            run = await app.state.orchestrator.rerun(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return JSONResponse(run.model_dump(mode="json"))

    @app.get("/api/runs/{run_id}/events")
    async def get_events(run_id: str) -> JSONResponse:
        if run_id not in {run.id for run in app.state.store.list_runs()}:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse([event.model_dump(mode="json") for event in app.state.store.get_events(run_id)])

    @app.get("/api/runs/{run_id}/artifacts/{node_id}/{name}")
    async def get_artifact(run_id: str, node_id: str, name: str) -> PlainTextResponse:
        try:
            content = app.state.store.read_artifact_text(run_id, node_id, name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid artifact path") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return PlainTextResponse(content)

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        if run_id not in {run.id for run in app.state.store.list_runs()}:
            raise HTTPException(status_code=404, detail="run not found")
        queue = await app.state.store.subscribe(run_id)

        async def event_stream():
            try:
                cached_events = app.state.store.get_events(run_id)
                for cached in cached_events:
                    yield f"data: {cached.model_dump_json()}\n\n"
                if cached_events and cached_events[-1].type == "run_completed":
                    return
                while True:
                    event = await asyncio.to_thread(queue.get)
                    yield f"data: {event.model_dump_json()}\n\n"
                    run = app.state.store.get_run(run_id)
                    if run.status.value in _TERMINAL_RUN_STATUSES and event.type == "run_completed":
                        break
            finally:
                await app.state.store.unsubscribe(run_id, queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        runs = app.state.store.list_runs()
        return JSONResponse(
            {
                "ok": True,
                "runs": {
                    "total": len(runs),
                    "queued": sum(run.status.value == "queued" for run in runs),
                    "running": sum(run.status.value in {"running", "cancelling"} for run in runs),
                },
            }
        )

    return app
