from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from agentflow.app import create_app
from agentflow.orchestrator import Orchestrator
from agentflow.specs import RunRecord
from agentflow.store import RunStore
from tests.test_orchestrator import make_orchestrator


def test_api_starts_and_returns_run_details(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    payload = {
        "pipeline": {
            "name": "api-run",
            "working_dir": str(tmp_path),
            "nodes": [
                {"id": "alpha", "agent": "codex", "prompt": "api success"},
            ],
        }
    }
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 200
    run_id = response.json()["id"]
    asyncio.run(orchestrator.wait(run_id, timeout=5))
    run_response = client.get(f"/api/runs/{run_id}")
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["status"] == "completed"
    assert body["nodes"]["alpha"]["output"] == "api success"


def test_api_returns_default_example_payload(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    response = client.get("/api/examples/default")
    assert response.status_code == 200
    payload = json.loads(response.json()["example"])
    assert payload["name"] == "airflow-like-example"
    assert payload["working_dir"] == "."
    assert response.json()["base_dir"] == os.getcwd()


def test_api_supports_validation_and_artifacts(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    validate = client.post(
        "/api/runs/validate",
        json={"pipeline_text": json.dumps({"name": "ok", "working_dir": ".", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}]})},
    )
    assert validate.status_code == 200
    assert validate.json()["pipeline"]["name"] == "ok"

    invalid = client.post(
        "/api/runs/validate",
        json={"pipeline_text": json.dumps({"name": "bad", "nodes": [{"id": "a", "agent": "codex", "prompt": "hi", "depends_on": ["b"]}]})},
    )
    assert invalid.status_code == 422

    create = client.post(
        "/api/runs",
        json={"pipeline": {"name": "artifact", "working_dir": str(tmp_path), "nodes": [{"id": "alpha", "agent": "codex", "prompt": "artifact output"}]}}
    )
    run_id = create.json()["id"]
    asyncio.run(orchestrator.wait(run_id, timeout=5))
    artifact = client.get(f"/api/runs/{run_id}/artifacts/alpha/output.txt")
    assert artifact.status_code == 200
    assert artifact.text == "artifact output"
    launch = client.get(f"/api/runs/{run_id}/artifacts/alpha/launch.json")
    assert launch.status_code == 200
    assert launch.json()["kind"] == "process"
    assert launch.json()["command"][0] == "python3"


def test_api_validate_resolves_inline_pipeline_text_relative_to_explicit_base_dir(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    workspace = tmp_path / "workspace"
    response = client.post(
        "/api/runs/validate",
        json={
            "pipeline_text": json.dumps({
                "name": "inline-json",
                "working_dir": ".",
                "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi", "target": {"kind": "local", "cwd": "task"}}],
            }),
            "base_dir": str(workspace),
        },
    )

    assert response.status_code == 200
    payload = response.json()["pipeline"]
    assert payload["working_dir"] == str(workspace.resolve())
    assert payload["nodes"][0]["target"]["cwd"] == str((workspace / "task").resolve())


def test_api_run_resolves_inline_pipeline_relative_to_explicit_base_dir(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    workspace = tmp_path / "workspace"
    (workspace / "task").mkdir(parents=True)
    response = client.post(
        "/api/runs",
        json={
            "base_dir": str(workspace),
            "pipeline": {
                "name": "inline-json",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "alpha",
                        "agent": "codex",
                        "prompt": "hi",
                        "target": {
                            "kind": "local",
                            "cwd": "task",
                        },
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    payload = body["pipeline"]
    assert payload["working_dir"] == str(workspace.resolve())
    assert payload["nodes"][0]["target"]["cwd"] == str((workspace / "task").resolve())
    asyncio.run(orchestrator.wait(body["id"], timeout=5))


def test_api_validate_rejects_pipeline_path_payload_by_default(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    pipeline_dir = tmp_path / "pipelines"
    pipeline_dir.mkdir()
    pipeline_path = pipeline_dir / "api.json"
    pipeline_path.write_text(
        json.dumps({"name": "pipeline-path", "working_dir": ".", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi", "target": {"kind": "local", "cwd": "task"}}]}),
        encoding="utf-8",
    )

    response = client.post("/api/runs/validate", json={"pipeline_path": str(pipeline_path)})

    assert response.status_code == 403
    assert response.json()["detail"] == "pipeline_path is disabled for the web API by default"


def test_api_validate_supports_pipeline_path_payload_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFLOW_API_ALLOW_PIPELINE_PATH", "1")
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    pipeline_dir = tmp_path / "pipelines"
    pipeline_dir.mkdir()
    pipeline_path = pipeline_dir / "api.json"
    pipeline_path.write_text(
        json.dumps({"name": "pipeline-path", "working_dir": ".", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi", "target": {"kind": "local", "cwd": "task"}}]}),
        encoding="utf-8",
    )

    response = client.post("/api/runs/validate", json={"pipeline_path": str(pipeline_path)})

    assert response.status_code == 200
    payload = response.json()["pipeline"]
    assert payload["working_dir"] == str(pipeline_dir.resolve())
    assert payload["nodes"][0]["target"]["cwd"] == str((pipeline_dir / "task").resolve())



def test_api_rejects_artifact_path_traversal(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("outside-runs-secret", encoding="utf-8")

    create = client.post(
        "/api/runs",
        json={"pipeline": {"name": "artifact", "working_dir": str(tmp_path), "nodes": [{"id": "alpha", "agent": "codex", "prompt": "artifact output"}]}},
    )
    run_id = create.json()["id"]
    asyncio.run(orchestrator.wait(run_id, timeout=5))

    sibling_run_file = client.get(f"/api/runs/{run_id}/artifacts/%2E%2E/run.json")
    assert sibling_run_file.status_code == 400

    outside_runs_file = client.get("/api/runs/%2E%2E/artifacts/%2E%2E/secret.txt")
    assert outside_runs_file.status_code == 400


async def test_store_create_run_rejects_invalid_run_id_atomically(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = RunRecord(
        id="../outside",
        pipeline={"name": "p", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}]},
    )

    with pytest.raises(ValueError, match="path segment"):
        await store.create_run(record)

    assert store.list_runs() == []
    assert not (tmp_path / "outside").exists()


async def test_store_rejects_artifact_write_path_traversal(tmp_path):
    store = RunStore(tmp_path / "runs")
    await store.create_run(
        RunRecord(
            id="run",
            pipeline={"name": "p", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}]},
        )
    )

    with pytest.raises(ValueError, match="path segment"):
        await store.write_artifact_text("run", "../../outside", "output.txt", "pwned")
    assert not (tmp_path / "outside" / "output.txt").exists()


async def test_store_rejects_artifact_read_path_traversal(tmp_path):
    store = RunStore(tmp_path / "runs")
    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("outside-runs-secret", encoding="utf-8")

    with pytest.raises(ValueError, match="path segment"):
        store.read_artifact_text("..", "..", "secret.txt")


def test_api_rejects_non_json_content_type(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    response = client.post(
        "/api/runs/validate",
        data=json.dumps({"pipeline": {"name": "ok", "working_dir": str(tmp_path), "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}]}}),
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "application/json content type required"


def test_api_supports_cancel_and_rerun(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    create = client.post(
        "/api/runs",
        json={"pipeline": {"name": "cancel", "working_dir": str(tmp_path), "nodes": [{"id": "slow", "agent": "codex", "prompt": "slow"}]}}
    )
    run_id = create.json()["id"]
    for _ in range(50):
        run = orchestrator.store.get_run(run_id)
        if run.status.value == "running":
            break
        import time
        time.sleep(0.05)
    cancel = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel.status_code == 200
    completed = asyncio.run(orchestrator.wait(run_id, timeout=5))
    assert completed.status.value == "cancelled"

    rerun = client.post(f"/api/runs/{run_id}/rerun")
    assert rerun.status_code == 200
    rerun_id = rerun.json()["id"]
    assert rerun_id != run_id


def test_api_stream_replays_completed_run_and_closes(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    app = create_app(store=orchestrator.store, orchestrator=orchestrator)
    client = TestClient(app)

    create = client.post(
        "/api/runs",
        json={
            "pipeline": {
                "name": "stream-replay",
                "working_dir": str(tmp_path),
                "nodes": [{"id": "alpha", "agent": "codex", "prompt": "stream ok"}],
            }
        },
    )
    run_id = create.json()["id"]
    asyncio.run(orchestrator.wait(run_id, timeout=5))

    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    events = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    assert events
    assert events[-1]["type"] == "run_completed"
    assert any(event["type"] == "node_completed" for event in events)
