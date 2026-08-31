from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentflow.graph_optimizer import (
    CHILD_PIPELINE_LOAD_TIMEOUT_SECONDS,
    GRAPH_OPTIMIZER_MAX_ATTEMPTS,
    GENERATED_PIPELINE_EDITED_FILENAME,
    GENERATED_PIPELINE_ORIGINAL_FILENAME,
    OPTIMIZER_VALIDATION_FILENAME,
    editable_pipeline_payload,
    load_child_pipeline_from_path,
    render_graph_optimizer_prompt,
    write_editable_pipeline_python,
)
from agentflow.loader import load_pipeline_from_path
from agentflow.specs import PipelineSpec, RunStatus
from agentflow.tuned_agents import CommandExecution
from tests.test_orchestrator import make_orchestrator


def test_editable_pipeline_python_round_trips(tmp_path):
    pipeline = PipelineSpec.model_validate(
        {
            "name": "roundtrip",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {"id": "plan", "agent": "codex", "prompt": "plan"},
                {"id": "review", "agent": "claude", "prompt": "review", "depends_on": ["plan"]},
            ],
        }
    )
    pipeline_path = tmp_path / "pipeline.py"

    write_editable_pipeline_python(pipeline_path, pipeline)
    loaded = load_pipeline_from_path(pipeline_path)

    payload = loaded.model_dump(mode="json")
    payload.pop("optimizer", None)
    payload.pop("n_run", None)
    assert payload == editable_pipeline_payload(pipeline)


def test_pipeline_spec_requires_optimizer_when_n_run_exceeds_one(tmp_path):
    with pytest.raises(ValueError, match="`optimizer` is required"):
        PipelineSpec.model_validate(
            {
                "name": "bad",
                "working_dir": str(tmp_path),
                "n_run": 2,
                "nodes": [{"id": "plan", "agent": "codex", "prompt": "hi"}],
            }
        )


def test_pipeline_spec_requires_at_least_one_node(tmp_path):
    with pytest.raises(ValidationError, match="pipeline must contain at least one node"):
        PipelineSpec.model_validate(
            {
                "name": "empty",
                "working_dir": str(tmp_path),
                "nodes": [],
            }
        )


def test_graph_optimizer_prompt_includes_goal_guardrails_and_validation(tmp_path):
    prompt = render_graph_optimizer_prompt(
        optimizer="codex",
        pipeline_path=tmp_path / "pipeline.py",
        graph_report_path=tmp_path / "graph_report.json",
        traces_dir=tmp_path / "traces",
        round_number=2,
        total_rounds=5,
        attempt_number=1,
        max_attempts=GRAPH_OPTIMIZER_MAX_ATTEMPTS,
        previous_failure=None,
    )

    assert "Goal:" in prompt
    assert "Working materials:" in prompt
    assert "Allowed graph changes:" in prompt
    assert "Guardrails:" in prompt
    assert "Validation checklist before finishing:" in prompt
    assert "Keep at least one node in the graph." in prompt
    assert "The resulting pipeline validates cleanly and contains at least one node." in prompt


def test_load_child_pipeline_from_path_times_out_with_debug_output(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.py"

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr("agentflow.graph_optimizer.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="timed out") as exc_info:
        load_child_pipeline_from_path(pipeline_path)

    message = str(exc_info.value)
    assert f"{CHILD_PIPELINE_LOAD_TIMEOUT_SECONDS:.1f}s" in message
    assert "partial stdout" in message
    assert "partial stderr" in message


def test_orchestrator_runs_graph_optimization_rounds(tmp_path, monkeypatch):
    orchestrator = make_orchestrator(tmp_path)

    def fake_optimizer(_optimizer, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        pipeline_path = repo_dir / "pipeline.py"
        text = pipeline_path.read_text(encoding="utf-8")
        pipeline_path.write_text(text.replace("round one", "round two"), encoding="utf-8")
        return CommandExecution(command="optimizer", exit_code=0, stdout="updated pipeline", stderr="")

    monkeypatch.setattr("agentflow.orchestrator._run_optimizer", fake_optimizer)

    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-opt",
            "working_dir": str(tmp_path),
            "optimizer": "codex",
            "n_run": 2,
            "nodes": [{"id": "plan", "agent": "codex", "prompt": "round one"}],
        }
    )

    run = asyncio.run(orchestrator.submit(pipeline))
    completed = asyncio.run(orchestrator.wait(run.id, timeout=5))

    assert completed.status == RunStatus.COMPLETED
    assert completed.nodes["plan"].output == "round two"
    assert completed.optimization_session is not None
    child_run_ids = completed.optimization_session["child_run_ids"]
    assert len(child_run_ids) == 2
    assert orchestrator.store.get_run(child_run_ids[0]).nodes["plan"].output == "round one"
    assert orchestrator.store.get_run(child_run_ids[1]).nodes["plan"].output == "round two"

    round_one_dir = orchestrator.store.run_dir(run.id) / "optimization" / "round-001"
    assert (round_one_dir / GENERATED_PIPELINE_ORIGINAL_FILENAME).exists()
    assert (round_one_dir / GENERATED_PIPELINE_EDITED_FILENAME).exists()


def test_orchestrator_retries_invalid_optimized_pipeline_with_error_context(tmp_path, monkeypatch):
    orchestrator = make_orchestrator(tmp_path)
    prompts: list[str] = []
    attempt_state = {"count": 0}

    def fake_optimizer(_optimizer, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        prompts.append(prompt)
        attempt_state["count"] += 1
        pipeline_path = repo_dir / "pipeline.py"
        if attempt_state["count"] == 1:
            pipeline_path.write_text("from __future__ import annotations\n\nthis is not valid python\n", encoding="utf-8")
        else:
            pipeline_path.write_text(
                (
                    "from __future__ import annotations\n\n"
                    "import json\n\n"
                    "PIPELINE = {\n"
                    f"    'name': 'graph-opt-retry',\n"
                    f"    'working_dir': {str(tmp_path)!r},\n"
                    "    'nodes': [\n"
                    "        {'id': 'plan', 'agent': 'codex', 'prompt': 'round two'},\n"
                    "    ],\n"
                    "}\n\n"
                    "if __name__ == '__main__':\n"
                    "    print(json.dumps(PIPELINE, ensure_ascii=False, indent=2))\n"
                ),
                encoding="utf-8",
            )
        return CommandExecution(command="optimizer", exit_code=0, stdout="updated pipeline", stderr="")

    monkeypatch.setattr("agentflow.orchestrator._run_optimizer", fake_optimizer)

    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-opt-retry",
            "working_dir": str(tmp_path),
            "optimizer": "codex",
            "n_run": 2,
            "nodes": [{"id": "plan", "agent": "codex", "prompt": "round one"}],
        }
    )

    run = asyncio.run(orchestrator.submit(pipeline))
    completed = asyncio.run(orchestrator.wait(run.id, timeout=5))

    assert completed.status == RunStatus.COMPLETED
    assert completed.nodes["plan"].output == "round two"
    assert attempt_state["count"] == 2
    assert "Previous optimizer/load failure to fix before finishing" in prompts[1]
    assert "optimized pipeline failed to load" in prompts[1]


def test_orchestrator_fails_when_optimized_pipeline_is_invalid(tmp_path, monkeypatch):
    orchestrator = make_orchestrator(tmp_path)
    attempt_state = {"count": 0}

    def fake_optimizer(_optimizer, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        attempt_state["count"] += 1
        pipeline_path = repo_dir / "pipeline.py"
        pipeline_path.write_text("from __future__ import annotations\n\nthis is not valid python\n", encoding="utf-8")
        return CommandExecution(command="optimizer", exit_code=0, stdout="broken pipeline", stderr="")

    monkeypatch.setattr("agentflow.orchestrator._run_optimizer", fake_optimizer)

    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-opt-invalid",
            "working_dir": str(tmp_path),
            "optimizer": "codex",
            "n_run": 2,
            "nodes": [{"id": "plan", "agent": "codex", "prompt": "round one"}],
        }
    )

    run = asyncio.run(orchestrator.submit(pipeline))
    completed = asyncio.run(orchestrator.wait(run.id, timeout=5))

    assert completed.status == RunStatus.FAILED
    assert completed.optimization_session is not None
    assert len(completed.optimization_session["child_run_ids"]) == 1
    assert attempt_state["count"] == GRAPH_OPTIMIZER_MAX_ATTEMPTS

    validation_payload = json.loads(
        (orchestrator.store.run_dir(run.id) / "optimization" / "round-001" / OPTIMIZER_VALIDATION_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert validation_payload["ok"] is False
    assert "failed to load" in validation_payload["error"]


def test_orchestrator_normalizes_optimizer_edits_to_iteration_controls(tmp_path, monkeypatch):
    orchestrator = make_orchestrator(tmp_path)

    def fake_optimizer(_optimizer, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        pipeline_path = repo_dir / "pipeline.py"
        pipeline_path.write_text(
            (
                "from __future__ import annotations\n\n"
                "import json\n\n"
                "PIPELINE = {\n"
                "    'name': 'graph-opt-nrun-edit',\n"
                f"    'working_dir': {str(tmp_path)!r},\n"
                "    'n_run': 3,\n"
                "    'nodes': [\n"
                "        {'id': 'plan', 'agent': 'codex', 'prompt': 'round two'},\n"
                "    ],\n"
                "}\n\n"
                "if __name__ == '__main__':\n"
                "    print(json.dumps(PIPELINE, ensure_ascii=False, indent=2))\n"
            ),
            encoding="utf-8",
        )
        return CommandExecution(command="optimizer", exit_code=0, stdout="updated pipeline", stderr="")

    monkeypatch.setattr("agentflow.orchestrator._run_optimizer", fake_optimizer)

    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-opt-nrun-edit",
            "working_dir": str(tmp_path),
            "optimizer": "codex",
            "n_run": 2,
            "nodes": [{"id": "plan", "agent": "codex", "prompt": "round one"}],
        }
    )

    run = asyncio.run(orchestrator.submit(pipeline))
    completed = asyncio.run(orchestrator.wait(run.id, timeout=5))

    assert completed.status == RunStatus.COMPLETED
    assert completed.nodes["plan"].output == "round two"
    assert completed.optimization_session is not None
    assert completed.optimization_session["current_round"] == pipeline.n_run
