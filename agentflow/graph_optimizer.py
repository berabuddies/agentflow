from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

from agentflow.loader import load_pipeline_from_data
from agentflow.specs import PipelineSpec, RunRecord, normalize_agent_name
from agentflow.store import RunStore
from agentflow.utils import ensure_dir, json_dumps


GENERATED_PIPELINE_FILENAME = "pipeline.py"
GENERATED_PIPELINE_ORIGINAL_FILENAME = "pipeline.original.py"
GENERATED_PIPELINE_EDITED_FILENAME = "pipeline.edited.py"
GRAPH_REPORT_FILENAME = "graph_report.json"
OPTIMIZER_PROMPT_FILENAME = "optimizer-prompt.txt"
OPTIMIZER_RESULT_FILENAME = "optimizer-result.json"
OPTIMIZER_VALIDATION_FILENAME = "optimizer-validation.json"
GRAPH_OPTIMIZER_MAX_ATTEMPTS = 3
CHILD_PIPELINE_LOAD_TIMEOUT_SECONDS = 5.0


def editable_pipeline_payload(pipeline: PipelineSpec) -> dict[str, Any]:
    payload = pipeline.model_dump(mode="json")
    payload.pop("optimizer", None)
    payload.pop("n_run", None)
    return payload


def render_editable_pipeline_python(pipeline: PipelineSpec) -> str:
    payload = editable_pipeline_payload(pipeline)
    payload_text = pformat(payload, sort_dicts=False, width=100)
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "# This file is generated for graph optimization rounds.\n"
        "# Edit PIPELINE directly. Reorder nodes by moving entries in PIPELINE['nodes'] and\n"
        "# updating `depends_on` lists as needed. Keep absolute execution paths unless you\n"
        "# intentionally want to retarget the graph.\n\n"
        f"PIPELINE = {payload_text}\n\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps(PIPELINE, ensure_ascii=False, indent=2))\n"
    )


def write_editable_pipeline_python(path: Path, pipeline: PipelineSpec) -> None:
    ensure_dir(path.parent)
    path.write_text(render_editable_pipeline_python(pipeline), encoding="utf-8")


def copy_run_traces(run: RunRecord, store: RunStore, traces_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    ensure_dir(traces_dir)
    for node in run.pipeline.nodes:
        source = store.artifact_path(run.id, node.id, "trace.jsonl")
        if not source.exists():
            continue
        target = traces_dir / f"{node.id}.trace.jsonl"
        shutil.copy2(source, target)
        copied[node.id] = str(target)
    return copied


def build_graph_report(
    *,
    parent_run_id: str,
    round_number: int,
    total_rounds: int,
    run: RunRecord,
    store: RunStore,
    copied_traces: dict[str, str],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    pipeline_nodes = run.pipeline.node_map
    for node_id, result in run.nodes.items():
        pipeline_node = pipeline_nodes[node_id]
        nodes.append(
            {
                "id": node_id,
                "agent": normalize_agent_name(pipeline_node.agent),
                "prompt_template": pipeline_node.prompt,
                "depends_on": list(pipeline_node.depends_on),
                "status": result.status.value,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "exit_code": result.exit_code,
                "output": result.output,
                "final_response": result.final_response,
                "success": result.success,
                "success_details": list(result.success_details),
                "attempt_count": len(result.attempts),
                "attempts": [attempt.model_dump(mode="json") for attempt in result.attempts],
                "artifacts": {
                    "trace_jsonl": copied_traces.get(node_id),
                    "stdout_log": str(store.artifact_path(run.id, node_id, "stdout.log")),
                    "stderr_log": str(store.artifact_path(run.id, node_id, "stderr.log")),
                    "output_txt": str(store.artifact_path(run.id, node_id, "output.txt")),
                    "result_json": str(store.artifact_path(run.id, node_id, "result.json")),
                    "launch_json": str(store.artifact_path(run.id, node_id, "launch.json")),
                },
            }
        )

    return {
        "parent_run_id": parent_run_id,
        "round_number": round_number,
        "total_rounds": total_rounds,
        "child_run_id": run.id,
        "pipeline": editable_pipeline_payload(run.pipeline),
        "run": {
            "id": run.id,
            "status": run.status.value,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        },
        "events": [event.model_dump(mode="json") for event in store.get_events(run.id)],
        "nodes": nodes,
    }


def render_graph_optimizer_prompt(
    *,
    optimizer: str,
    pipeline_path: Path,
    graph_report_path: Path,
    traces_dir: Path,
    round_number: int,
    total_rounds: int,
    attempt_number: int = 1,
    max_attempts: int = GRAPH_OPTIMIZER_MAX_ATTEMPTS,
    previous_failure: str | None = None,
) -> str:
    failure_section = ""
    if previous_failure:
        failure_section = (
            "\nPrevious optimizer/load failure to fix before finishing:\n"
            f"{previous_failure}\n"
        )

    return (
        f"You are optimizing an AgentFlow graph using `{optimizer}`.\n"
        f"Round: {round_number} of {total_rounds}\n"
        f"Optimizer attempt: {attempt_number} of {max_attempts}\n"
        f"Editable pipeline file: {pipeline_path}\n"
        f"Graph report JSON: {graph_report_path}\n"
        f"Copied node traces directory: {traces_dir}\n"
        f"{failure_section}\n"
        "Goal:\n"
        "- Improve the next round of graph execution using evidence from the graph report and copied traces.\n"
        "- Prefer changes that materially improve correctness, reliability, latency, or coordination quality.\n"
        "- Prefer focused, high-leverage edits over broad churn.\n"
        "\n"
        "Working materials:\n"
        "- `pipeline.py` is the only file you should edit.\n"
        "- `graph_report.json` contains the current graph, node outcomes, timings, outputs, attempts, events, and artifact paths.\n"
        "- `traces/` contains per-node trace files copied from the completed round.\n"
        "\n"
        "Allowed graph changes:\n"
        "- Reorder nodes.\n"
        "- Rewrite node prompts.\n"
        "- Add, remove, or rewire dependencies.\n"
        "- Add or remove nodes when justified.\n"
        "- Modify graph-level settings such as concurrency, fail-fast behavior, or iteration controls.\n"
        "\n"
        "Guardrails:\n"
        "- Edit `pipeline.py` in place.\n"
        "- Preserve a valid Python script that defines `PIPELINE` and prints it as JSON when executed.\n"
        "- Keep at least one node in the graph.\n"
        "- Keep node ids stable unless there is a strong reason to rename, add, or remove them.\n"
        "- Keep `working_dir` and local target `cwd` paths absolute unless you intentionally want to retarget execution.\n"
        "- If the previous attempt failed, fix that load or validation error in `pipeline.py` before making further changes.\n"
        "- Do not run the full graph yourself; the outer harness will validate and run the next round.\n"
        "\n"
        "Validation checklist before finishing:\n"
        "- `pipeline.py` is syntactically valid Python.\n"
        "- Running `pipeline.py` prints a valid pipeline JSON payload.\n"
        "- The resulting pipeline validates cleanly and contains at least one node.\n"
        "- Dependency references are valid after any reorder or rewrite.\n"
        "- The edited graph still reflects the intended workflow.\n"
        "- Review carefully for regressions before finishing.\n"
    )


def write_optimizer_result(path: Path, *, command: str, exit_code: int, stdout: str, stderr: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json_dumps(
            {
                "command": command,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }
        ),
        encoding="utf-8",
    )


def write_validation_result(path: Path, *, ok: bool, error: str | None = None) -> None:
    ensure_dir(path.parent)
    payload: dict[str, Any] = {"ok": ok}
    if error is not None:
        payload["error"] = error
    path.write_text(json_dumps(payload), encoding="utf-8")


def load_child_pipeline_from_path(path: Path) -> PipelineSpec:
    """Load optimizer-edited pipeline as a child-run shape (`optimizer=None`, `n_run=1`)."""

    path = Path(path)
    if path.suffix == ".py":
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                cwd=str(path.parent),
                timeout=CHILD_PIPELINE_LOAD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            details: list[str] = []
            if exc.stdout:
                details.append(f"stdout:\n{exc.stdout.strip()}")
            if exc.stderr:
                details.append(f"stderr:\n{exc.stderr.strip()}")
            rendered_details = "" if not details else "\n" + "\n\n".join(details)
            raise ValueError(
                f"pipeline script `{path}` timed out after {CHILD_PIPELINE_LOAD_TIMEOUT_SECONDS:.1f}s.{rendered_details}"
            ) from exc
        if result.returncode != 0:
            raise ValueError(f"pipeline script `{path}` failed:\n{result.stderr.strip()}")
        raw_text = result.stdout
    else:
        raw_text = path.read_text(encoding="utf-8")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"optimized pipeline `{path}` did not produce JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"optimized pipeline `{path}` did not produce an object payload")

    parsed["optimizer"] = None
    parsed["n_run"] = 1
    return load_pipeline_from_data(parsed, base_dir=path.parent)
