from __future__ import annotations

import asyncio
import os
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from pathlib import Path

import typer
from jinja2 import TemplateError
from pydantic import ValidationError
from agentflow.defaults import (
    bundled_templates,
    bundled_template_names,
    default_smoke_pipeline_path,
    render_bundled_template,
)
from agentflow.doctor import (
    DoctorCheck,
    DoctorReport,
    LocalToolchainReport,
    ShellBridgeRecommendation,
    build_bash_login_shell_bridge_recommendation,
    build_local_kimi_toolchain_report,
    build_local_kimi_bootstrap_doctor_report,
    build_pipeline_local_claude_readiness_checks,
    build_pipeline_local_claude_readiness_info_checks,
    build_pipeline_local_codex_auth_checks,
    build_pipeline_local_codex_auth_info_checks,
    build_pipeline_local_codex_readiness_checks,
    build_pipeline_local_codex_readiness_info_checks,
    build_pipeline_local_kimi_readiness_checks,
    build_pipeline_local_kimi_readiness_info_checks,
    build_local_smoke_doctor_report,
)
from agentflow.env import merge_env_layers
from agentflow.local_shell import (
    kimi_shell_init_requires_bash_warning,
    kimi_shell_init_requires_interactive_bash_warning,
    probe_target_bash_startup_env_var,
    shell_command_overrides_env_var,
    shell_command_prefix_env_value,
    shell_command_uses_kimi_helper,
    shell_init_exported_env_var_value,
    shell_init_uses_kimi_helper,
    shell_template_exported_env_var_value_before_command,
    target_bash_home,
    target_bash_login_startup_warning,
    target_uses_interactive_bash,
    target_uses_login_bash,
)
from agentflow.prepared import resolve_local_workdir
from agentflow.specs import AgentKind, LocalTarget, PipelineSpec, normalize_agent_name, provider_uses_kimi_anthropic_auth, resolve_provider
from agentflow.tuned_agents import list_tuned_agent_records, resolve_tuned_agent_version, run_evolution_from_payload

app = typer.Typer(add_completion=False)


class StructuredOutputFormat(StrEnum):
    AUTO = "auto"
    JSON = "json"
    JSON_SUMMARY = "json-summary"
    SUMMARY = "summary"


class InspectionOutputFormat(StrEnum):
    AUTO = "auto"
    JSON = "json"
    JSON_SUMMARY = "json-summary"
    SUMMARY = "summary"


class RunOutputFormat(StrEnum):
    AUTO = "auto"
    JSON = "json"
    JSON_SUMMARY = "json-summary"
    SUMMARY = "summary"


class SmokePreflightMode(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


class InferenceEngine(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class InferenceMode(StrEnum):
    SERVICE = "service"
    BATCH = "batch"


_KIMI_SHELL_PREFLIGHT_AGENTS = {"codex", "claude", "kimi"}
_PIPELINE_LAUNCH_INSPECTION_ERRORS: dict[int, str] = {}


@dataclass(frozen=True)
class _LocalBootstrapCredentialProbe:
    found: bool
    timeout_seconds: float | None = None


def _build_runtime(runs_dir: str, max_concurrent_runs: int) -> tuple[object, object]:
    from agentflow.orchestrator import Orchestrator
    from agentflow.store import RunStore

    store = RunStore(runs_dir)
    orchestrator = Orchestrator(store=store, max_concurrent_runs=max_concurrent_runs)
    return store, orchestrator


def _build_store(runs_dir: str) -> object:
    from agentflow.store import RunStore

    return RunStore(runs_dir)


def _render_tuned_agents_summary(records: list[object]) -> str:
    if not records:
        return "No tuned agents found."
    lines: list[str] = []
    for record in records:
        lines.append(
            f"{getattr(record, 'name', '-')}"
            f" [{_status_value(getattr(record, 'base_agent', '-'))}] "
            f"latest={getattr(record, 'latest_version', '-') or '-'} "
            f"versions={len(getattr(record, 'versions', []) or [])}"
        )
    return "\n".join(lines)


def _render_tuned_agent_detail(record: object | None) -> str:
    if record is None:
        return "Tuned agent not found."
    lines = [
        f"Name: {getattr(record, 'name', '-')}",
        f"Base agent: {_status_value(getattr(record, 'base_agent', '-'))}",
        f"Latest version: {getattr(record, 'latest_version', '-') or '-'}",
        f"Versions: {len(getattr(record, 'versions', []) or [])}",
    ]
    versions = getattr(record, "versions", []) or []
    for version in versions:
        lines.append(
            f" - {getattr(version, 'id', '-')} status={getattr(version, 'status', '-')} "
            f"repo={getattr(version, 'repo_path', '-')}"
        )
    return "\n".join(lines)


def _render_evolution_summary(result: dict[str, object]) -> str:
    return "\n".join(
        [
            f"Agent: {result.get('agent_name', '-')}",
            f"Version: {result.get('version', '-')}",
            f"Base agent: {result.get('base_agent', '-')}",
            f"Executable: {result.get('executable', '-')}",
            f"Repo path: {result.get('repo_path', '-')}",
        ]
    )


def _render_inference_summary(result: object) -> str:
    records = getattr(result, "records", None) or []
    lines = [
        f"Inference job: {getattr(result, 'name', '-')}",
        f"Engine: {getattr(result, 'engine', '-')}",
        f"GPU: {getattr(getattr(result, 'gpu', None), 'raw', '-')}",
        f"Spot: {getattr(result, 'use_spot', '-')}",
        f"Job IDs: {', '.join(str(job_id) for job_id in getattr(result, 'job_ids', []) or []) or '-'}",
    ]
    if getattr(result, "pool", None):
        lines.append(f"Pool: {getattr(result, 'pool')}")
    if getattr(result, "detached", False):
        lines.append("Status: submitted")
    elif records:
        statuses = sorted({
            str(record.get("status") or record.get("job_status") or "-")
            for record in records
            if isinstance(record, dict)
        })
        lines.append(f"Status: {', '.join(statuses) if statuses else 'finished'}")
    return "\n".join(lines)


def _render_inference_service_summary(result: object) -> str:
    return "\n".join(
        [
            f"Inference service: {getattr(result, 'name', '-')}",
            f"Cluster: {getattr(result, 'cluster_name', '-')}",
            f"Engine: {getattr(result, 'engine', '-')}",
            f"GPU: {getattr(getattr(result, 'gpu', None), 'raw', '-')}",
            f"Spot: {getattr(result, 'use_spot', '-')}",
            f"Base URL: {getattr(result, 'base_url', '-')}",
            f"API key: {getattr(result, 'api_key', '-')}",
        ]
    )


def _create_web_app(store: object, orchestrator: object) -> object:
    from agentflow.app import create_app

    return create_app(store=store, orchestrator=orchestrator)


def _serve_web_app(web_app: object, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(web_app, host=host, port=port)


def _load_pipeline(path: str) -> object:
    from agentflow.loader import load_pipeline_from_path

    try:
        return load_pipeline_from_path(path)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Failed to load pipeline `{path}`:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc


def _status_value(status: object) -> str:
    return getattr(status, "value", str(status))


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_duration(started_at: str | None, finished_at: str | None) -> str | None:
    started = _parse_iso8601(started_at)
    finished = _parse_iso8601(finished_at)
    if started is None or finished is None:
        return None
    duration_seconds = max((finished - started).total_seconds(), 0.0)
    if duration_seconds < 10:
        return f"{duration_seconds:.1f}s"
    if duration_seconds < 60:
        return f"{duration_seconds:.0f}s"
    minutes, seconds = divmod(int(duration_seconds), 60)
    return f"{minutes}m {seconds}s"


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    started = _parse_iso8601(started_at)
    finished = _parse_iso8601(finished_at)
    if started is None or finished is None:
        return None
    return max((finished - started).total_seconds(), 0.0)


def _preview_text(text: str | None, *, limit: int = 100) -> str | None:
    if text is None:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _node_attempt_count(node: object) -> int:
    current_attempt = getattr(node, "current_attempt", 0) or 0
    attempts = getattr(node, "attempts", []) or []
    return current_attempt or len(attempts)


def _provider_name(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else None
    name = getattr(value, "name", None)
    if name:
        return str(name)
    if hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
        if isinstance(data, dict):
            name = data.get("name")
            if name:
                return str(name)
    return None


def _pipeline_node_map(record: object) -> dict[str, object]:
    pipeline_nodes = getattr(getattr(record, "pipeline", None), "nodes", None) or []
    return {
        node_id: node
        for node in pipeline_nodes
        if (node_id := getattr(node, "id", None))
    }


def _node_identity(node_id: str, pipeline_node: object | None) -> str:
    if pipeline_node is None:
        return node_id

    parts: list[str] = []
    agent = getattr(pipeline_node, "agent", None)
    if agent is not None:
        parts.append(_status_value(agent))

    model = getattr(pipeline_node, "model", None)
    if model:
        parts.append(f"model={model}")

    provider = _provider_name(getattr(pipeline_node, "provider", None))
    if provider:
        parts.append(f"provider={provider}")

    if not parts:
        return node_id
    return f"{node_id} [{', '.join(parts)}]"


def _node_text_candidates(node: object) -> list[str]:
    candidates: list[str] = []
    for value in (getattr(node, "final_response", None), getattr(node, "output", None)):
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    for stream_name in ("stderr_lines", "stdout_lines"):
        for line in getattr(node, stream_name, []) or []:
            if isinstance(line, str) and line.strip():
                candidates.append(line)
    return candidates


def _provider_error_subject(pipeline_node: object | None) -> str:
    agent_value = getattr(pipeline_node, "agent", None)
    agent_name = _status_value(agent_value).strip().lower() if agent_value is not None else ""
    provider_name = (_provider_name(getattr(pipeline_node, "provider", None)) or "").strip().lower()

    if agent_name == "claude" and provider_name == "kimi":
        return "Claude-on-Kimi"
    if agent_name == "codex":
        return "Codex"
    if agent_name == "claude":
        return "Claude"
    if agent_name == "kimi":
        return "Kimi"
    return "The agent"


def _provider_error_diagnosis(node: object, pipeline_node: object | None) -> str | None:
    combined = "\n".join(_node_text_candidates(node))
    if "API Error:" not in combined:
        return None

    lowered = combined.lower()
    subject = _provider_error_subject(pipeline_node)
    if any(marker in lowered for marker in ("api error: 402", "membership", "benefits", "billing", "credits", "quota")):
        return (
            f"{subject} reached the provider, but the request was rejected with a "
            "membership/billing-style API error. The local shell bootstrap is likely working; "
            "check the upstream provider account state."
        )
    return (
        f"{subject} reached the provider, but the request was rejected upstream. "
        "The local shell bootstrap is likely working; inspect the raw API error above."
    )


def _node_preview(node: object) -> str | None:
    for candidate in (getattr(node, "final_response", None), getattr(node, "output", None)):
        preview = _preview_text(candidate)
        if preview is not None:
            return preview
    stderr_lines = getattr(node, "stderr_lines", []) or []
    if stderr_lines:
        return _preview_text(stderr_lines[-1])
    return None


def _build_run_summary(record: object, run_dir: Path | str | None = None) -> dict[str, object]:
    summary: dict[str, object] = {
        "id": record.id,
        "status": _status_value(record.status),
        "nodes": [],
    }
    pipeline_name = getattr(getattr(record, "pipeline", None), "name", None)
    if pipeline_name:
        summary["pipeline"] = {"name": pipeline_name}
    started_at = getattr(record, "started_at", None)
    if started_at:
        summary["started_at"] = started_at
    finished_at = getattr(record, "finished_at", None)
    if finished_at:
        summary["finished_at"] = finished_at
    duration = _format_duration(started_at, finished_at)
    if duration is not None:
        summary["duration"] = duration
    duration_seconds = _duration_seconds(started_at, finished_at)
    if duration_seconds is not None:
        summary["duration_seconds"] = duration_seconds
    if run_dir is not None:
        summary["run_dir"] = str(run_dir)

    nodes: list[dict[str, object]] = []
    pipeline_nodes = _pipeline_node_map(record)
    for node_id, node in (getattr(record, "nodes", {}) or {}).items():
        pipeline_node = pipeline_nodes.get(node_id)
        node_summary: dict[str, object] = {
            "id": node_id,
            "status": _status_value(getattr(node, "status", "unknown")),
        }
        if pipeline_node is not None:
            agent = getattr(pipeline_node, "agent", None)
            if agent is not None:
                node_summary["agent"] = _status_value(agent)
            model = getattr(pipeline_node, "model", None)
            if model:
                node_summary["model"] = model
            provider = _provider_name(getattr(pipeline_node, "provider", None))
            if provider:
                node_summary["provider"] = provider
        attempts = _node_attempt_count(node)
        if attempts:
            node_summary["attempts"] = attempts
        exit_code = getattr(node, "exit_code", None)
        if exit_code is not None:
            node_summary["exit_code"] = exit_code
        preview = _node_preview(node)
        if preview is not None:
            node_summary["preview"] = preview
        diagnosis = _provider_error_diagnosis(node, pipeline_node)
        if diagnosis is not None:
            node_summary["diagnosis"] = diagnosis
        nodes.append(node_summary)

    summary["nodes"] = nodes
    return summary


def _render_run_summary(record: object, run_dir: Path | str | None = None) -> str:
    summary = _build_run_summary(record, run_dir=run_dir)
    lines = [f"Run {summary['id']}: {summary['status']}"]
    pipeline = summary.get("pipeline")
    if isinstance(pipeline, dict) and pipeline.get("name"):
        lines.append(f"Pipeline: {pipeline['name']}")
    duration = summary.get("duration")
    if duration is not None:
        lines.append(f"Duration: {duration}")
    run_dir_value = summary.get("run_dir")
    if run_dir_value is not None:
        lines.append(f"Run dir: {run_dir_value}")
    nodes = summary.get("nodes")
    if isinstance(nodes, list) and nodes:
        lines.append("Nodes:")
        for node in nodes:
            node_id = str(node["id"])
            parts: list[str] = []
            agent = node.get("agent")
            if agent is not None:
                parts.append(str(agent))
            model = node.get("model")
            if model:
                parts.append(f"model={model}")
            provider = node.get("provider")
            if provider:
                parts.append(f"provider={provider}")
            identity = node_id if not parts else f"{node_id} [{', '.join(parts)}]"
            rendered = f"{identity}: {node['status']}"
            metadata: list[str] = []
            attempts = node.get("attempts")
            if attempts:
                metadata.append(f"attempt {attempts}")
            exit_code = node.get("exit_code")
            if exit_code is not None:
                metadata.append(f"exit {exit_code}")
            if metadata:
                rendered += f" ({', '.join(metadata)})"
            preview = node.get("preview")
            if preview is not None:
                rendered += f" - {preview}"
            lines.append(f"- {rendered}")
            diagnosis = node.get("diagnosis")
            if diagnosis:
                lines.append(f"  Diagnosis: {diagnosis}")
    return "\n".join(lines)


def _resolve_run_output(output: RunOutputFormat, *, err: bool = False) -> RunOutputFormat:
    if output != RunOutputFormat.AUTO:
        return output
    if _stream_supports_tty_summary(err=err):
        return RunOutputFormat.SUMMARY
    return RunOutputFormat.JSON


def _echo_run_result(record: object, *, output: RunOutputFormat, run_dir: Path | str | None = None) -> None:
    resolved_output = _resolve_run_output(output)
    if resolved_output == RunOutputFormat.SUMMARY:
        typer.echo(_render_run_summary(record, run_dir=run_dir))
        return
    if resolved_output == RunOutputFormat.JSON_SUMMARY:
        typer.echo(json.dumps(_build_run_summary(record, run_dir=run_dir), indent=2))
        return
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


def _run_dir_for_record(store: object | None, run_id: str) -> Path | str | None:
    if store is None:
        return None
    run_dir = getattr(store, "run_dir", None)
    if not callable(run_dir):
        return None
    try:
        return run_dir(run_id)
    except (OSError, TypeError, ValueError):
        return None


def _build_runs_summary(records: list[object], *, store: object | None = None) -> list[dict[str, object]]:
    return [
        _build_run_summary(record, run_dir=_run_dir_for_record(store, getattr(record, "id", "")))
        for record in records
    ]


def _render_runs_summary(records: list[object], *, store: object | None = None, total: int | None = None) -> str:
    summaries = _build_runs_summary(records, store=store)
    if not summaries:
        return "No runs found."

    visible_count = len(summaries)
    total_count = visible_count if total is None else total
    header = f"Runs: {visible_count}" if total_count == visible_count else f"Runs: {visible_count} of {total_count}"
    lines = [header]
    for summary in summaries:
        rendered = f"- {summary['id']}: {summary['status']}"
        pipeline = summary.get("pipeline")
        if isinstance(pipeline, dict) and pipeline.get("name"):
            rendered += f" - {pipeline['name']}"
        duration = summary.get("duration")
        if duration is not None:
            rendered += f" ({duration})"
        lines.append(rendered)
    return "\n".join(lines)


def _echo_runs_result(records: list[object], *, store: object | None, output: RunOutputFormat, total: int | None = None) -> None:
    resolved_output = _resolve_run_output(output)
    if resolved_output == RunOutputFormat.SUMMARY:
        typer.echo(_render_runs_summary(records, store=store, total=total))
        return
    if resolved_output == RunOutputFormat.JSON_SUMMARY:
        typer.echo(json.dumps(_build_runs_summary(records, store=store), indent=2))
        return

    payload: list[object] = []
    for record in records:
        model_dump = getattr(record, "model_dump", None)
        if callable(model_dump):
            payload.append(model_dump(mode="json"))
            continue
        payload.append(_build_run_summary(record, run_dir=_run_dir_for_record(store, getattr(record, "id", ""))))
    typer.echo(json.dumps(payload, indent=2))


def _get_run_or_exit(store: object, run_id: str, *, runs_dir: str) -> object:
    try:
        return store.get_run(run_id)
    except KeyError as exc:
        typer.echo(f"Run `{run_id}` not found in `{runs_dir}`.", err=True)
        raise typer.Exit(code=1) from exc


def _run_pipeline(pipeline: object, runs_dir: str, max_concurrent_runs: int, output: RunOutputFormat) -> None:
    store, orchestrator = _build_runtime(runs_dir, max_concurrent_runs)

    async def _run() -> None:
        run_record = await orchestrator.submit(pipeline)
        completed = await orchestrator.wait(run_record.id, timeout=None)
        run_dir = store.run_dir(run_record.id) if hasattr(store, "run_dir") else None
        _echo_run_result(completed, output=output, run_dir=run_dir)
        raise typer.Exit(code=0 if _status_value(completed.status) == "completed" else 1)

    asyncio.run(_run())


def _run_pipeline_path(path: str, runs_dir: str, max_concurrent_runs: int, output: RunOutputFormat) -> None:
    _run_pipeline(_load_pipeline(path), runs_dir, max_concurrent_runs, output)


def _doctor_report():
    return build_local_smoke_doctor_report()


def _preflight_base_report(path: str, pipeline: object) -> object:
    if _path_matches_bundled_smoke(path):
        return _doctor_report()
    if _pipeline_uses_kimi_smoke_preflight(pipeline):
        return build_local_kimi_bootstrap_doctor_report()
    return _empty_doctor_report()


def _empty_doctor_report() -> DoctorReport:
    return DoctorReport(status="ok", checks=[])


def _path_matches_bundled_smoke(path: str) -> bool:
    return Path(path).expanduser().resolve() == Path(default_smoke_pipeline_path()).expanduser().resolve()


def _extend_doctor_report(report: object, extra_checks: list[DoctorCheck]) -> object:
    if not extra_checks:
        return report

    current_checks = list(getattr(report, "checks", []) or [])
    current_status = _status_value(getattr(report, "status", "ok"))
    next_status = _merge_doctor_status(current_status, extra_checks)
    return replace(report, status=next_status, checks=[*current_checks, *extra_checks])


def _pipeline_launch_inspection_nodes(pipeline: object) -> list[dict[str, object]]:
    from agentflow.inspection import build_launch_inspection

    try:
        report = build_launch_inspection(
            pipeline,
            runs_dir=str((Path.cwd() / ".agentflow" / "doctor").resolve()),
        )
    except (AttributeError, OSError, RuntimeError, TemplateError, TypeError, ValueError) as exc:
        if not isinstance(pipeline, PipelineSpec) and isinstance(exc, AttributeError):
            _PIPELINE_LAUNCH_INSPECTION_ERRORS.pop(id(pipeline), None)
            return []
        _PIPELINE_LAUNCH_INSPECTION_ERRORS[id(pipeline)] = _format_launch_inspection_error(exc)
        return []
    _PIPELINE_LAUNCH_INSPECTION_ERRORS.pop(id(pipeline), None)

    nodes = report.get("nodes")
    if not isinstance(nodes, list):
        return []

    return [node for node in nodes if isinstance(node, dict)]


def _format_launch_inspection_error(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _pipeline_launch_inspection_error(pipeline: object | None) -> str | None:
    if pipeline is None:
        return None
    return _PIPELINE_LAUNCH_INSPECTION_ERRORS.get(id(pipeline))


def _pipeline_has_local_preflight_relevant_nodes(pipeline: object | None) -> bool:
    if pipeline is None:
        return False

    for node in getattr(pipeline, "nodes", None) or []:
        agent = _status_value(getattr(node, "agent", None)).lower()
        if agent not in _KIMI_SHELL_PREFLIGHT_AGENTS:
            continue
        target = getattr(node, "target", None)
        if getattr(target, "kind", None) == "local":
            return True
    return False


def _pipeline_launch_inspection_failed_for_preflight(pipeline: object | None) -> bool:
    return bool(_pipeline_launch_inspection_error(pipeline)) and _pipeline_has_local_preflight_relevant_nodes(pipeline)


def _pipeline_launch_inspection_failure_checks(pipeline: object | None) -> list[DoctorCheck]:
    detail = _pipeline_launch_inspection_error(pipeline)
    if not detail or not _pipeline_has_local_preflight_relevant_nodes(pipeline):
        return []
    return [
        DoctorCheck(
            name="launch_inspection",
            status="failed",
            detail=(
                "AgentFlow could not inspect the resolved local launch plan for preflight safety checks: "
                f"{detail}."
            ),
        )
    ]


def _shell_bridge_recommendation_from_payload(payload: object) -> ShellBridgeRecommendation | None:
    if not isinstance(payload, dict):
        return None

    target = payload.get("target")
    source = payload.get("source")
    snippet = payload.get("snippet")
    reason = payload.get("reason")
    if not all(isinstance(value, str) and value for value in (target, source, snippet, reason)):
        return None

    return ShellBridgeRecommendation(
        target=target,
        source=source,
        snippet=snippet,
        reason=reason,
    )


def _pipeline_shell_bridge_recommendation(pipeline: object | None) -> ShellBridgeRecommendation | None:
    if pipeline is None:
        return None

    for node in _pipeline_launch_inspection_nodes(pipeline):
        recommendation = _shell_bridge_recommendation_from_payload(node.get("shell_bridge"))
        if recommendation is not None:
            return recommendation
    return None


def _node_auth_depends_on_local_shell_bootstrap(node: dict[str, object]) -> bool:
    from agentflow.inspection import inspection_node_auth_depends_on_local_shell_bootstrap

    return inspection_node_auth_depends_on_local_shell_bootstrap(node)


def _pipeline_auto_shell_bridge_recommendation(pipeline: object | None) -> ShellBridgeRecommendation | None:
    if pipeline is None:
        return None

    for node in _pipeline_launch_inspection_nodes(pipeline):
        if not isinstance(node, dict):
            continue
        if not _node_auth_depends_on_local_shell_bootstrap(node):
            continue
        recommendation = _shell_bridge_recommendation_from_payload(node.get("shell_bridge"))
        if recommendation is not None:
            return recommendation
    return None


def _pipeline_launch_env_override_checks(nodes: list[dict[str, object]]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in nodes:
        node_id = str(node.get("id") or "node")
        for override in node.get("launch_env_overrides", []) or []:
            if not isinstance(override, dict):
                continue
            key = str(override.get("key") or "")
            if not key:
                continue

            source = override.get("source")
            source_label = f" via `{source}`"
            if source == "provider.api_key_env":
                source_env_key = override.get("source_env_key")
                if isinstance(source_env_key, str) and source_env_key:
                    source_label = f" via `provider.api_key_env` (`{source_env_key}`)"
            if not isinstance(source, str) or not source:
                source_label = ""

            status = "warning"
            if source in {
                "node.env",
                "provider.env",
                "provider.base_url",
                "provider.headers",
                "provider.api_key_env",
            }:
                status = "ok"

            if status == "ok":
                detail = f"Node `{node_id}`: Launch env uses configured `{key}` for this node{source_label}."
            else:
                detail = f"Node `{node_id}`: Launch env overrides current `{key}` for this node{source_label}."
            if override.get("redacted") and override.get("cleared"):
                detail = f"Node `{node_id}`: Launch env clears current `{key}` for this node{source_label}."
            if not override.get("redacted"):
                current_value = override.get("current_value")
                launch_value = override.get("launch_value")
                if isinstance(current_value, str) and isinstance(launch_value, str):
                    if not launch_value.strip():
                        detail = (
                            f"Node `{node_id}`: Launch env clears current `{key}` value `{current_value}`"
                            f"{source_label}."
                        )
                    elif status == "ok":
                        detail = (
                            f"Node `{node_id}`: Launch env uses configured `{key}` value `{launch_value}` "
                            f"instead of current `{current_value}`{source_label}."
                        )
                    else:
                        detail = (
                            f"Node `{node_id}`: Launch env overrides current `{key}` from `{current_value}` "
                            f"to `{launch_value}`{source_label}."
                        )

            context = {"node_id": node_id, **override}
            checks.append(
                DoctorCheck(
                    name="launch_env_override",
                    status=status,
                    detail=detail,
                    context=context,
                )
            )
    return checks


def _pipeline_bootstrap_env_override_checks(nodes: list[dict[str, object]]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in nodes:
        node_id = str(node.get("id") or "node")
        for override in node.get("bootstrap_env_overrides", []) or []:
            if not isinstance(override, dict):
                continue

            key = str(override.get("key") or "")
            if not key:
                continue

            source = override.get("source")
            source_label = f" via `{source}`"
            if override.get("helper") == "kimi" and isinstance(source, str) and source:
                source_label = f" via `{source}` (`kimi` helper)"
            elif source == "target.bash_startup":
                source_label = " via local bash startup files"
            elif not isinstance(source, str) or not source:
                source_label = ""

            subject = "launch" if override.get("origin") == "launch_env" else "current"
            detail = f"Node `{node_id}`: Local shell bootstrap overrides {subject} `{key}` for this node{source_label}."
            if not override.get("redacted"):
                current_value = override.get("current_value")
                bootstrap_value = override.get("bootstrap_value")
                origin = override.get("origin")
                subject = "launch" if origin == "launch_env" else "current"
                if isinstance(current_value, str) and isinstance(bootstrap_value, str):
                    if current_value.strip():
                        detail = (
                            f"Node `{node_id}`: Local shell bootstrap overrides {subject} `{key}` from "
                            f"`{current_value}` to `{bootstrap_value}`{source_label}."
                        )
                    else:
                        detail = (
                            f"Node `{node_id}`: Local shell bootstrap sets {subject} `{key}` to "
                            f"`{bootstrap_value}`{source_label}."
                        )

            checks.append(
                DoctorCheck(
                    name="bootstrap_env_override",
                    status="ok",
                    detail=detail,
                    context={"node_id": node_id, **override},
                )
            )
    return checks


def _pipeline_launch_env_inheritance_checks(nodes: list[dict[str, object]]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in nodes:
        node_id = str(node.get("id") or "node")
        agent_name = str(node.get("agent") or "agent").capitalize()
        for inheritance in node.get("launch_env_inheritances", []) or []:
            if not isinstance(inheritance, dict):
                continue
            key = str(inheritance.get("key") or "")
            current_value = str(inheritance.get("current_value") or "")
            if not key or not current_value:
                continue

            detail = (
                f"Node `{node_id}`: Launch inherits current `{key}` value `{current_value}`; configure `provider` "
                f"or `node.env` explicitly if you want {agent_name} routing pinned for this node."
            )
            checks.append(
                DoctorCheck(
                    name="launch_env_inheritance",
                    status="warning",
                    detail=detail,
                    context={"node_id": node_id, **inheritance},
                )
            )
    return checks


def _doctor_report_for_path(path: str | None = None) -> tuple[object, dict[str, object] | None, object | None]:
    if path is None:
        selected_path = default_smoke_pipeline_path()
        report = _doctor_report()
        try:
            pipeline = _load_pipeline(selected_path)
        except typer.Exit:
            return report, None, None
        include_ok_local_checks = _include_ok_local_preflight_checks(selected_path, pipeline)
        return (
            _augment_preflight_report(
                report,
                pipeline,
                include_ok_local_checks=include_ok_local_checks,
            ),
            {"auto_preflight": _auto_smoke_preflight_metadata(selected_path, pipeline)},
            pipeline,
        )
    pipeline = _load_pipeline(path)
    report = _preflight_base_report(path, pipeline)
    include_ok_local_checks = _include_ok_local_preflight_checks(path, pipeline)
    return (
        _augment_preflight_report(report, pipeline, include_ok_local_checks=include_ok_local_checks),
        {"auto_preflight": _auto_smoke_preflight_metadata(path, pipeline)},
        pipeline,
    )


def _preflight_shell_bridge_recommendation(
    report: object,
    *,
    pipeline: object | None = None,
) -> ShellBridgeRecommendation | None:
    for check in getattr(report, "checks", []) or []:
        if getattr(check, "name", None) != "bash_login_startup":
            continue
        if _status_value(getattr(check, "status", "unknown")) not in {"warning", "failed"}:
            continue
        return _pipeline_shell_bridge_recommendation(pipeline) or build_bash_login_shell_bridge_recommendation()
    pipeline_recommendation = _pipeline_auto_shell_bridge_recommendation(pipeline)
    if pipeline_recommendation is not None:
        return pipeline_recommendation
    if pipeline is not None and _status_value(getattr(report, "status", "ok")) == "failed" and _pipeline_uses_auto_preflight(pipeline):
        return _pipeline_shell_bridge_recommendation(pipeline)
    return None


def _doctor_shell_bridge_output(
    report: object,
    *,
    requested: bool,
    pipeline: object | None = None,
) -> tuple[bool, ShellBridgeRecommendation | None]:
    if requested:
        return True, _pipeline_shell_bridge_recommendation(pipeline) or build_bash_login_shell_bridge_recommendation()

    recommendation = _preflight_shell_bridge_recommendation(report, pipeline=pipeline)
    return recommendation is not None, recommendation


def _structured_output_from_run_output(output: RunOutputFormat) -> StructuredOutputFormat:
    # Keep preflight/doctor payloads aligned with the stdout-facing run mode so wrappers can
    # redirect stdout without unexpectedly flipping stderr back to a human summary.
    resolved_output = _resolve_run_output(output, err=False)
    if resolved_output == RunOutputFormat.SUMMARY:
        return StructuredOutputFormat.SUMMARY
    if resolved_output == RunOutputFormat.JSON_SUMMARY:
        return StructuredOutputFormat.JSON_SUMMARY
    return StructuredOutputFormat.JSON


def _is_click_testing_stream(stream: object) -> bool:
    stream_type = type(stream)
    return stream_type.__module__ == "click.testing" and stream_type.__name__ == "_NamedTextIOWrapper"


def _stream_supports_tty_summary(*, err: bool) -> bool:
    stream = sys.stderr if err else sys.stdout
    if _is_click_testing_stream(stream):
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _resolve_structured_output(output: StructuredOutputFormat, *, err: bool) -> StructuredOutputFormat:
    if output != StructuredOutputFormat.AUTO:
        return output
    if _stream_supports_tty_summary(err=err):
        return StructuredOutputFormat.SUMMARY
    return StructuredOutputFormat.JSON


def _node_uses_kimi_smoke_bootstrap(node: object) -> bool:
    return _node_kimi_smoke_preflight_match(node) is not None


def _node_kimi_shell_bootstrap_check(node: object) -> DoctorCheck | None:
    agent = _status_value(getattr(node, "agent", None)).lower()
    if agent not in _KIMI_SHELL_PREFLIGHT_AGENTS:
        return None

    target = getattr(node, "target", None)
    if getattr(target, "kind", None) != "local":
        return None

    node_id = str(getattr(node, "id", "node"))
    agent = _status_value(getattr(node, "agent", None)).lower()
    provider = None
    if agent in {member.value for member in AgentKind}:
        provider = resolve_provider(getattr(node, "provider", None), AgentKind(agent))
    launch_env = merge_env_layers(getattr(provider, "env", None), getattr(node, "env", None))
    launch_cwd = _local_target_launch_cwd(node)
    effective_home = target_bash_home(target, env=launch_env, cwd=launch_cwd)

    bash_warning = kimi_shell_init_requires_bash_warning(target)
    if bash_warning is not None:
        return DoctorCheck(
            name="kimi_shell_bootstrap",
            status="failed",
            detail=f"Node `{node_id}`: {bash_warning}",
        )

    interactive_warning = kimi_shell_init_requires_interactive_bash_warning(
        target,
        home=effective_home,
        cwd=launch_cwd,
        env=launch_env,
    )
    if interactive_warning is not None:
        return DoctorCheck(
            name="kimi_shell_bootstrap",
            status="warning",
            detail=f"Node `{node_id}`: {interactive_warning}",
        )

    return None


def _node_kimi_smoke_preflight_match(node: object) -> dict[str, str] | None:
    agent = _status_value(getattr(node, "agent", None)).lower()
    if agent not in _KIMI_SHELL_PREFLIGHT_AGENTS:
        return None

    target = getattr(node, "target", None)
    if getattr(target, "kind", None) != "local":
        return None

    node_id = str(getattr(node, "id", None) or agent)

    if str(getattr(target, "bootstrap", "")).strip().lower() == "kimi":
        return {
            "node_id": node_id,
            "agent": agent,
            "trigger": "target.bootstrap",
        }

    shell_init = getattr(target, "shell_init", None)
    if shell_init_uses_kimi_helper(shell_init):
        return {
            "node_id": node_id,
            "agent": agent,
            "trigger": "target.shell_init",
        }

    shell = getattr(target, "shell", None)
    if shell_command_uses_kimi_helper(shell if isinstance(shell, str) else None):
        return {
            "node_id": node_id,
            "agent": agent,
            "trigger": "target.shell",
        }
    return None


def _node_auto_preflight_match(node: object) -> dict[str, str] | None:
    bootstrap_match = _node_kimi_smoke_preflight_match(node)
    if bootstrap_match is not None:
        return bootstrap_match

    agent = _status_value(getattr(node, "agent", None)).lower()
    if agent not in _KIMI_SHELL_PREFLIGHT_AGENTS:
        return None

    target = getattr(node, "target", None)
    if getattr(target, "kind", None) != "local":
        return None

    node_id = str(getattr(node, "id", None) or agent)
    if agent == AgentKind.KIMI.value:
        return {
            "node_id": node_id,
            "agent": agent,
            "trigger": "agent",
        }

    if agent == AgentKind.CLAUDE.value:
        provider = resolve_provider(getattr(node, "provider", None), AgentKind.CLAUDE)
        if provider_uses_kimi_anthropic_auth(provider):
            return {
                "node_id": node_id,
                "agent": agent,
                "trigger": "provider",
            }

    return None


def _inspection_node_uses_local_target(node: dict[str, object]) -> bool:
    target = node.get("target")
    if isinstance(target, dict) and str(target.get("kind") or "").lower() == "local":
        return True

    launch = node.get("launch")
    return isinstance(launch, dict) and str(launch.get("kind") or "").lower() == "local"


def _inspection_node_auto_preflight_match(node: dict[str, object]) -> dict[str, str] | None:
    agent = str(node.get("agent") or "").strip().lower()
    if agent not in _KIMI_SHELL_PREFLIGHT_AGENTS:
        return None
    if not _inspection_node_uses_local_target(node):
        return None
    if not _node_auth_depends_on_local_shell_bootstrap(node):
        return None

    node_id = str(node.get("id") or agent)
    return {
        "node_id": node_id,
        "agent": agent,
        "trigger": "target.bash_startup",
    }


def _pipeline_kimi_smoke_preflight_matches(pipeline: object) -> list[dict[str, str]]:
    nodes = getattr(pipeline, "nodes", None) or []
    matches: list[dict[str, str]] = []
    for node in nodes:
        match = _node_kimi_smoke_preflight_match(node)
        if match is not None:
            matches.append(match)
    return matches


def _pipeline_auto_preflight_matches(pipeline: object) -> list[dict[str, str]]:
    nodes = getattr(pipeline, "nodes", None) or []
    matches: list[dict[str, str]] = []
    matched_node_ids: set[str] = set()
    for node in nodes:
        match = _node_auto_preflight_match(node)
        if match is not None:
            matches.append(match)
            matched_node_ids.add(match["node_id"])

    for node in _pipeline_launch_inspection_nodes(pipeline):
        match = _inspection_node_auto_preflight_match(node)
        if match is None or match["node_id"] in matched_node_ids:
            continue
        matches.append(match)
        matched_node_ids.add(match["node_id"])
    return matches


def _render_kimi_smoke_preflight_matches(matches: list[dict[str, str]]) -> list[str]:
    rendered: list[str] = []
    for match in matches:
        node_id = match["node_id"]
        agent = match["agent"]
        trigger = match["trigger"]
        rendered.append(f"{node_id} ({agent}) via `{trigger}`")
    return rendered


def _pipeline_uses_kimi_smoke_preflight(pipeline: object) -> bool:
    return bool(_pipeline_kimi_smoke_preflight_matches(pipeline))


def _pipeline_uses_auto_preflight(pipeline: object) -> bool:
    return bool(_pipeline_auto_preflight_matches(pipeline)) or _pipeline_launch_inspection_failed_for_preflight(pipeline)


def _auto_preflight_reason_for_matches(matches: list[dict[str, str]], *, pipeline: object | None = None) -> str | None:
    if _pipeline_launch_inspection_failed_for_preflight(pipeline):
        return "AgentFlow could not inspect local launch details while deciding whether shell-startup auth preflight is required."
    if any(match.get("trigger") == "target.bash_startup" for match in matches):
        return "local Codex/Claude/Kimi nodes depend on shell startup for auth."
    if matches:
        return "local Kimi-backed nodes require pipeline-specific readiness checks."
    return None


def _pipeline_kimi_shell_bootstrap_checks(pipeline: object) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in getattr(pipeline, "nodes", None) or []:
        check = _node_kimi_shell_bootstrap_check(node)
        if check is None:
            continue
        checks.append(check)
    return checks


def _target_value(target: object, key: str, default: object | None = None) -> object | None:
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _coerce_local_target(target: object) -> LocalTarget | None:
    if _status_value(_target_value(target, "kind")).lower() != "local":
        return None

    payload = {
        "kind": "local",
        "cwd": _target_value(target, "cwd"),
        "shell": _target_value(target, "shell"),
        "shell_login": bool(_target_value(target, "shell_login", False)),
        "shell_interactive": bool(_target_value(target, "shell_interactive", False)),
        "shell_init": _target_value(target, "shell_init"),
    }
    try:
        return LocalTarget.model_validate(payload)
    except ValidationError:
        # Doctor/preflight helpers still need to reason about local shell wiring even when
        # they are given an ad hoc pipeline-like object instead of a fully validated spec.
        return LocalTarget.model_construct(**payload)


def _local_target_launch_cwd(node: object, pipeline: object | None = None) -> Path | None:
    target = _coerce_local_target(getattr(node, "target", None))
    if target is None:
        return None

    working_path = getattr(node, "working_path", None)
    if working_path is None and pipeline is not None:
        working_path = getattr(pipeline, "working_path", None)
    pipeline_workdir = Path(str(working_path or Path.cwd())).expanduser().resolve()
    return resolve_local_workdir(pipeline_workdir, target.cwd)


def _resolved_provider_api_key_env(node: object) -> tuple[str | None, str | None]:
    agent = _status_value(getattr(node, "agent", None)).lower()
    if agent not in {member.value for member in AgentKind}:
        return None, None

    provider = resolve_provider(getattr(node, "provider", None), AgentKind(agent))
    if provider is not None and provider.api_key_env:
        return provider.api_key_env, provider.name
    if agent == AgentKind.CLAUDE.value:
        return "ANTHROPIC_API_KEY", "anthropic"
    if agent == AgentKind.KIMI.value:
        return "KIMI_API_KEY", "moonshot"
    return None, None


def _provider_credentials_defer_to_local_codex_auth(node: object, *, api_key_env: str) -> bool:
    if api_key_env != "OPENAI_API_KEY":
        return False
    if _status_value(getattr(node, "agent", None)).lower() != AgentKind.CODEX.value:
        return False
    return _coerce_local_target(getattr(node, "target", None)) is not None


def _format_timeout_seconds(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}s"
    return f"{value:g}s"


def _has_nonempty_shell_value(value: str | None) -> bool:
    return bool(isinstance(value, str) and value.strip())


def _provider_credentials_local_bootstrap_probe(
    node: object,
    *,
    api_key_env: str,
    provider: object | None,
    pipeline: object | None = None,
) -> _LocalBootstrapCredentialProbe:
    target = _coerce_local_target(getattr(node, "target", None))
    if target is not None:
        launch_env = merge_env_layers(getattr(provider, "env", None), getattr(node, "env", None))
        launch_cwd = _local_target_launch_cwd(node, pipeline)
        effective_home = target_bash_home(target, env=launch_env, cwd=launch_cwd)
        shell_init = getattr(target, "shell_init", None)
        if _has_nonempty_shell_value(
            shell_init_exported_env_var_value(
                shell_init,
                api_key_env,
                home=effective_home,
                cwd=launch_cwd,
                env=launch_env,
            )
        ):
            return _LocalBootstrapCredentialProbe(found=True)

        shell = getattr(target, "shell", None)
        if _has_nonempty_shell_value(
            shell_template_exported_env_var_value_before_command(
                shell if isinstance(shell, str) else None,
                api_key_env,
                home=effective_home,
                cwd=launch_cwd,
                env=launch_env,
                interactive_bash=target_uses_interactive_bash(target),
            )
        ):
            return _LocalBootstrapCredentialProbe(found=True)
        if _has_nonempty_shell_value(shell_command_prefix_env_value(shell if isinstance(shell, str) else None, api_key_env)):
            return _LocalBootstrapCredentialProbe(found=True)

        startup_probe = probe_target_bash_startup_env_var(
            target,
            api_key_env,
            home=effective_home,
            env=launch_env,
            cwd=launch_cwd,
        )
        if startup_probe.exported:
            return _LocalBootstrapCredentialProbe(found=True)
        if startup_probe.timeout_seconds is not None:
            return _LocalBootstrapCredentialProbe(found=False, timeout_seconds=startup_probe.timeout_seconds)

    if api_key_env == "ANTHROPIC_API_KEY":
        return _LocalBootstrapCredentialProbe(found=_node_uses_kimi_smoke_bootstrap(node))
    return _LocalBootstrapCredentialProbe(found=False)


def _provider_credentials_probe_timeout_check(
    *,
    node_id: str,
    agent: str,
    api_key_env: str,
    provider_name: str | None,
    timeout_seconds: float,
) -> DoctorCheck:
    provider_detail = f" provider `{provider_name}`" if provider_name else ""
    timeout_detail = _format_timeout_seconds(timeout_seconds)
    return DoctorCheck(
        name="provider_credentials_probe",
        status="warning",
        detail=(
            f"Node `{node_id}` ({agent}) could not confirm `{api_key_env}` for{provider_detail} from local bash "
            f"startup because the probe timed out after {timeout_detail}. Fix the shell startup or increase "
            "`AGENTFLOW_BASH_STARTUP_PROBE_TIMEOUT_SECONDS`."
        ),
        context={
            "node_id": node_id,
            "agent": agent,
            "api_key_env": api_key_env,
            "timeout_seconds": timeout_seconds,
        },
    )


def _effective_launch_env_value(key: str, launch_env: dict[str, str], *, use_current_env: bool = True) -> str:
    if key in launch_env:
        return str(launch_env.get(key, "") or "")
    if not use_current_env:
        return ""
    return str(os.getenv(key, "") or "")


def _provider_credentials_override_source(
    api_key_env: str,
    *,
    node_env: dict[str, str],
    provider_env: dict[str, str],
) -> str | None:
    if api_key_env in node_env:
        return "node.env"
    if api_key_env in provider_env:
        return "provider.env"
    return None


def _provider_credentials_missing_detail(
    *,
    node_id: str,
    agent: str,
    api_key_env: str,
    provider_name: str | None,
    launch_env: dict[str, str],
    node_env: dict[str, str],
    provider_env: dict[str, str],
    shell_overrides_env: bool = False,
) -> str:
    provider_detail = f" provider `{provider_name}`" if provider_name else ""
    current_value = str(os.getenv(api_key_env, "") or "").strip()
    launch_value = _effective_launch_env_value(
        api_key_env,
        launch_env,
        use_current_env=not shell_overrides_env,
    ).strip()
    override_source = _provider_credentials_override_source(
        api_key_env,
        node_env=node_env,
        provider_env=provider_env,
    )
    if current_value and api_key_env in launch_env and not launch_value:
        source_detail = f" via `{override_source}`" if override_source else ""
        return (
            f"Node `{node_id}` ({agent}) requires `{api_key_env}` for{provider_detail}, but the launch env clears "
            f"the current environment value{source_detail}."
        )
    if current_value and shell_overrides_env and not launch_value:
        return (
            f"Node `{node_id}` ({agent}) requires `{api_key_env}` for{provider_detail}, but `target.shell` "
            "overrides or clears the current environment value before launch."
        )

    return (
        f"Node `{node_id}` ({agent}) requires `{api_key_env}` for{provider_detail}, but it is not set in "
        "the current environment, `node.env`, or `provider.env`."
    )


def _pipeline_provider_credential_checks(pipeline: object) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in getattr(pipeline, "nodes", None) or []:
        node_id = str(getattr(node, "id", "node"))
        agent = _status_value(getattr(node, "agent", None)).lower()
        api_key_env, provider_name = _resolved_provider_api_key_env(node)
        if not api_key_env:
            continue
        if _provider_credentials_defer_to_local_codex_auth(node, api_key_env=api_key_env):
            continue

        node_env = getattr(node, "env", None) or {}
        provider = resolve_provider(getattr(node, "provider", None), AgentKind(_status_value(getattr(node, "agent", None)).lower()))
        provider_env = getattr(provider, "env", None) or {}
        launch_env = merge_env_layers(provider_env, node_env)
        target = _coerce_local_target(getattr(node, "target", None))
        shell = getattr(target, "shell", None) if target is not None else None
        shell_overrides_env = isinstance(shell, str) and shell_command_overrides_env_var(shell, api_key_env)
        has_key = bool(
            _effective_launch_env_value(
                api_key_env,
                launch_env,
                use_current_env=not shell_overrides_env,
            ).strip()
        )
        bootstrap_probe = _provider_credentials_local_bootstrap_probe(
            node,
            api_key_env=api_key_env,
            provider=provider,
            pipeline=pipeline,
        )
        if not has_key and bootstrap_probe.found:
            has_key = True
        if has_key:
            continue

        if bootstrap_probe.timeout_seconds is not None:
            checks.append(
                _provider_credentials_probe_timeout_check(
                    node_id=node_id,
                    agent=agent,
                    api_key_env=api_key_env,
                    provider_name=provider_name,
                    timeout_seconds=bootstrap_probe.timeout_seconds,
                )
            )
            continue

        provider_detail = f" provider `{provider_name}`" if provider_name else ""
        checks.append(
            DoctorCheck(
                name="provider_credentials",
                status="failed",
                detail=_provider_credentials_missing_detail(
                    node_id=node_id,
                    agent=agent,
                    api_key_env=api_key_env,
                    provider_name=provider_name,
                    launch_env=launch_env,
                    node_env=node_env,
                    provider_env=provider_env,
                    shell_overrides_env=shell_overrides_env,
                ),
            )
        )
    return checks


def _merge_doctor_status(current_status: str, extra_checks: list[DoctorCheck]) -> str:
    statuses = {current_status, *(_status_value(check.status) for check in extra_checks)}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return current_status


def _pipeline_launch_bash_login_startup_checks(nodes: list[dict[str, object]]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for node in nodes:
        warnings = node.get("warnings")
        if not isinstance(warnings, list):
            continue

        node_id = str(node.get("id") or "node")
        bootstrap_home = node.get("bootstrap_home")
        for warning in warnings:
            if not isinstance(warning, str) or not warning.startswith("Bash login startup"):
                continue

            if isinstance(bootstrap_home, str) and bootstrap_home:
                detail = f"Node `{node_id}` uses bash login startup from `{bootstrap_home}`: {warning}"
            else:
                detail = f"Node `{node_id}`: {warning}"
            context: dict[str, object] = {"node_id": node_id}
            if isinstance(bootstrap_home, str) and bootstrap_home:
                context["bootstrap_home"] = bootstrap_home
            checks.append(
                DoctorCheck(
                    name="bash_login_startup",
                    status="warning",
                    detail=detail,
                    context=context,
                )
            )
    return checks


def _augment_preflight_report(
    report: object,
    pipeline: object,
    *,
    include_ok_local_checks: bool = False,
) -> object:
    report = _extend_doctor_report(
        report,
        [
            *_pipeline_kimi_shell_bootstrap_checks(pipeline),
            *_pipeline_provider_credential_checks(pipeline),
            *build_pipeline_local_kimi_readiness_checks(pipeline),
            *build_pipeline_local_claude_readiness_checks(pipeline),
            *build_pipeline_local_codex_readiness_checks(pipeline),
            *build_pipeline_local_codex_auth_checks(pipeline),
        ],
    )
    if include_ok_local_checks and _status_value(getattr(report, "status", "ok")) != "failed":
        report = _extend_doctor_report(
            report,
            [
                *build_pipeline_local_kimi_readiness_info_checks(pipeline),
                *build_pipeline_local_claude_readiness_info_checks(pipeline),
                *build_pipeline_local_codex_readiness_info_checks(pipeline),
                *build_pipeline_local_codex_auth_info_checks(pipeline),
            ],
        )
    if _status_value(getattr(report, "status", "ok")) == "failed":
        return report

    inspection_nodes = _pipeline_launch_inspection_nodes(pipeline)
    inspection_failure_checks = _pipeline_launch_inspection_failure_checks(pipeline)
    if inspection_failure_checks:
        return _extend_doctor_report(report, inspection_failure_checks)
    return _extend_doctor_report(
        report,
        [
            *_pipeline_launch_bash_login_startup_checks(inspection_nodes),
            *_pipeline_launch_env_override_checks(inspection_nodes),
            *_pipeline_bootstrap_env_override_checks(inspection_nodes),
            *_pipeline_launch_env_inheritance_checks(inspection_nodes),
        ],
    )


def _auto_smoke_preflight_reason(path: str, pipeline: object) -> str | None:
    if _path_matches_bundled_smoke(path):
        return "path matches the bundled real-agent smoke pipeline."
    if _pipeline_uses_kimi_smoke_preflight(pipeline):
        return "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap."
    return _auto_preflight_reason_for_matches(_pipeline_auto_preflight_matches(pipeline), pipeline=pipeline)


def _auto_smoke_preflight_metadata(path: str, pipeline: object) -> dict[str, object]:
    matches = _pipeline_auto_preflight_matches(pipeline)
    match_summary = _render_kimi_smoke_preflight_matches(matches)
    reason = (
        "path matches the bundled real-agent smoke pipeline."
        if _path_matches_bundled_smoke(path)
        else "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap."
        if _pipeline_uses_kimi_smoke_preflight(pipeline)
        else _auto_preflight_reason_for_matches(matches, pipeline=pipeline)
    )
    if reason is not None:
        return {
            "enabled": True,
            "reason": reason,
            "matches": matches,
            "match_summary": match_summary,
        }
    return {
        "enabled": False,
        "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
        "matches": matches,
        "match_summary": match_summary,
    }


def _include_ok_local_preflight_checks(path: str, pipeline: object) -> bool:
    return _path_matches_bundled_smoke(path) or _pipeline_uses_auto_preflight(pipeline)


def _should_run_smoke_preflight(
    path: str | None,
    preflight: SmokePreflightMode,
    *,
    pipeline: object | None = None,
) -> bool:
    if preflight == SmokePreflightMode.ALWAYS:
        return True
    if preflight == SmokePreflightMode.NEVER:
        return False
    if path is None:
        return True
    if _path_matches_bundled_smoke(path):
        return True
    if pipeline is None:
        return False
    return _pipeline_uses_auto_preflight(pipeline)


def _load_pipeline_with_optional_smoke_preflight(
    path: str | None,
    selected_path: str,
    preflight: SmokePreflightMode,
    output: RunOutputFormat,
    *,
    show_preflight: bool = False,
) -> object:
    pipeline = None
    should_run_preflight = _should_run_smoke_preflight(path, preflight)
    selected_path_matches_bundled = (
        Path(selected_path).expanduser().resolve() == Path(default_smoke_pipeline_path()).expanduser().resolve()
    )
    if not should_run_preflight and preflight == SmokePreflightMode.AUTO and path is not None:
        pipeline = _load_pipeline(selected_path)
        should_run_preflight = _should_run_smoke_preflight(path, preflight, pipeline=pipeline)

    if should_run_preflight:
        if pipeline is None and preflight == SmokePreflightMode.ALWAYS and not selected_path_matches_bundled:
            pipeline = _load_pipeline(selected_path)
        preflight_pipeline = pipeline
        if pipeline is not None:
            base_report = _preflight_base_report(path or selected_path, pipeline)
            report = _augment_preflight_report(
                base_report,
                pipeline,
                include_ok_local_checks=_include_ok_local_preflight_checks(path or selected_path, pipeline),
            )
        else:
            report = _doctor_report()
        if pipeline is None and selected_path_matches_bundled and _status_value(getattr(report, "status", "ok")) != "failed":
            preflight_pipeline = _load_pipeline(selected_path)
            report = _augment_preflight_report(
                report,
                preflight_pipeline,
                include_ok_local_checks=_include_ok_local_preflight_checks(path or selected_path, preflight_pipeline),
            )
        doctor_output = _structured_output_from_run_output(output)
        shell_bridge = _preflight_shell_bridge_recommendation(report, pipeline=preflight_pipeline)
        include_shell_bridge = shell_bridge is not None
        preflight_context = None
        if preflight_pipeline is not None:
            preflight_context = {
                "auto_preflight": _auto_smoke_preflight_metadata(path or selected_path, preflight_pipeline)
            }
        if report.status == "failed":
            _echo_doctor_report(
                report,
                output=doctor_output,
                include_shell_bridge=include_shell_bridge,
                shell_bridge=shell_bridge,
                pipeline=preflight_context,
            )
            raise typer.Exit(code=1)
        if report.status == "warning":
            _echo_doctor_report(
                report,
                output=doctor_output,
                err=True,
                include_shell_bridge=include_shell_bridge,
                shell_bridge=shell_bridge,
                pipeline=preflight_context,
            )
        elif show_preflight:
            _echo_doctor_report(
                report,
                output=StructuredOutputFormat.SUMMARY,
                err=True,
                include_shell_bridge=include_shell_bridge,
                shell_bridge=shell_bridge,
                pipeline=preflight_context,
            )
        if preflight_pipeline is not None:
            pipeline = preflight_pipeline

    return pipeline if pipeline is not None else _load_pipeline(selected_path)


def _render_shell_bridge_summary(shell_bridge: object | None) -> str:
    if shell_bridge is None:
        return "Shell bridge suggestion: not needed"

    return "\n".join(
        [
            (
                f"Shell bridge suggestion for `{getattr(shell_bridge, 'target', '~/.profile')}` "
                f"from `{getattr(shell_bridge, 'source', '~/.bashrc')}`:"
            ),
            f"Reason: {getattr(shell_bridge, 'reason', '')}",
            getattr(shell_bridge, "snippet", "").rstrip(),
        ]
    )


def _doctor_check_summary_suffix(check: object) -> str:
    if getattr(check, "name", None) != "bash_login_startup":
        return ""
    context = getattr(check, "context", None)
    if not isinstance(context, dict):
        return ""
    parts: list[str] = []
    startup_summary = context.get("startup_summary")
    if isinstance(startup_summary, str) and startup_summary:
        parts.append(f"startup={startup_summary}")
    startup_files_summary = context.get("startup_files_summary")
    if isinstance(startup_files_summary, str) and startup_files_summary:
        parts.append(f"files={startup_files_summary}")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def _render_doctor_summary(
    report: object,
    *,
    include_shell_bridge: bool = False,
    shell_bridge: object | None = None,
    pipeline: dict[str, object] | None = None,
) -> str:
    lines = [f"Doctor: {_status_value(getattr(report, 'status', 'unknown'))}"]
    for check in getattr(report, "checks", []) or []:
        lines.append(
            f"- {getattr(check, 'name', 'unknown')}: {_status_value(getattr(check, 'status', 'unknown'))}"
            f" - {getattr(check, 'detail', '')}{_doctor_check_summary_suffix(check)}"
        )
    auto_preflight_scope = ""
    if isinstance(pipeline, dict):
        auto_preflight_scope = str(pipeline.get("auto_preflight_scope") or "").strip().lower()
    auto_preflight_label = "Pipeline auto preflight"
    if auto_preflight_scope == "run/smoke":
        auto_preflight_label = "Pipeline run/smoke auto preflight"
    raw_auto_preflight = pipeline.get("auto_preflight") if isinstance(pipeline, dict) else None
    if isinstance(raw_auto_preflight, dict):
        enabled = raw_auto_preflight.get("enabled")
        reason = raw_auto_preflight.get("reason")
        if isinstance(reason, str) and reason:
            status = "enabled" if enabled else "disabled"
            lines.append(f"{auto_preflight_label}: {status} - {reason}")
        matches = raw_auto_preflight.get("match_summary")
        if isinstance(matches, list):
            rendered_matches = [match for match in matches if isinstance(match, str) and match]
            if rendered_matches:
                lines.append(f"{auto_preflight_label} matches: {', '.join(rendered_matches)}")
    if include_shell_bridge:
        lines.append(_render_shell_bridge_summary(shell_bridge))
    return "\n".join(lines)


def _build_doctor_payload(
    report: object,
    *,
    include_shell_bridge: bool = False,
    shell_bridge: object | None = None,
    pipeline: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = report.as_dict()
    if pipeline is not None:
        payload["pipeline"] = dict(pipeline)
    if include_shell_bridge:
        payload["shell_bridge"] = None if shell_bridge is None else shell_bridge.as_dict()
    return payload


def _build_doctor_summary_payload(
    report: object,
    *,
    include_shell_bridge: bool = False,
    shell_bridge: object | None = None,
    pipeline: dict[str, object] | None = None,
) -> dict[str, object]:
    counts = {"ok": 0, "warning": 0, "failed": 0}
    checks = []
    for check in getattr(report, "checks", []) or []:
        status = _status_value(getattr(check, "status", "unknown"))
        if status in counts:
            counts[status] += 1
        entry = {
            "name": getattr(check, "name", "unknown"),
            "status": status,
            "detail": getattr(check, "detail", ""),
        }
        if getattr(check, "name", None) == "bash_login_startup":
            context = getattr(check, "context", None)
            if isinstance(context, dict):
                startup_summary = context.get("startup_summary")
                if isinstance(startup_summary, str) and startup_summary:
                    entry["startup_summary"] = startup_summary
                startup_files = context.get("startup_files")
                if isinstance(startup_files, dict) and startup_files:
                    entry["startup_files"] = {
                        str(path): str(file_status)
                        for path, file_status in startup_files.items()
                    }
        checks.append(entry)

    payload: dict[str, object] = {
        "status": _status_value(getattr(report, "status", "unknown")),
        "counts": counts,
        "checks": checks,
    }
    if pipeline is not None:
        payload["pipeline"] = dict(pipeline)
    if include_shell_bridge:
        payload["shell_bridge"] = None if shell_bridge is None else shell_bridge.as_dict()
    return payload


def _echo_doctor_report(
    report: object,
    *,
    output: StructuredOutputFormat = StructuredOutputFormat.JSON,
    err: bool = False,
    include_shell_bridge: bool = False,
    shell_bridge: object | None = None,
    pipeline: dict[str, object] | None = None,
) -> None:
    resolved_output = _resolve_structured_output(output, err=err)
    if resolved_output == StructuredOutputFormat.SUMMARY:
        typer.echo(
            _render_doctor_summary(
                report,
                include_shell_bridge=include_shell_bridge,
                shell_bridge=shell_bridge,
                pipeline=pipeline,
            ),
            err=err,
        )
        return
    payload = (
        _build_doctor_summary_payload(
            report,
            include_shell_bridge=include_shell_bridge,
            shell_bridge=shell_bridge,
            pipeline=pipeline,
        )
        if resolved_output == StructuredOutputFormat.JSON_SUMMARY
        else _build_doctor_payload(
            report,
            include_shell_bridge=include_shell_bridge,
            shell_bridge=shell_bridge,
            pipeline=pipeline,
        )
    )
    typer.echo(
        json.dumps(
            payload,
            indent=2,
        ),
        err=err,
    )


def _render_local_toolchain_summary(report: LocalToolchainReport) -> str:
    lines = [f"Toolchain: {report.status}"]
    startup_order = ("~/.bash_profile", "~/.bash_login", "~/.profile")
    for path in startup_order:
        lines.append(f"{path}: {report.startup_files.get(path, 'missing')}")
    lines.append(f"bash login startup: {report.bash_login_startup}")
    if report.shell_bridge is None:
        lines.append("bash login bridge: not needed")
    else:
        lines.append(f"bash login bridge target: {report.shell_bridge.target}")
        lines.append(f"bash login bridge source: {report.shell_bridge.source}")
        lines.append(f"bash login bridge reason: {report.shell_bridge.reason}")
        lines.append("bash login bridge snippet:")
        for line in report.shell_bridge.snippet.rstrip().splitlines():
            lines.append(f"  {line}")
    if report.kimi_kind and report.kimi_path:
        lines.append(f"kimi: {report.kimi_kind} ({report.kimi_path})")
    elif report.kimi_kind:
        lines.append(f"kimi: {report.kimi_kind}")
    elif report.kimi_path:
        lines.append(f"kimi: {report.kimi_path}")
    if report.anthropic_base_url:
        lines.append(f"ANTHROPIC_BASE_URL={report.anthropic_base_url}")
    if report.ambient_base_urls:
        for key, value in report.ambient_base_urls.items():
            lines.append(f"ambient {key}={value}")
        lines.append(
            "routing note: bundled smoke clears or pins these values, but custom local Codex/Claude pipelines "
            "inherit them unless `provider.base_url`, `provider.env`, or `node.env` overrides routing."
        )
    if report.codex_auth:
        lines.append(f"codex auth: {report.codex_auth}")
    if report.codex_path and report.codex_version:
        lines.append(f"codex: {report.codex_path} ({report.codex_version})")
    elif report.codex_path:
        lines.append(f"codex: {report.codex_path}")
    elif report.codex_version:
        lines.append(f"codex: {report.codex_version}")
    if report.claude_path and report.claude_version:
        lines.append(f"claude: {report.claude_path} ({report.claude_version})")
    elif report.claude_path:
        lines.append(f"claude: {report.claude_path}")
    elif report.claude_version:
        lines.append(f"claude: {report.claude_version}")
    if report.detail:
        lines.append(f"detail: {report.detail}")
    return "\n".join(lines)


def _build_local_toolchain_summary_payload(report: LocalToolchainReport) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": report.status,
        "startup": {
            "bash_login_startup": report.bash_login_startup,
            "files": dict(report.startup_files),
            "shell_bridge": (
                None
                if report.shell_bridge is None
                else {
                    "target": report.shell_bridge.target,
                    "source": report.shell_bridge.source,
                    "reason": report.shell_bridge.reason,
                }
            ),
        },
    }

    kimi: dict[str, str] = {}
    if report.kimi_kind is not None:
        kimi["kind"] = report.kimi_kind
    if report.kimi_path is not None:
        kimi["path"] = report.kimi_path
    if report.anthropic_base_url is not None:
        kimi["anthropic_base_url"] = report.anthropic_base_url
    if kimi:
        payload["kimi"] = kimi

    if report.ambient_base_urls:
        payload["routing"] = {"ambient_base_urls": dict(report.ambient_base_urls)}

    codex: dict[str, str] = {}
    if report.codex_auth is not None:
        codex["auth"] = report.codex_auth
    if report.codex_path is not None:
        codex["path"] = report.codex_path
    if report.codex_version is not None:
        codex["version"] = report.codex_version
    if codex:
        payload["codex"] = codex

    claude: dict[str, str] = {}
    if report.claude_path is not None:
        claude["path"] = report.claude_path
    if report.claude_version is not None:
        claude["version"] = report.claude_version
    if claude:
        payload["claude"] = claude

    if report.detail is not None:
        payload["detail"] = report.detail
    return payload


def _echo_local_toolchain_report(
    report: LocalToolchainReport,
    *,
    output: StructuredOutputFormat = StructuredOutputFormat.AUTO,
) -> None:
    resolved_output = _resolve_structured_output(output, err=False)
    if resolved_output == StructuredOutputFormat.SUMMARY:
        typer.echo(_render_local_toolchain_summary(report))
        return
    if resolved_output == StructuredOutputFormat.JSON_SUMMARY:
        typer.echo(json.dumps(_build_local_toolchain_summary_payload(report), indent=2))
        return

    typer.echo(json.dumps(report.as_dict(), indent=2))


def _check_local_pipeline_context(pipeline: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(pipeline, dict):
        return pipeline

    context = dict(pipeline)
    if isinstance(context.get("auto_preflight"), dict):
        context["auto_preflight_scope"] = "run/smoke"
    return context


def _echo_inspection(report: dict[str, object], *, output: InspectionOutputFormat) -> None:
    from agentflow.inspection import build_launch_inspection_summary

    resolved_output = _resolve_inspection_output(output)

    if resolved_output == InspectionOutputFormat.SUMMARY:
        from agentflow.inspection import render_launch_inspection_summary

        typer.echo(render_launch_inspection_summary(report))
        return
    if resolved_output == InspectionOutputFormat.JSON_SUMMARY:
        typer.echo(json.dumps(build_launch_inspection_summary(report), indent=2))
        return
    typer.echo(json.dumps(report, indent=2))


def _resolve_inspection_output(output: InspectionOutputFormat) -> InspectionOutputFormat:
    if output != InspectionOutputFormat.AUTO:
        return output
    if _stream_supports_tty_summary(err=False):
        return InspectionOutputFormat.SUMMARY
    return InspectionOutputFormat.JSON


def _parse_template_settings(raw_settings: list[str] | None) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_setting in raw_settings or []:
        key, separator, value = raw_setting.partition("=")
        if separator != "=" or not key or not value:
            raise ValueError(f"template settings must use KEY=VALUE form, got `{raw_setting}`")
        if key in settings:
            raise ValueError(f"template setting `{key}` was provided more than once")
        settings[key] = value
    return settings


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
) -> None:
    store, orchestrator = _build_runtime(runs_dir, max_concurrent_runs)
    _serve_web_app(_create_web_app(store=store, orchestrator=orchestrator), host=host, port=port)


@app.command()
def validate(path: str) -> None:
    pipeline = _load_pipeline(path)
    typer.echo(json.dumps(pipeline.model_dump(mode="json"), indent=2))


@app.command()
def templates() -> None:
    lines = ["Bundled templates:"]
    for template in bundled_templates():
        details = [
            f"source: `examples/{template.example_name}`",
            f"use: `agentflow init --template {template.name}`",
        ]
        if template.support_files:
            details.insert(0, "assets: " + ", ".join(f"`{path}`" for path in template.support_files))
        if template.parameters:
            details.insert(
                0,
                "params: " + ", ".join(f"`{parameter.name}={parameter.default}`" for parameter in template.parameters),
            )
        lines.append(
            f"- {template.name}: {template.description} "
            f"({'; '.join(details)})"
        )
    typer.echo("\n".join(lines))


@app.command()
def init(
    path: str = typer.Argument(
        "",
        help="Optional destination path. When omitted or `-`, print the selected template to stdout.",
    ),
    template: str = typer.Option(
        "pipeline",
        "--template",
        "-t",
        help=f"Bundled template name ({', '.join(bundled_template_names())}). Use `agentflow templates` to list details.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing destination file.",
    ),
    set_value: list[str] = typer.Option(
        None,
        "--set",
        help="Template setting in KEY=VALUE form. Repeat to customize parameterized templates.",
    ),
) -> None:
    try:
        template_settings = _parse_template_settings(set_value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--set") from exc

    try:
        rendered_template = render_bundled_template(template, values=template_settings)
    except ValueError as exc:
        param_hint = "--template" if template not in bundled_template_names() else "--set"
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc
    support_files = rendered_template.support_files

    if not path or path == "-":
        if support_files:
            typer.echo(
                f"Template `{template}` includes support files and requires a destination path.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(rendered_template.content, nl=False)
        return

    destination = Path(path).expanduser()
    if destination.exists() and destination.is_dir():
        typer.echo(f"Destination `{destination}` is a directory.", err=True)
        raise typer.Exit(code=1)
    if destination.exists() and not force:
        typer.echo(f"Destination `{destination}` already exists. Use `--force` to overwrite it.", err=True)
        raise typer.Exit(code=1)

    support_copies: list[tuple[str, str, Path]] = []
    for support_file in support_files:
        target = destination.parent / support_file.relative_path
        support_copies.append((support_file.relative_path, support_file.content, target))
        if target.exists() and not force:
            typer.echo(f"Destination `{target}` already exists. Use `--force` to overwrite it.", err=True)
            raise typer.Exit(code=1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered_template.content, encoding="utf-8")
    for _relative_path, content, target in support_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    typer.echo(f"Wrote `{template}` template to `{destination}`.")


@app.command()
def runs(
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
    limit: int = typer.Option(20, min=0, help="Maximum runs to show. Use `0` to show all persisted runs."),
) -> None:
    store = _build_store(runs_dir)
    all_runs = store.list_runs()
    selected_runs = all_runs if limit == 0 else all_runs[:limit]
    _echo_runs_result(selected_runs, store=store, output=output, total=len(all_runs))


@app.command()
def show(
    run_id: str,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    store = _build_store(runs_dir)
    record = _get_run_or_exit(store, run_id, runs_dir=runs_dir)
    _echo_run_result(record, output=output, run_dir=_run_dir_for_record(store, run_id))


@app.command()
def cancel(
    run_id: str,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    store, orchestrator = _build_runtime(runs_dir, max_concurrent_runs)

    async def _cancel() -> None:
        try:
            record = await orchestrator.cancel(run_id)
        except KeyError as exc:
            typer.echo(f"Run `{run_id}` not found in `{runs_dir}`.", err=True)
            raise typer.Exit(code=1) from exc
        _echo_run_result(record, output=output, run_dir=_run_dir_for_record(store, record.id))

    asyncio.run(_cancel())


@app.command()
def rerun(
    run_id: str,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    store, orchestrator = _build_runtime(runs_dir, max_concurrent_runs)

    async def _rerun() -> None:
        try:
            record = await orchestrator.rerun(run_id)
        except KeyError as exc:
            typer.echo(f"Run `{run_id}` not found in `{runs_dir}`.", err=True)
            raise typer.Exit(code=1) from exc
        completed = await orchestrator.wait(record.id, timeout=None)
        _echo_run_result(completed, output=output, run_dir=_run_dir_for_record(store, record.id))
        raise typer.Exit(code=0 if _status_value(completed.status) == "completed" else 1)

    asyncio.run(_rerun())


@app.command()
def resume(
    run_id: str,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    """Resume a failed or cancelled run from where it left off.

    Completed nodes are preserved and skipped; failed/cancelled/skipped nodes
    are reset to pending and re-executed. The scratchboard and artifacts from
    completed nodes are copied to the new run.
    """
    store, orchestrator = _build_runtime(runs_dir, max_concurrent_runs)

    async def _resume() -> None:
        try:
            record = await orchestrator.resume(run_id)
        except KeyError as exc:
            typer.echo(f"Run `{run_id}` not found in `{runs_dir}`.", err=True)
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Resumed as new run `{record.id}` (preserving completed nodes from `{run_id}`).")
        completed = await orchestrator.wait(record.id, timeout=None)
        _echo_run_result(completed, output=output, run_dir=_run_dir_for_record(store, record.id))
        raise typer.Exit(code=0 if _status_value(completed.status) == "completed" else 1)

    asyncio.run(_resume())


@app.command()
def evolve(
    run_id: str,
    node: list[str] = typer.Option(..., "--node", "-n", help="Source node ids to harvest traces from."),
    target: str = typer.Option("codex", help="Base agent kind to evolve."),
    optimizer: str = typer.Option("codex", help="Optimizer agent kind to patch the cloned repo."),
    profile: str = typer.Option("", "--profile", help="Tuner profile name under `agent_tuner/`. Defaults to `target`."),
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.SUMMARY,
        "--output",
        help="Structured output format for evolution results.",
    ),
) -> None:
    store = _build_store(runs_dir)
    record = _get_run_or_exit(store, run_id, runs_dir=runs_dir)
    pipeline_nodes = record.pipeline.node_map
    missing_nodes = [node_id for node_id in node if node_id not in pipeline_nodes]
    if missing_nodes:
        typer.echo(f"Unknown node ids for run `{run_id}`: {missing_nodes}", err=True)
        raise typer.Exit(code=1)

    normalized_target = target.strip()
    selected_nodes = [
        node_id
        for node_id in node
        if normalize_agent_name(pipeline_nodes[node_id].agent) == normalized_target
    ]
    if not selected_nodes:
        typer.echo(
            f"No selected nodes in run `{run_id}` use target agent `{normalized_target}`.",
            err=True,
        )
        raise typer.Exit(code=1)

    payload = {
        "profile": (profile.strip() or normalized_target),
        "target": normalized_target,
        "optimizer": optimizer.strip(),
        "source_nodes": selected_nodes,
        "trace_paths": {
            node_id: str(store.artifact_path(run_id, node_id, "trace.jsonl"))
            for node_id in selected_nodes
        },
        "workspace_dir": record.pipeline.working_dir,
        "run_id": run_id,
    }
    try:
        result = run_evolution_from_payload(payload)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if output == StructuredOutputFormat.JSON:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    typer.echo(_render_evolution_summary(result))


@app.command("tuned-agents")
def tuned_agents(
    workspace: str = typer.Option(".", help="Workspace root that holds `.agentflow/tuned_agents`."),
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.SUMMARY,
        "--output",
        help="Structured output format for tuned agent listings.",
    ),
) -> None:
    records = list_tuned_agent_records(Path(workspace).expanduser().resolve())
    if output == StructuredOutputFormat.JSON:
        typer.echo(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(_render_tuned_agents_summary(records))


@app.command("tuned-agent")
def tuned_agent(
    name: str,
    workspace: str = typer.Option(".", help="Workspace root that holds `.agentflow/tuned_agents`."),
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.SUMMARY,
        "--output",
        help="Structured output format for tuned agent details.",
    ),
) -> None:
    records = {record.name: record for record in list_tuned_agent_records(Path(workspace).expanduser().resolve())}
    record = records.get(name)
    if record is None:
        typer.echo(f"Tuned agent `{name}` not found.", err=True)
        raise typer.Exit(code=1)
    latest = resolve_tuned_agent_version(Path(workspace).expanduser().resolve(), name)
    payload = record.model_dump(mode="json")
    payload["latest"] = latest.model_dump(mode="json") if latest is not None else None
    if output == StructuredOutputFormat.JSON:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(_render_tuned_agent_detail(record))


@app.command()
def inspect(
    path: str,
    node: list[str] = typer.Option(None, "--node", "-n", help="Inspect only the selected node ids."),
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    output: InspectionOutputFormat = typer.Option(
        InspectionOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    from agentflow.inspection import build_launch_inspection

    pipeline = _load_pipeline(path)
    try:
        report = build_launch_inspection(pipeline, runs_dir=runs_dir, node_ids=node or None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--node") from exc
    report.setdefault("pipeline", {})["auto_preflight"] = _auto_smoke_preflight_metadata(path, pipeline)
    _echo_inspection(report, output=output)


@app.command()
def inference(
    model_id: str,
    gpu: str = typer.Option(
        ...,
        "--gpu",
        help="SkyPilot GPU selector, e.g. `aws:8xb200@us-east-1` or `aws:8x8xb200@us-east-2`.",
    ),
    mode: InferenceMode = typer.Option(
        InferenceMode.SERVICE,
        "--mode",
        help="Launch a reusable OpenAI-compatible service by default, or an explicit JSONL batch job.",
    ),
    engine: InferenceEngine = typer.Option(InferenceEngine.VLLM, "--engine", help="Serving engine to run remotely."),
    input_path: str | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Local JSONL input file with one object per line and a `prompt` field.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Single prompt for smoke runs when `--input` is not provided.",
    ),
    result_output: str | None = typer.Option(
        None,
        "--result-output",
        help="Remote path for JSONL inference results. Results are also emitted in job logs.",
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Prompt batch size inside the worker."),
    max_tokens: int = typer.Option(128, "--max-tokens", min=1, help="Maximum generated tokens per prompt."),
    temperature: float = typer.Option(0.0, "--temperature", min=0.0, help="Sampling temperature."),
    use_spot: bool = typer.Option(True, "--spot/--no-spot", help="Use spot/preemptible resources by default."),
    max_hourly_cost: float | None = typer.Option(None, "--max-hourly-cost", min=0.0, help="SkyPilot max hourly cost cap."),
    image_id: str | None = typer.Option(None, "--image-id", help="Explicit SkyPilot image id override."),
    name: str | None = typer.Option(None, "--name", help="Service or managed job name. Defaults to a model-derived name."),
    cluster_name: str | None = typer.Option(None, "--cluster-name", help="SkyPilot cluster name for service mode."),
    api_key: str | None = typer.Option(None, "--api-key", help="Explicit service API key. Defaults to a generated key."),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="OpenAI-compatible service port."),
    idle_minutes_to_autostop: int = typer.Option(60, "--idle-minutes-to-autostop", min=0, help="Service cluster idle autostop minutes."),
    retry_until_up: bool = typer.Option(False, "--retry-until-up", help="Keep retrying SkyPilot service provisioning until resources are available."),
    endpoint_timeout: int = typer.Option(600, "--endpoint-timeout", min=1, help="Seconds to wait for service endpoint readiness."),
    pool: str | None = typer.Option(None, "--pool", help="Submit to an existing SkyPilot jobs pool."),
    workers: int | None = typer.Option(None, "--workers", min=1, help="Number of pool jobs to launch."),
    detach: bool = typer.Option(False, "--detach", help="Submit the managed job and return after launch."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", min=1.0, help="Seconds between managed-job polls."),
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.SUMMARY,
        "--output",
        help="Command output format.",
    ),
) -> None:
    """Launch SkyPilot-backed inference with vLLM or SGLang."""

    from agentflow.inference import (
        SkyInferenceRequest,
        SkyInferenceServiceRequest,
        launch_sky_inference_job,
        launch_sky_inference_service,
        parse_gpu_selector,
    )

    try:
        parsed_gpu = parse_gpu_selector(gpu)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu") from exc

    resolved_input = Path(input_path).expanduser().resolve() if input_path else None
    if resolved_input is not None and not resolved_input.exists():
        raise typer.BadParameter(f"input file `{resolved_input}` does not exist", param_hint="--input")
    if workers is not None and not pool:
        raise typer.BadParameter("`--workers` requires `--pool` because SkyPilot maps it to pool job count", param_hint="--workers")

    if mode == InferenceMode.SERVICE:
        if any(value is not None for value in (resolved_input, prompt, result_output, pool, workers)):
            raise typer.BadParameter("`--input`, `--prompt`, `--result-output`, `--pool`, and `--workers` require `--mode batch`")
        request = SkyInferenceServiceRequest(
            model_id=model_id,
            gpu=parsed_gpu,
            engine=engine.value,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
            image_id=image_id,
            name=name,
            cluster_name=cluster_name,
            api_key=api_key,
            port=port,
            idle_minutes_to_autostop=idle_minutes_to_autostop,
            retry_until_up=retry_until_up,
            endpoint_timeout_seconds=endpoint_timeout,
        )
        try:
            result = launch_sky_inference_service(request)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        if output in {StructuredOutputFormat.JSON, StructuredOutputFormat.JSON_SUMMARY}:
            typer.echo(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
            return
        typer.echo(_render_inference_service_summary(result))
        return

    request = SkyInferenceRequest(
        model_id=model_id,
        gpu=parsed_gpu,
        engine=engine.value,
        input_path=resolved_input,
        output_path=Path(result_output) if result_output else None,
        prompt=prompt,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        use_spot=use_spot,
        max_hourly_cost=max_hourly_cost,
        image_id=image_id,
        name=name,
        workers=workers,
        detach=detach,
        pool=pool,
        wait_interval_seconds=poll_interval,
    )
    try:
        result = launch_sky_inference_job(request)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if output in {StructuredOutputFormat.JSON, StructuredOutputFormat.JSON_SUMMARY}:
        typer.echo(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
        return
    typer.echo(_render_inference_summary(result))


@app.command()
def run(
    path: str,
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
    preflight: SmokePreflightMode = typer.Option(
        SmokePreflightMode.AUTO,
        "--preflight",
        help="When to run the local smoke preflight for bundled or Kimi-bootstrapped local pipelines.",
    ),
    show_preflight: bool = typer.Option(
        False,
        "--show-preflight",
        help="Print a successful local preflight summary to stderr when preflight runs.",
    ),
) -> None:
    pipeline = _load_pipeline_with_optional_smoke_preflight(
        path,
        path,
        preflight,
        output,
        show_preflight=show_preflight,
    )
    _run_pipeline(pipeline, runs_dir, max_concurrent_runs, output)


@app.command()
def smoke(
    path: str = typer.Argument("", help="Optional pipeline path. Defaults to the bundled real-agent smoke example."),
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(RunOutputFormat.SUMMARY, "--output", help="Result output format."),
    preflight: SmokePreflightMode = typer.Option(
        SmokePreflightMode.AUTO,
        "--preflight",
        help="When to run the local smoke preflight for bundled or Kimi-bootstrapped local pipelines.",
    ),
    show_preflight: bool = typer.Option(
        False,
        "--show-preflight",
        help="Print a successful local preflight summary to stderr when preflight runs.",
    ),
) -> None:
    selected_path = path or default_smoke_pipeline_path()
    pipeline = _load_pipeline_with_optional_smoke_preflight(
        path or None,
        selected_path,
        preflight,
        output,
        show_preflight=show_preflight,
    )
    _run_pipeline(pipeline, runs_dir, max_concurrent_runs, output)


@app.command("check-local")
def check_local(
    path: str = typer.Argument("", help="Optional pipeline path. Defaults to the bundled real-agent smoke example."),
    runs_dir: str = typer.Option(".agentflow/runs", envvar="AGENTFLOW_RUNS_DIR"),
    max_concurrent_runs: int = typer.Option(2, envvar="AGENTFLOW_MAX_CONCURRENT_RUNS"),
    output: RunOutputFormat = typer.Option(
        RunOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
    preflight: SmokePreflightMode = typer.Option(
        SmokePreflightMode.ALWAYS,
        "--preflight",
        help=(
            "Accepted for CLI parity with `run` and `smoke`. "
            "`check-local` always runs the local doctor preflight, so only `auto` and `always` are allowed."
        ),
    ),
    show_preflight: bool = typer.Option(
        False,
        "--show-preflight",
        help="Accepted for CLI parity with `run` and `smoke`; `check-local` already prints preflight output to stderr.",
    ),
    shell_bridge: bool = typer.Option(
        False,
        "--shell-bridge",
        help="Include a ready-to-paste bash login bridge suggestion when local shell startup needs one.",
    ),
) -> None:
    if preflight == SmokePreflightMode.NEVER:
        raise typer.BadParameter(
            "`check-local` always runs the local doctor preflight; use `run` or `smoke` with `--preflight never` to skip it.",
            param_hint="--preflight",
        )
    _ = show_preflight
    selected_path = path or default_smoke_pipeline_path()
    report, pipeline, _loaded_pipeline = _doctor_report_for_path(selected_path)
    pipeline_context = _check_local_pipeline_context(pipeline)
    include_shell_bridge, recommendation = _doctor_shell_bridge_output(
        report,
        requested=shell_bridge,
        pipeline=_loaded_pipeline,
    )
    doctor_output = _structured_output_from_run_output(output)
    _echo_doctor_report(
        report,
        output=doctor_output,
        err=True,
        include_shell_bridge=include_shell_bridge,
        shell_bridge=recommendation,
        pipeline=pipeline_context,
    )
    if report.status == "failed":
        raise typer.Exit(code=1)
    _run_pipeline(_loaded_pipeline if _loaded_pipeline is not None else _load_pipeline(selected_path), runs_dir, max_concurrent_runs, output)


@app.command("toolchain-local")
def toolchain_local(
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
) -> None:
    report = build_local_kimi_toolchain_report()
    _echo_local_toolchain_report(report, output=output)
    raise typer.Exit(code=0 if report.status == "ok" else 1)


@app.command()
def doctor(
    path: str = typer.Argument(
        "",
        help="Optional pipeline path. Adds pipeline-specific local shell bootstrap warnings to the doctor report.",
    ),
    output: StructuredOutputFormat = typer.Option(
        StructuredOutputFormat.AUTO,
        "--output",
        help="Result output format. Defaults to `summary` on a terminal and `json` otherwise.",
    ),
    shell_bridge: bool = typer.Option(
        False,
        "--shell-bridge",
        help="Include a ready-to-paste bash login bridge suggestion when local shell startup needs one.",
    ),
) -> None:
    report, pipeline, _loaded_pipeline = _doctor_report_for_path(path or None)
    include_shell_bridge, recommendation = _doctor_shell_bridge_output(
        report,
        requested=shell_bridge,
        pipeline=_loaded_pipeline,
    )
    _echo_doctor_report(
        report,
        output=output,
        include_shell_bridge=include_shell_bridge,
        shell_bridge=recommendation,
        pipeline=pipeline,
    )
    raise typer.Exit(code=0 if report.status != "failed" else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
