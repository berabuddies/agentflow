from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentflow.agents.registry import default_adapter_registry
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.specs import (
    AgentKind,
    LocalTarget,
    NodeSpec,
    builtin_agent_kind,
    normalize_agent_name,
)
from agentflow.traces import create_trace_parser
from agentflow.utils import ensure_dir, json_dumps, redact_sensitive_shell_text, utcnow_iso


_TUNER_CONFIG_DIR = "agent_tuner"
_TUNED_AGENTS_ROOT = Path(".agentflow") / "tuned_agents"
_REGISTRY_PATH = _TUNED_AGENTS_ROOT / "registry.json"
_INTERACTIVE_AGENTS = {AgentKind.CODEX, AgentKind.CLAUDE, AgentKind.KIMI}


class TunableSurface(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    notes: str | None = None
    paths: list[str]

    @field_validator("name", "notes")
    @classmethod
    def _validate_surface_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("paths")
    @classmethod
    def _validate_surface_paths(cls, value: list[str]) -> list[str]:
        normalized_paths = []
        for raw_path in value:
            normalized = raw_path.strip()
            if not normalized:
                raise ValueError("path must not be empty")
            normalized_paths.append(normalized)
        if not normalized_paths:
            raise ValueError("paths must not be empty")
        return normalized_paths


class TunerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    base_agent: AgentKind
    repo_url: str
    default_branch: str = "main"
    workdir_subpath: str | None = None
    build_command: str
    test_command: str
    smoke_command: str
    executable_path: str | None = None
    evolution_prompt: str
    tunable_surfaces: list[TunableSurface] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    max_attempts: int = Field(default=3, ge=1)

    @field_validator("name", "repo_url", "default_branch", "build_command", "test_command", "smoke_command", "evolution_prompt")
    @classmethod
    def _validate_nonempty_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("workdir_subpath", "executable_path")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ResolvedTunerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    agent_name: str
    path: str
    config: TunerConfig


class TunedAgentVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    profile: str
    agent_name: str
    base_agent: AgentKind
    status: Literal["ready", "failed"] = "ready"
    created_at: str = Field(default_factory=utcnow_iso)
    source_run_id: str | None = None
    source_nodes: list[str] = Field(default_factory=list)
    repo_path: str
    workdir: str
    executable: str
    env: dict[str, str] = Field(default_factory=dict)
    summary: str | None = None


class TunedAgentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_agent: AgentKind
    latest_version: str | None = None
    versions: list[TunedAgentVersion] = Field(default_factory=list)


class TunedAgentRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: dict[str, TunedAgentRecord] = Field(default_factory=dict)


class EvolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    target: str
    optimizer: str
    source_nodes: list[str]
    trace_paths: dict[str, str]
    workspace_dir: str | None = None
    run_id: str | None = None

    @field_validator("profile", "target", "optimizer")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


@dataclass(slots=True)
class PreparedAgentResolution:
    node: NodeSpec
    runtime_agent: AgentKind
    version: TunedAgentVersion | None = None


@dataclass(slots=True)
class CommandExecution:
    command: str
    exit_code: int
    stdout: str
    stderr: str


def tuned_agents_root(workspace: Path) -> Path:
    return workspace / _TUNED_AGENTS_ROOT


def tuned_agent_registry_path(workspace: Path) -> Path:
    return tuned_agents_root(workspace) / "registry.json"


def tuned_agent_versions_dir(workspace: Path, agent_name: str) -> Path:
    return tuned_agents_root(workspace) / agent_name / "versions"


def tuned_agent_version_dir(workspace: Path, agent_name: str, version_id: str) -> Path:
    return tuned_agent_versions_dir(workspace, agent_name) / version_id


def tuned_agent_version_metadata_path(workspace: Path, agent_name: str, version_id: str) -> Path:
    return tuned_agent_version_dir(workspace, agent_name, version_id) / "version.json"


def _profile_alias(profile: str, config: TunerConfig) -> str:
    if config.name:
        alias = config.name.strip()
    else:
        alias = f"{profile}_tuned" if builtin_agent_kind(profile) is not None else profile
    if builtin_agent_kind(alias) is not None:
        raise ValueError(
            f"tuner profile `{profile}` resolves to built-in agent name `{alias}`; "
            "set `name` to a non-built-in alias such as `codex_tuned`"
        )
    return alias


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _tuner_config_path(workspace: Path, profile: str) -> Path:
    config_dir = workspace / _TUNER_CONFIG_DIR
    candidates = [
        config_dir / f"{profile}.yaml",
        config_dir / f"{profile}.yml",
        config_dir / f"{profile}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing tuner config for profile `{profile}` under `{config_dir}`")


def load_tuner_config(workspace: Path, profile: str) -> ResolvedTunerConfig:
    path = _tuner_config_path(workspace, profile)
    payload = _load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"tuner config `{path}` must define an object")
    config = TunerConfig.model_validate(payload)
    return ResolvedTunerConfig(
        profile=profile,
        agent_name=_profile_alias(profile, config),
        path=str(path),
        config=config,
    )


def load_tuned_agent_registry(workspace: Path) -> TunedAgentRegistry:
    path = tuned_agent_registry_path(workspace)
    if not path.exists():
        return TunedAgentRegistry()
    return TunedAgentRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def save_tuned_agent_registry(workspace: Path, registry: TunedAgentRegistry) -> None:
    ensure_dir(tuned_agents_root(workspace))
    tuned_agent_registry_path(workspace).write_text(registry.model_dump_json(indent=2), encoding="utf-8")


def register_tuned_agent_version(workspace: Path, version: TunedAgentVersion) -> TunedAgentVersion:
    registry = load_tuned_agent_registry(workspace)
    record = registry.agents.get(version.agent_name)
    if record is None:
        record = TunedAgentRecord(name=version.agent_name, base_agent=version.base_agent)
        registry.agents[version.agent_name] = record
    record.base_agent = version.base_agent
    record.latest_version = version.id
    record.versions = [existing for existing in record.versions if existing.id != version.id]
    record.versions.append(version)
    save_tuned_agent_registry(workspace, registry)
    version_path = tuned_agent_version_metadata_path(workspace, version.agent_name, version.id)
    ensure_dir(version_path.parent)
    version_path.write_text(version.model_dump_json(indent=2), encoding="utf-8")
    latest_path = tuned_agents_root(workspace) / version.agent_name / "latest.json"
    latest_path.write_text(version.model_dump_json(indent=2), encoding="utf-8")
    return version


def resolve_tuned_agent_version(workspace: Path, agent_name: str) -> TunedAgentVersion | None:
    registry = load_tuned_agent_registry(workspace)
    record = registry.agents.get(agent_name)
    if record is None or not record.latest_version:
        return None
    for version in record.versions:
        if version.id == record.latest_version:
            return version
    return None


def list_tuned_agent_records(workspace: Path) -> list[TunedAgentRecord]:
    registry = load_tuned_agent_registry(workspace)
    return sorted(registry.agents.values(), key=lambda record: record.name)


def resolve_node_for_execution(node: NodeSpec, workspace: Path) -> PreparedAgentResolution:
    resolved_agent = builtin_agent_kind(node.agent)
    if resolved_agent is not None:
        return PreparedAgentResolution(node=node, runtime_agent=resolved_agent)

    version = resolve_tuned_agent_version(workspace, normalize_agent_name(node.agent))
    if version is None:
        raise KeyError(f"unknown tuned agent `{node.agent}`")
    if node.target.kind != "local":
        raise ValueError(
            f"tuned agent `{node.agent}` currently requires a local target; got `{node.target.kind}`"
        )

    target = node.target
    resolved_target = target.model_copy(update={"cwd": target.cwd or version.workdir})
    resolved_node = node.model_copy(
        update={
            "agent": version.base_agent,
            "env": {**version.env, **dict(node.env or {})},
            "executable": node.executable or version.executable,
            "target": resolved_target,
        }
    )
    return PreparedAgentResolution(node=resolved_node, runtime_agent=version.base_agent, version=version)


def _app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _execution_paths(cwd: Path, runtime_dir: Path) -> ExecutionPaths:
    cwd = cwd.resolve()
    runtime_dir = runtime_dir.resolve()
    ensure_dir(runtime_dir)
    return ExecutionPaths(
        host_workdir=cwd,
        host_runtime_dir=runtime_dir,
        target_workdir=str(cwd),
        target_runtime_dir=str(runtime_dir),
        app_root=_app_root(),
    )


def _materialize_runtime_files(prepared: PreparedExecution, runtime_dir: Path) -> None:
    for relative_path, content in prepared.runtime_files.items():
        target = runtime_dir / relative_path
        ensure_dir(target.parent)
        target.write_text(content, encoding="utf-8")
    for relative_path, source in prepared.runtime_symlinks.items():
        target = runtime_dir / relative_path
        ensure_dir(target.parent)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)


def _run_prepared(prepared: PreparedExecution) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(prepared.env)
    run_kwargs: dict[str, object] = {
        "cwd": prepared.cwd,
        "env": env,
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if prepared.stdin is None:
        run_kwargs["stdin"] = subprocess.DEVNULL
    else:
        run_kwargs["stdin"] = subprocess.PIPE
        run_kwargs["input"] = prepared.stdin
    return subprocess.run(prepared.command, **run_kwargs)


def _parse_agent_output(agent: AgentKind, node_id: str, stdout: str) -> str:
    parser = create_trace_parser(agent, node_id)
    for line in stdout.splitlines():
        parser.feed(line)
    return parser.finalize() or stdout.strip()


def _write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json_dumps(payload))


def _run_optimizer(
    optimizer: AgentKind,
    *,
    prompt: str,
    repo_dir: Path,
    runtime_dir: Path,
    env: dict[str, str],
) -> CommandExecution:
    normalized_env = dict(env)
    ambient_openai_base_url = (
        normalized_env.get("OPENAI_BASE_URL")
        or normalized_env.get("AGENTFLOW_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("AGENTFLOW_OPENAI_BASE_URL")
    )
    provider: dict[str, str] | None = None
    if optimizer == AgentKind.CODEX and ambient_openai_base_url:
        normalized_env.setdefault("OPENAI_BASE_URL", ambient_openai_base_url)
        provider = {
            "name": "openai-custom",
            "base_url": ambient_openai_base_url,
            "api_key_env": "OPENAI_API_KEY",
            "wire_api": "responses",
        }
    node = NodeSpec.model_validate(
        {
            "id": "optimizer",
            "agent": optimizer.value,
            "prompt": prompt,
            "tools": "read_write",
            "provider": provider,
            "repo_instructions_mode": "ignore",
            "target": {"kind": "local", "cwd": str(repo_dir)},
            "env": normalized_env,
        }
    )
    adapter = default_adapter_registry.get(optimizer)
    paths = _execution_paths(repo_dir, runtime_dir)
    prepared = adapter.prepare(node, prompt, paths)
    _materialize_runtime_files(prepared, runtime_dir)
    completed = _run_prepared(prepared)
    return CommandExecution(
        command=" ".join(prepared.command),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _render_command(template: str, *, repo_dir: Path, version_dir: Path, traces_dir: Path, executable: str) -> str:
    return template.format(
        repo=str(repo_dir),
        repo_dir=str(repo_dir),
        version_dir=str(version_dir),
        traces_dir=str(traces_dir),
        executable=executable,
    )


def _run_shell_command(
    command_template: str,
    *,
    repo_dir: Path,
    version_dir: Path,
    traces_dir: Path,
    executable: str,
    env: dict[str, str],
) -> CommandExecution:
    command = _render_command(
        command_template,
        repo_dir=repo_dir,
        version_dir=version_dir,
        traces_dir=traces_dir,
        executable=executable,
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=str(repo_dir),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandExecution(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _clone_repo(config: TunerConfig, repo_dir: Path) -> None:
    ensure_dir(repo_dir.parent)
    completed = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            config.default_branch,
            config.repo_url,
            str(repo_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to clone `{config.repo_url}`:\n{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _copy_trace_files(request: EvolutionRequest, traces_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    ensure_dir(traces_dir)
    for node_id in request.source_nodes:
        source = request.trace_paths.get(node_id)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"missing trace path for source node `{node_id}`")
        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"trace for source node `{node_id}` not found at `{source_path}`")
        target_path = traces_dir / f"{node_id}.trace.jsonl"
        shutil.copy2(source_path, target_path)
        copied[node_id] = str(target_path)
    return copied


def _default_executable_path(repo_dir: Path, base_agent: AgentKind) -> str:
    return str((repo_dir / ".venv" / "bin" / base_agent.value).resolve())


def _resolved_executable_path(config: TunerConfig, repo_dir: Path) -> str:
    raw_path = config.executable_path or _default_executable_path(repo_dir, config.base_agent)
    executable_path = Path(raw_path)
    if not executable_path.is_absolute():
        executable_path = (repo_dir / executable_path).resolve()
    return str(executable_path)


def _optimizer_prompt(
    resolved_config: ResolvedTunerConfig,
    *,
    repo_root: Path,
    repo_workdir: Path,
    traces_dir: Path,
    source_nodes: list[str],
    previous_failure: str | None,
) -> str:
    tunable_surfaces_section = _render_tunable_surfaces(resolved_config.config)
    workdir_section = ""
    if repo_workdir != repo_root:
        workdir_section = f"Build/test workdir: {repo_workdir}\n"
    failure_section = ""
    if previous_failure:
        failure_section = (
            "\nPrevious build/test/smoke failure to fix before finishing:\n"
            f"{previous_failure}\n"
        )
    return (
        f"You are evolving the `{resolved_config.agent_name}` agent profile.\n"
        f"Base agent type: {resolved_config.config.base_agent.value}\n"
        f"Clone root: {repo_root}\n"
        f"{workdir_section}"
        f"Copied trace directory: {traces_dir}\n"
        f"Source trace nodes: {', '.join(source_nodes)}\n"
        f"Tuner config: {resolved_config.path}\n\n"
        "Apply the following evolution brief to the cloned source tree only:\n"
        f"{resolved_config.config.evolution_prompt}\n"
        f"{tunable_surfaces_section}"
        f"{failure_section}\n"
        "Requirements:\n"
        "- Review carefully for regressions and obvious edge cases before finishing.\n"
        "- Use the copied traces as primary context for what to improve.\n"
        "- Treat this as a direct implementation task, not a design exercise.\n"
        "- Do not write design docs, implementation plans, or other planning artifacts.\n"
        "- Do not wait for user confirmation, brainstorming approval, or review checkpoints before editing.\n"
        "- Ignore installed process skills such as brainstorming, writing-plans, systematic-debugging, and test-driven-development; the outer harness is already handling planning, debugging discipline, and verification.\n"
        "- Do not survey every tunable surface. Start from the trace evidence and inspect only the files needed for the smallest coherent fix.\n"
        "- If the evolution brief calls for it, you may change system prompts, developer instructions, prompt templates, tool definitions, tool registration, tool descriptions, and related agent-behavior configuration.\n"
        "- Modify only files under the cloned repo.\n"
        "- You may run lightweight inspections or focused tests if they help validate the change, but do not run the configured build or smoke commands yourself; the outer harness will run build, test, and smoke after you finish editing.\n"
        "- Leave the repo in a working state for the next build/test/smoke pass.\n"
    )


def _render_tunable_surfaces(config: TunerConfig) -> str:
    if not config.tunable_surfaces:
        return ""

    lines = [
        "Known tunable surfaces and implementing files",
        "(all paths are relative to the clone root above):",
    ]
    for surface in config.tunable_surfaces:
        lines.append(f"- {surface.name}")
        if surface.notes:
            lines.append(f"  Notes: {surface.notes}")
        for path in surface.paths:
            lines.append(f"  - {path}")
    lines.append("")
    return "\n".join(lines)


def _attempt_summary(label: str, execution: CommandExecution) -> str:
    pieces = [f"{label} failed with exit code {execution.exit_code}."]
    stdout = execution.stdout.strip()
    stderr = execution.stderr.strip()
    if stdout:
        pieces.append(f"stdout:\n{stdout}")
    if stderr:
        pieces.append(f"stderr:\n{stderr}")
    return "\n\n".join(pieces)


def _write_attempt_artifact(attempt_dir: Path, name: str, execution: CommandExecution) -> None:
    _write_json(
        attempt_dir / f"{name}.json",
        {
            "command": redact_sensitive_shell_text(execution.command),
            "exit_code": execution.exit_code,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        },
    )


def _write_failure_metadata(
    version_dir: Path,
    *,
    agent_name: str,
    base_agent: AgentKind,
    profile: str,
    repo_dir: Path,
    resolved_executable: str,
    env: dict[str, str],
    source_run_id: str | None,
    source_nodes: list[str],
    summary: str,
) -> None:
    failed_version = TunedAgentVersion(
        id=version_dir.name,
        profile=profile,
        agent_name=agent_name,
        base_agent=base_agent,
        status="failed",
        source_run_id=source_run_id,
        source_nodes=source_nodes,
        repo_path=str(repo_dir),
        workdir=str(repo_dir),
        executable=resolved_executable,
        env=env,
        summary=summary,
    )
    _write_json(version_dir / "version.json", failed_version.model_dump(mode="json"))


def run_evolution_from_payload(
    payload: dict[str, Any],
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    request = EvolutionRequest.model_validate(payload)
    workspace = Path(request.workspace_dir or os.getcwd()).expanduser().resolve()
    resolved_config = load_tuner_config(workspace, request.profile)
    target_kind = builtin_agent_kind(request.target)
    optimizer_kind = builtin_agent_kind(request.optimizer)
    if target_kind is None or target_kind not in _INTERACTIVE_AGENTS:
        raise ValueError("evolution target must be one of: codex, claude, kimi")
    if optimizer_kind is None or optimizer_kind not in _INTERACTIVE_AGENTS:
        raise ValueError("optimizer must be one of: codex, claude, kimi")
    if resolved_config.config.base_agent != target_kind:
        raise ValueError(
            f"tuner profile `{request.profile}` targets `{resolved_config.config.base_agent.value}`, "
            f"but evolve() requested target `{target_kind.value}`"
        )
    if not request.source_nodes:
        raise ValueError("evolution requires at least one source node")

    def _emit_progress(
        stage: str,
        *,
        attempt: int,
        status: str | None = None,
        command: str | None = None,
        detail: str | None = None,
    ) -> None:
        if progress is None:
            return
        payload: dict[str, object] = {
            "agentflow_event": "evolution_progress",
            "stage": stage,
            "attempt": attempt,
        }
        if status is not None:
            payload["status"] = status
        if command is not None:
            payload["command"] = command
        if detail is not None:
            payload["detail"] = detail
        progress(payload)

    version_id = uuid4().hex[:12]
    version_dir = tuned_agent_version_dir(workspace, resolved_config.agent_name, version_id)
    traces_dir = version_dir / "traces"
    repo_dir = version_dir / "repo"
    attempt_root = version_dir / "attempts"
    ensure_dir(attempt_root)
    _clone_repo(resolved_config.config, repo_dir)
    copied_traces = _copy_trace_files(request, traces_dir)

    repo_workdir = repo_dir
    if resolved_config.config.workdir_subpath:
        repo_workdir = (repo_dir / resolved_config.config.workdir_subpath).resolve()
    if not repo_workdir.exists():
        raise FileNotFoundError(
            f"configured workdir_subpath `{resolved_config.config.workdir_subpath}` does not exist in `{repo_dir}`"
        )

    env = dict(resolved_config.config.env)
    resolved_executable = _resolved_executable_path(resolved_config.config, repo_workdir)
    failure_summary: str | None = None

    _emit_progress("start", attempt=1)

    for attempt_number in range(1, resolved_config.config.max_attempts + 1):
        attempt_dir = ensure_dir(attempt_root / f"attempt-{attempt_number}")
        _emit_progress("attempt", attempt=attempt_number, status="started")
        prompt = _optimizer_prompt(
            resolved_config,
            repo_root=repo_dir,
            repo_workdir=repo_workdir,
            traces_dir=traces_dir,
            source_nodes=request.source_nodes,
            previous_failure=failure_summary,
        )
        _write_text(attempt_dir / "optimizer-prompt.txt", prompt)

        _emit_progress("optimizer", attempt=attempt_number, status="started", command="optimizer")
        optimizer_result = _run_optimizer(
            optimizer_kind,
            prompt=prompt,
            repo_dir=repo_dir,
            runtime_dir=attempt_dir / "optimizer-runtime",
            env=env,
        )
        _write_attempt_artifact(attempt_dir, "optimizer", optimizer_result)
        if optimizer_result.exit_code != 0:
            failure_summary = _attempt_summary("Optimizer", optimizer_result)
            _emit_progress("optimizer", attempt=attempt_number, status="failed", detail=failure_summary)
            continue
        _emit_progress("optimizer", attempt=attempt_number, status="completed")

        _emit_progress(
            "build",
            attempt=attempt_number,
            status="started",
            command=resolved_config.config.build_command,
        )
        build_result = _run_shell_command(
            resolved_config.config.build_command,
            repo_dir=repo_workdir,
            version_dir=version_dir,
            traces_dir=traces_dir,
            executable=resolved_executable,
            env=env,
        )
        _write_attempt_artifact(attempt_dir, "build", build_result)
        if build_result.exit_code != 0:
            failure_summary = _attempt_summary("Build", build_result)
            _emit_progress("build", attempt=attempt_number, status="failed", detail=failure_summary)
            continue
        _emit_progress("build", attempt=attempt_number, status="completed")

        _emit_progress(
            "test",
            attempt=attempt_number,
            status="started",
            command=resolved_config.config.test_command,
        )
        test_result = _run_shell_command(
            resolved_config.config.test_command,
            repo_dir=repo_workdir,
            version_dir=version_dir,
            traces_dir=traces_dir,
            executable=resolved_executable,
            env=env,
        )
        _write_attempt_artifact(attempt_dir, "test", test_result)
        if test_result.exit_code != 0:
            failure_summary = _attempt_summary("Test", test_result)
            _emit_progress("test", attempt=attempt_number, status="failed", detail=failure_summary)
            continue
        _emit_progress("test", attempt=attempt_number, status="completed")

        _emit_progress(
            "smoke",
            attempt=attempt_number,
            status="started",
            command=resolved_config.config.smoke_command,
        )
        smoke_result = _run_shell_command(
            resolved_config.config.smoke_command,
            repo_dir=repo_workdir,
            version_dir=version_dir,
            traces_dir=traces_dir,
            executable=resolved_executable,
            env=env,
        )
        _write_attempt_artifact(attempt_dir, "smoke", smoke_result)
        if smoke_result.exit_code != 0:
            failure_summary = _attempt_summary("Smoke", smoke_result)
            _emit_progress("smoke", attempt=attempt_number, status="failed", detail=failure_summary)
            continue
        _emit_progress("smoke", attempt=attempt_number, status="completed")

        executable_path = Path(resolved_executable)
        if not executable_path.exists():
            detail = (
                f"successful evolution did not produce executable `{executable_path}`; "
                "set `executable_path` in the tuner config or make the build produce the default path"
            )
            _emit_progress("final", attempt=attempt_number, status="failed", detail=detail)
            raise FileNotFoundError(
                detail
            )

        version = TunedAgentVersion(
            id=version_id,
            profile=request.profile,
            agent_name=resolved_config.agent_name,
            base_agent=resolved_config.config.base_agent,
            source_run_id=request.run_id,
            source_nodes=request.source_nodes,
            repo_path=str(repo_dir),
            workdir=str(repo_workdir),
            executable=str(executable_path),
            env=env,
            summary=_parse_agent_output(optimizer_kind, f"optimizer_{version_id}", optimizer_result.stdout),
        )
        register_tuned_agent_version(workspace, version)
        _emit_progress("final", attempt=attempt_number, status="success")
        return {
            "ok": True,
            "agent_name": version.agent_name,
            "version": version.id,
            "base_agent": version.base_agent.value,
            "repo_path": version.repo_path,
            "workdir": version.workdir,
            "executable": version.executable,
            "traces": copied_traces,
        }

    _emit_progress(
        "final",
        attempt=resolved_config.config.max_attempts,
        status="failed",
        detail=failure_summary or "evolution failed without diagnostics",
    )
    _write_failure_metadata(
        version_dir,
        agent_name=resolved_config.agent_name,
        base_agent=resolved_config.config.base_agent,
        profile=request.profile,
        repo_dir=repo_dir,
        resolved_executable=resolved_executable,
        env=env,
        source_run_id=request.run_id,
        source_nodes=request.source_nodes,
        summary=failure_summary or "evolution failed without diagnostics",
    )
    raise RuntimeError(failure_summary or "evolution failed without diagnostics")
