from __future__ import annotations

import asyncio
import json
from pathlib import Path

import agentflow.cli
from agentflow import Graph, agent, codex, evolve
from agentflow.agents.base import AgentAdapter
from agentflow.agents.registry import AdapterRegistry
from agentflow.inspection import build_launch_inspection
from agentflow.orchestrator import Orchestrator
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.runners.registry import RunnerRegistry
from agentflow.specs import AgentKind, NodeSpec, PipelineSpec, RunRecord, RunStatus
from agentflow.store import RunStore
from agentflow.tuned_agents import (
    CommandExecution,
    TunableSurface,
    TunedAgentVersion,
    list_tuned_agent_records,
    load_tuner_config,
    load_tuned_agent_registry,
    _optimizer_prompt,
    register_tuned_agent_version,
    ResolvedTunerConfig,
    TunerConfig,
    resolve_node_for_execution,
    run_evolution_from_payload,
)


class SimpleCodexAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        script = (
            'import json, sys\n'
            'prompt = sys.argv[1]\n'
            'print(json.dumps({"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":prompt}]}}))\n'
        )
        return PreparedExecution(
            command=["python3", "-c", script, prompt],
            env={},
            cwd=paths.target_workdir,
            trace_kind="codex",
        )


def test_loader_supports_yaml_pipeline(tmp_path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        "name: yaml-pipeline\nworking_dir: .\nnodes:\n  - id: plan\n    agent: codex\n    prompt: hi\n",
        encoding="utf-8",
    )

    from agentflow.loader import load_pipeline_from_path

    pipeline = load_pipeline_from_path(pipeline_path)

    assert pipeline.name == "yaml-pipeline"
    assert pipeline.node_map["plan"].agent == AgentKind.CODEX


def test_agent_helper_supports_custom_agent_names():
    with Graph("custom-agent") as graph:
        custom = agent("my_codex", task_id="plan", prompt="hi")

    payload = graph.to_payload()
    assert payload["nodes"][0]["agent"] == "my_codex"
    assert repr(custom) == 'NodeBuilder(id="plan", agent="my_codex")'


def test_evolve_helper_filters_source_nodes_and_builds_payload():
    with Graph("evolve-graph") as graph:
        plan = codex(task_id="plan", prompt="hi")
        review = agent("review_bot", task_id="review", prompt="hello")
        tuner = evolve([plan, review], target="codex", optimizer="codex")

    payload = graph.to_payload()
    evolve_node = next(node for node in payload["nodes"] if node["id"] == tuner.id)

    assert evolve_node["agent"] == "python"
    assert evolve_node["depends_on"] == ["plan"]
    assert '"source_nodes": ["plan"]' in evolve_node["prompt"]
    assert "{{ nodes.plan.artifacts.trace_jsonl }}" in evolve_node["prompt"]
    assert "sys.path.insert(0," in evolve_node["prompt"]
    assert "progress=_evolution_progress" in evolve_node["prompt"]


def test_resolve_node_for_execution_uses_latest_registry_entry(tmp_path):
    workspace = tmp_path
    version = TunedAgentVersion(
        id="v1",
        profile="codex",
        agent_name="codex_tuned",
        base_agent=AgentKind.CODEX,
        repo_path=str(workspace / "repo"),
        workdir=str(workspace / "repo" / "src"),
        executable=str(workspace / "repo" / ".venv" / "bin" / "codex"),
        env={"FROM_REGISTRY": "1"},
    )
    register_tuned_agent_version(workspace, version)

    node = NodeSpec.model_validate(
        {
            "id": "custom",
            "agent": "codex_tuned",
            "prompt": "hi",
            "target": {"kind": "local"},
            "env": {"FROM_NODE": "1"},
        }
    )

    resolved = resolve_node_for_execution(node, workspace)

    assert resolved.runtime_agent == AgentKind.CODEX
    assert resolved.node.executable == version.executable
    assert resolved.node.target.cwd == version.workdir
    assert resolved.node.env == {"FROM_REGISTRY": "1", "FROM_NODE": "1"}


def test_build_launch_inspection_plans_tuned_agent_executable(tmp_path):
    workspace = tmp_path
    version = TunedAgentVersion(
        id="v1",
        profile="codex",
        agent_name="codex_tuned",
        base_agent=AgentKind.CODEX,
        repo_path=str(workspace / "repo"),
        workdir=str(workspace / "repo"),
        executable=str(workspace / "repo" / ".venv" / "bin" / "codex"),
        env={"FROM_REGISTRY": "1"},
    )
    register_tuned_agent_version(workspace, version)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "inspect-tuned",
            "working_dir": str(workspace),
            "nodes": [{"id": "plan", "agent": "codex_tuned", "prompt": "hi"}],
        }
    )

    report = build_launch_inspection(pipeline, runs_dir=str(workspace / "runs"))

    node_plan = report["nodes"][0]
    assert node_plan["agent"] == "codex_tuned"
    assert node_plan["runtime_agent"] == "codex"
    assert node_plan["prepared"]["command"][0] == version.executable
    assert node_plan["target"]["cwd"] == version.workdir


def test_orchestrator_executes_tuned_agent_from_registry(tmp_path):
    workspace = tmp_path
    version = TunedAgentVersion(
        id="v1",
        profile="codex",
        agent_name="codex_tuned",
        base_agent=AgentKind.CODEX,
        repo_path=str(workspace / "repo"),
        workdir=str(workspace / "repo"),
        executable=str(workspace / "repo" / ".venv" / "bin" / "codex"),
        env={},
    )
    register_tuned_agent_version(workspace, version)

    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, SimpleCodexAdapter())
    orchestrator = Orchestrator(store=RunStore(workspace / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "run-tuned",
            "working_dir": str(workspace),
            "nodes": [{"id": "plan", "agent": "codex_tuned", "prompt": "hello tuned"}],
        }
    )

    run = asyncio.run(orchestrator.submit(pipeline))
    completed = asyncio.run(orchestrator.wait(run.id, timeout=5))

    assert completed.status == RunStatus.COMPLETED
    assert completed.nodes["plan"].output == "hello tuned"


def test_run_evolution_from_payload_retries_and_registers_latest(tmp_path, monkeypatch):
    workspace = tmp_path
    config_dir = workspace / "agent_tuner"
    config_dir.mkdir()
    (config_dir / "codex.yaml").write_text(
        "\n".join(
            [
                "name: codex_tuned",
                "base_agent: codex",
                "repo_url: https://example.invalid/repo.git",
                "build_command: build",
                "test_command: test",
                "smoke_command: smoke",
                "evolution_prompt: improve the agent",
                "executable_path: .venv/bin/codex",
                "max_attempts: 3",
            ]
        ),
        encoding="utf-8",
    )
    trace_path = workspace / "trace.jsonl"
    trace_path.write_text('{"kind":"assistant_message","content":"hello"}\n', encoding="utf-8")

    def fake_clone(_config, repo_dir: Path) -> None:
        repo_dir.mkdir(parents=True)
        (repo_dir / "README.md").write_text("base", encoding="utf-8")

    attempt_state = {"smoke": 0}

    def fake_optimizer(_optimizer: AgentKind, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "README.md").write_text(prompt, encoding="utf-8")
        return CommandExecution(command="optimizer", exit_code=0, stdout='{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"updated"}]}}', stderr="")

    def fake_shell(command_template: str, *, repo_dir: Path, version_dir: Path, traces_dir: Path, executable: str, env: dict[str, str]):
        if command_template == "build":
            executable_path = Path(executable)
            executable_path.parent.mkdir(parents=True, exist_ok=True)
            executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        if command_template == "smoke":
            attempt_state["smoke"] += 1
            if attempt_state["smoke"] == 1:
                return CommandExecution(command="smoke", exit_code=1, stdout="", stderr="ping failed")
        return CommandExecution(command=command_template, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr("agentflow.tuned_agents._clone_repo", fake_clone)
    monkeypatch.setattr("agentflow.tuned_agents._run_optimizer", fake_optimizer)
    monkeypatch.setattr("agentflow.tuned_agents._run_shell_command", fake_shell)

    result = run_evolution_from_payload(
        {
            "profile": "codex",
            "target": "codex",
            "optimizer": "codex",
            "source_nodes": ["plan"],
            "trace_paths": {"plan": str(trace_path)},
            "workspace_dir": str(workspace),
            "run_id": "run123",
        }
    )

    assert result["ok"] is True
    assert result["agent_name"] == "codex_tuned"
    assert attempt_state["smoke"] == 2
    registry = load_tuned_agent_registry(workspace)
    assert registry.agents["codex_tuned"].latest_version == result["version"]
    assert Path(result["executable"]).exists()


def test_run_evolution_from_payload_reports_progress(tmp_path, monkeypatch):
    workspace = tmp_path
    config_dir = workspace / "agent_tuner"
    config_dir.mkdir()
    (config_dir / "codex.yaml").write_text(
        "\n".join(
            [
                "name: codex_tuned",
                "base_agent: codex",
                "repo_url: https://example.invalid/repo.git",
                "build_command: build",
                "test_command: test",
                "smoke_command: smoke",
                "evolution_prompt: improve the agent",
                "executable_path: .venv/bin/codex",
                "max_attempts: 2",
            ]
        ),
        encoding="utf-8",
    )
    trace_path = workspace / "trace.jsonl"
    trace_path.write_text('{"kind":"assistant_message","content":"hello"}\n', encoding="utf-8")

    def fake_clone(_config, repo_dir: Path) -> None:
        repo_dir.mkdir(parents=True)
        (repo_dir / "README.md").write_text("base", encoding="utf-8")

    attempt_state = {"smoke": 0}

    def fake_optimizer(_optimizer: AgentKind, *, prompt: str, repo_dir: Path, runtime_dir: Path, env: dict[str, str]):
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "README.md").write_text(prompt, encoding="utf-8")
        return CommandExecution(
            command="optimizer",
            exit_code=0,
            stdout='{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"updated"}]}}',
            stderr="",
        )

    def fake_shell(command_template: str, *, repo_dir: Path, version_dir: Path, traces_dir: Path, executable: str, env: dict[str, str]):
        if command_template == "build":
            executable_path = Path(executable)
            executable_path.parent.mkdir(parents=True, exist_ok=True)
            executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        if command_template == "smoke":
            attempt_state["smoke"] += 1
            if attempt_state["smoke"] == 1:
                return CommandExecution(command="smoke", exit_code=1, stdout="", stderr="ping failed")
        return CommandExecution(command=command_template, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr("agentflow.tuned_agents._clone_repo", fake_clone)
    monkeypatch.setattr("agentflow.tuned_agents._run_optimizer", fake_optimizer)
    monkeypatch.setattr("agentflow.tuned_agents._run_shell_command", fake_shell)

    progress: list[dict[str, object]] = []

    def capture(event: dict[str, object]) -> None:
        progress.append(event)

    result = run_evolution_from_payload(
        {
            "profile": "codex",
            "target": "codex",
            "optimizer": "codex",
            "source_nodes": ["plan"],
            "trace_paths": {"plan": str(trace_path)},
            "workspace_dir": str(workspace),
            "run_id": "run-progress",
        },
        progress=capture,
    )

    assert result["ok"] is True
    assert any(event.get("agentflow_event") == "evolution_progress" for event in progress)
    stages = [(event.get("stage"), event.get("status"), event.get("attempt")) for event in progress]
    assert stages[0][0] == "start"
    assert ("attempt", "started", 1) in stages
    assert ("optimizer", "started", 1) in stages
    assert ("optimizer", "completed", 1) in stages
    assert ("build", "started", 1) in stages
    assert ("build", "completed", 1) in stages
    assert ("test", "started", 1) in stages
    assert ("test", "completed", 1) in stages
    assert ("smoke", "started", 1) in stages
    assert ("smoke", "failed", 1) in stages
    assert ("attempt", "started", 2) in stages
    assert ("final", "success", 2) in stages
    smoke_failure = next(event for event in progress if event.get("stage") == "smoke" and event.get("status") == "failed")
    assert "Smoke" in str(smoke_failure.get("detail", ""))
    build_start = next(event for event in progress if event.get("stage") == "build" and event.get("status") == "started")
    assert build_start.get("command") == "build"


def test_optimizer_prompt_explicitly_allows_prompt_and_tool_edits(tmp_path):
    resolved = ResolvedTunerConfig(
        profile="codex",
        agent_name="codex_tuned",
        path=str(tmp_path / "agent_tuner" / "codex.yaml"),
        config=TunerConfig(
            name="codex_tuned",
            base_agent=AgentKind.CODEX,
            repo_url="https://github.com/openai/codex.git",
            build_command="build",
            test_command="test",
            smoke_command="smoke",
            evolution_prompt="Improve Codex.",
            tunable_surfaces=[
                TunableSurface(
                    name="Base prompts",
                    notes="Primary system prompts.",
                    paths=["core/gpt_5_codex_prompt.md", "tools/src/local_tool.rs"],
                )
            ],
        ),
    )

    prompt = _optimizer_prompt(
        resolved,
        repo_dir=tmp_path / "repo",
        traces_dir=tmp_path / "traces",
        source_nodes=["plan"],
        previous_failure=None,
    )

    assert "you may change system prompts" in prompt
    assert "tool definitions" in prompt
    assert "tool descriptions" in prompt
    assert "Known tunable surfaces and implementing files" in prompt
    assert "core/gpt_5_codex_prompt.md" in prompt
    assert "tools/src/local_tool.rs" in prompt


def test_repo_includes_codex_tuner_profile():
    workspace = Path(__file__).resolve().parents[1]

    resolved = load_tuner_config(workspace, "codex")

    assert resolved.agent_name == "codex_tuned"
    assert resolved.config.base_agent == AgentKind.CODEX
    assert resolved.config.repo_url == "https://github.com/openai/codex.git"
    assert resolved.config.workdir_subpath == "codex-rs"
    assert resolved.config.executable_path == "target/debug/codex"
    assert "System prompts" in resolved.config.evolution_prompt
    assert "tool descriptions" in resolved.config.evolution_prompt
    assert len(resolved.config.tunable_surfaces) >= 10
    assert resolved.config.tunable_surfaces[0].name == "Base model prompts and prompt assembly"
    assert "core/gpt_5_codex_prompt.md" in resolved.config.tunable_surfaces[0].paths


def test_cli_lists_tuned_agents(tmp_path, capsys):
    workspace = tmp_path
    register_tuned_agent_version(
        workspace,
        TunedAgentVersion(
            id="v1",
            profile="codex",
            agent_name="codex_tuned",
            base_agent=AgentKind.CODEX,
            repo_path=str(workspace / "repo"),
            workdir=str(workspace / "repo"),
            executable=str(workspace / "repo" / ".venv" / "bin" / "codex"),
            env={},
            ),
        )

    agentflow.cli.tuned_agents(
        workspace=str(workspace),
        output=agentflow.cli.StructuredOutputFormat.SUMMARY,
    )
    captured = capsys.readouterr()

    assert "codex_tuned [codex]" in captured.out


def test_cli_evolve_builds_payload_from_run(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    store = RunStore(runs_dir)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cli-evolve",
            "working_dir": str(tmp_path),
            "nodes": [
                {"id": "plan", "agent": "codex", "prompt": "hi"},
                {"id": "review", "agent": "claude", "prompt": "hi"},
            ],
        }
    )
    record = RunRecord(
        id="run123",
        status=RunStatus.COMPLETED,
        pipeline=pipeline,
        nodes={node.id: {"node_id": node.id} for node in pipeline.nodes},
    )
    asyncio.run(store.create_run(record))
    trace_path = store.artifact_path("run123", "plan", "trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("trace\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_evolution(payload: dict[str, object]) -> dict[str, object]:
        captured["payload"] = payload
        return {
            "ok": True,
            "agent_name": "codex_tuned",
            "version": "v1",
            "base_agent": "codex",
            "repo_path": str(tmp_path / "repo"),
            "executable": str(tmp_path / "repo" / ".venv" / "bin" / "codex"),
        }

    monkeypatch.setattr(agentflow.cli, "run_evolution_from_payload", fake_run_evolution)

    agentflow.cli.evolve(
        "run123",
        node=["plan", "review"],
        target="codex",
        optimizer="codex",
        profile="",
        runs_dir=str(runs_dir),
        output=agentflow.cli.StructuredOutputFormat.SUMMARY,
    )
    captured_out = capsys.readouterr().out

    assert "Agent: codex_tuned" in captured_out
    assert captured["payload"] == {
        "profile": "codex",
        "target": "codex",
        "optimizer": "codex",
        "source_nodes": ["plan"],
        "trace_paths": {"plan": str(trace_path)},
        "workspace_dir": str(tmp_path),
        "run_id": "run123",
    }


def test_cli_tuned_agent_detail_json(tmp_path, capsys):
    workspace = tmp_path
    version = TunedAgentVersion(
        id="v1",
        profile="codex",
        agent_name="codex_tuned",
        base_agent=AgentKind.CODEX,
        repo_path=str(workspace / "repo"),
        workdir=str(workspace / "repo"),
        executable=str(workspace / "repo" / ".venv" / "bin" / "codex"),
        env={},
    )
    register_tuned_agent_version(workspace, version)

    agentflow.cli.tuned_agent(
        "codex_tuned",
        workspace=str(workspace),
        output=agentflow.cli.StructuredOutputFormat.JSON,
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["name"] == "codex_tuned"
    assert payload["latest"]["id"] == "v1"
    assert len(list_tuned_agent_records(workspace)) == 1
