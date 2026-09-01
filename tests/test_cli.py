from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import agentflow.cli
import agentflow.inspection
import agentflow.local_shell
from agentflow.cli import app, _render_doctor_summary
from agentflow.doctor import (
    DoctorCheck,
    DoctorReport,
    ShellBridgeRecommendation,
    _CODEX_AUTH_VIA_API_KEY_AND_LOGIN_STATUS_EXIT_CODE,
    build_bash_login_shell_bridge_recommendation,
)
from agentflow.specs import ProviderConfig
from agentflow.inference import SkyInferenceLaunch, SkyInferenceService

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_ambient_base_url_env(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def _capture_pipeline_loader(captured: dict[str, object], fake_pipeline: object):
    def _load(path: str):
        captured["loaded_path"] = path
        return fake_pipeline

    return _load


def test_stream_supports_tty_summary_treats_click_testing_streams_as_terminal():
    probe_app = typer.Typer(add_completion=False)

    @probe_app.command()
    def probe() -> None:
        typer.echo(
            json.dumps(
                {
                    "stdout": agentflow.cli._stream_supports_tty_summary(err=False),
                    "stderr": agentflow.cli._stream_supports_tty_summary(err=True),
                }
            )
        )

    result = runner.invoke(probe_app, [])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"stdout": True, "stderr": True}


def _doctor_report(status: str = "ok", detail: str = "ready") -> DoctorReport:
    check_status = "failed" if status == "failed" else "ok"
    return DoctorReport(
        status=status,
        checks=[DoctorCheck(name="kimi_shell_helper", status=check_status, detail=detail)],
    )


def _custom_kimi_preflight_report(
    *,
    kimi_status: str = "ok",
    kimi_detail: str = "ready",
    bash_status: str = "ok",
    bash_detail: str = "startup ready",
) -> DoctorReport:
    statuses = {bash_status, kimi_status}
    if "failed" in statuses:
        report_status = "failed"
    elif "warning" in statuses:
        report_status = "warning"
    else:
        report_status = "ok"
    return DoctorReport(
        status=report_status,
        checks=[
            DoctorCheck(name="bash_login_startup", status=bash_status, detail=bash_detail),
            DoctorCheck(name="kimi_shell_helper", status=kimi_status, detail=kimi_detail),
        ],
    )


def _reject_bundled_smoke_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("bundled smoke doctor should not run for custom kimi preflight")),
    )


def _mock_custom_kimi_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kimi_status: str = "ok",
    kimi_detail: str = "ready",
    bash_status: str = "ok",
    bash_detail: str = "startup ready",
) -> None:
    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: _custom_kimi_preflight_report(
            kimi_status=kimi_status,
            kimi_detail=kimi_detail,
            bash_status=bash_status,
            bash_detail=bash_detail,
        ),
    )


def _completed_subprocess(returncode: int = 0, *, stdout: str = "", stderr: str = ""):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=returncode, stdout=stdout, stderr=stderr)

    return _run


def _codex_ready_and_auth_subprocess(auth_returncode: int):
    def _run(*args, **kwargs):
        env = kwargs.get("env") or {}
        target_command = str(env.get("AGENTFLOW_TARGET_COMMAND", ""))
        if "codex --version" in target_command:
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")
        if "OPENAI_API_KEY" in target_command and "subprocess.run" in target_command:
            return subprocess.CompletedProcess(args=args[0], returncode=auth_returncode, stdout="", stderr="")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    return _run


def _shell_bridge_recommendation() -> ShellBridgeRecommendation:
    return ShellBridgeRecommendation(
        target="~/.bash_profile",
        source="~/.profile",
        snippet='if [ -f "$HOME/.profile" ]; then\n  . "$HOME/.profile"\nfi\n',
        reason="Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.",
    )


def _bash_startup_context(
    summary: str,
    *,
    startup_files: dict[str, str] | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {"startup_summary": summary}
    if startup_files is not None:
        context["startup_files"] = startup_files
        context["startup_files_summary"] = ", ".join(f"{path}={status}" for path, status in startup_files.items())
    return context




def _disable_local_readiness_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_info_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_info_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_info_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_info_checks", lambda pipeline: [])


def _mock_local_readiness_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_info_checks", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_claude_readiness_info_checks",
        lambda pipeline: [
            DoctorCheck(
                name="claude_ready",
                status="ok",
                detail=(
                    "Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; "
                    "`claude --version` succeeds in the prepared local shell."
                ),
            )
        ],
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_codex_readiness_info_checks",
        lambda pipeline: [
            DoctorCheck(
                name="codex_ready",
                status="ok",
                detail=(
                    "Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; "
                    "`codex --version` succeeds in the prepared local shell."
                ),
            )
        ],
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_codex_auth_info_checks",
        lambda pipeline: [
            DoctorCheck(
                name="codex_auth",
                status="ok",
                detail=(
                    "Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via "
                    "`codex login status` or `OPENAI_API_KEY`."
                ),
            )
        ],
    )


def _bundled_kimi_smoke_pipeline(*, trigger: str = "target.bootstrap") -> SimpleNamespace:
    target_kwargs = {"kind": "local"}
    if trigger == "target.shell_init":
        target_kwargs.update({"shell": "bash", "shell_login": True, "shell_interactive": True, "shell_init": "kimi"})
    else:
        target_kwargs["bootstrap"] = "kimi"

    target = SimpleNamespace(**target_kwargs)
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=target,
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=target,
            ),
        ]
    )


def _completed_run(
    run_id: str,
    *,
    pipeline_name: str = "demo",
    status: str = "completed",
    nodes: dict[str, object] | None = None,
    pipeline_nodes: list[object] | None = None,
):
    return SimpleNamespace(
        id=run_id,
        status=SimpleNamespace(value=status),
        pipeline=SimpleNamespace(name=pipeline_name, nodes=pipeline_nodes or []),
        started_at="2026-03-08T04:11:03+00:00",
        finished_at="2026-03-08T04:11:10+00:00",
        nodes=nodes or {},
        model_dump=lambda mode="json": {"id": run_id, "status": status},
    )


def test_validate_command_outputs_normalized_pipeline(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": "cli", "working_dir": ".", "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", str(pipeline_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "cli"
    assert payload["nodes"][0]["id"] == "alpha"


def test_init_command_prints_default_pipeline_template():
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "from agentflow import" in result.stdout
    assert "codex(" in result.stdout
    assert "claude(" in result.stdout
    assert "kimi(" in result.stdout


def test_init_command_prints_codex_repo_sweep_batched_template():
    result = runner.invoke(app, ["init", "--template", "codex-repo-sweep-batched"])

    assert result.exit_code == 0
    assert "codex-repo-sweep-batched-128" in result.stdout
    assert "node_defaults" in result.stdout
    assert "agent_defaults" in result.stdout
    assert "fanout, merge" in result.stdout


def test_init_command_prints_custom_codex_repo_sweep_batched_template():
    result = runner.invoke(
        app,
        [
            "init",
            "--template",
            "codex-repo-sweep-batched",
            "--set",
            "shards=64",
            "--set",
            "batch_size=8",
            "--set",
            "concurrency=20",
            "--set",
            "focus=security bugs, privilege boundaries, and missing coverage",
            "--set",
            "name=custom-repo-sweep-64",
            "--set",
            "working_dir=./custom_repo_sweep",
        ],
    )

    assert result.exit_code == 0
    assert "custom-repo-sweep-64" in result.stdout
    assert "./custom_repo_sweep" in result.stdout
    assert "concurrency=20" in result.stdout
    assert "64," in result.stdout
    assert "size=8" in result.stdout
    assert "Focus on security bugs, privilege boundaries, and missing coverage." in result.stdout


def test_templates_command_lists_current_bundled_templates():
    result = runner.invoke(app, ["templates"])

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        "Bundled templates:",
        "- pipeline: Generic Codex/Claude/Kimi starter DAG. (source: `examples/airflow_like.py`; use: `agentflow init --template pipeline`)",
        "- codex-repo-sweep-batched: Configurable large-scale Codex repo sweep that uses `fanout` and `merge` to keep 128-shard maintainer reviews readable. (params: `shards=128`, `batch_size=16`, `concurrency=32`, `focus=bugs, risky code paths, and missing tests`, `name=codex-repo-sweep-batched-<shards>`, `working_dir=./codex_repo_sweep_batched_<shards>`; source: `examples/airflow_like_fuzz_batched.py`; use: `agentflow init --template codex-repo-sweep-batched`)",
    ]


def test_init_command_writes_selected_template_to_destination(tmp_path):
    destination = tmp_path / "templates" / "pipeline.py"

    result = runner.invoke(app, ["init", str(destination), "--template", "pipeline"])

    assert result.exit_code == 0
    assert result.stdout == f"Wrote `pipeline` template to `{destination}`.\n"
    assert "from agentflow import" in destination.read_text(encoding="utf-8")


def test_init_command_writes_codex_repo_sweep_batched_template_to_destination(tmp_path):
    destination = tmp_path / "templates" / "repo-sweep-batched.py"

    result = runner.invoke(
        app,
        [
            "init",
            str(destination),
            "--template",
            "codex-repo-sweep-batched",
            "--set",
            "shards=64",
            "--set",
            "batch_size=8",
            "--set",
            "concurrency=20",
            "--set",
            "focus=security bugs, privilege boundaries, and missing coverage",
            "--set",
            "name=custom-repo-sweep-64",
            "--set",
            "working_dir=./custom_repo_sweep",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == f"Wrote `codex-repo-sweep-batched` template to `{destination}`.\n"
    rendered = destination.read_text(encoding="utf-8")
    assert "custom-repo-sweep-64" in rendered
    assert "concurrency=20" in rendered
    assert "node_defaults" in rendered
    assert "agent_defaults" in rendered
    assert "Focus on security bugs, privilege boundaries, and missing coverage." in rendered


def test_init_command_refuses_to_overwrite_existing_file_without_force(tmp_path):
    destination = tmp_path / "pipeline.py"
    destination.write_text("name: keep-me\n", encoding="utf-8")

    result = runner.invoke(app, ["init", str(destination), "--template", "pipeline"])

    assert result.exit_code == 1
    assert result.stderr == f"Destination `{destination}` already exists. Use `--force` to overwrite it.\n"
    assert destination.read_text(encoding="utf-8") == "name: keep-me\n"


def test_init_command_overwrites_existing_file_with_force(tmp_path):
    destination = tmp_path / "pipeline.py"
    destination.write_text("name: keep-me\n", encoding="utf-8")

    result = runner.invoke(app, ["init", str(destination), "--template", "pipeline", "--force"])

    assert result.exit_code == 0
    assert result.stdout == f"Wrote `pipeline` template to `{destination}`.\n"
    assert "from agentflow import" in destination.read_text(encoding="utf-8")


def test_init_command_rejects_directory_destination(tmp_path):
    destination = tmp_path / "templates"
    destination.mkdir()

    result = runner.invoke(app, ["init", str(destination), "--template", "pipeline"])

    assert result.exit_code == 1
    assert result.stderr == f"Destination `{destination}` is a directory.\n"


def test_init_command_rejects_unknown_template():
    result = runner.invoke(app, ["init", "--template", "missing-template"])

    assert result.exit_code != 0


def test_init_command_rejects_invalid_template_setting_format():
    result = runner.invoke(app, ["init", "--template", "codex-repo-sweep-batched", "--set", "shards"])

    assert result.exit_code != 0
    assert "template settings must use KEY=VALUE form" in result.stderr


def test_init_command_rejects_duplicate_template_setting():
    result = runner.invoke(
        app,
        [
            "init",
            "--template",
            "codex-repo-sweep-batched",
            "--set",
            "shards=64",
            "--set",
            "shards=32",
        ],
    )

    assert result.exit_code != 0
    assert "template setting `shards` was provided more than" in result.stderr
    assert "once" in result.stderr


def test_init_command_rejects_unknown_template_setting():
    result = runner.invoke(app, ["init", "--template", "codex-repo-sweep-batched", "--set", "unknown=1"])

    assert result.exit_code != 0
    assert "does not" in result.stderr
    assert "recognize `unknown`" in result.stderr
    assert "`shards`" in result.stderr
    assert "`focus`" in result.stderr
    assert "`working_dir`" in result.stderr


def test_init_command_rejects_invalid_integer_template_setting():
    result = runner.invoke(app, ["init", "--template", "codex-repo-sweep-batched", "--set", "shards=many"])

    assert result.exit_code != 0
    assert "template `codex-repo-sweep-batched` expects" in result.stderr
    assert "`shards` to be an integer" in result.stderr


def test_init_command_rejects_template_settings_for_static_template():
    result = runner.invoke(app, ["init", "--template", "pipeline", "--set", "shards=64"])

    assert result.exit_code != 0
    assert "template `pipeline` does not accept `--set` values" in result.stderr


def test_python_module_entrypoint_displays_help():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "-m", "agentflow", "--help"],
        capture_output=True,
        check=False,
        cwd=repo_root,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Usage:" in completed.stdout
    assert "init" in completed.stdout
    assert "templates" in completed.stdout
    assert "validate" in completed.stdout
    assert "runs" in completed.stdout
    assert "show" in completed.stdout
    assert "cancel" in completed.stdout
    assert "rerun" in completed.stdout
    assert "check-local" in completed.stdout
    assert "toolchain-local" in completed.stdout
    assert "inference" in completed.stdout
    assert "smoke" in completed.stdout


def test_inference_command_launches_service_by_default(monkeypatch):
    captured: dict[str, object] = {}

    def _launch(request):
        captured["request"] = request
        return SkyInferenceService(
            name="agentflow-inference-qwen",
            cluster_name="agentflow-infer-qwen",
            base_url="https://inference.example/v1",
            api_key=request.api_key or "generated-key",
            model_id=request.model_id,
            engine=request.engine,
            gpu=request.gpu,
            port=request.port,
            use_spot=request.use_spot,
            request_id="request-1",
            provider={
                "name": "agentflow-inference-qwen",
                "base_url": "https://inference.example/v1",
                "api_key_env": "OPENAI_API_KEY",
                "wire_api": "openai-completions",
                "env": {"OPENAI_API_KEY": request.api_key or "generated-key"},
            },
        )

    monkeypatch.setattr(agentflow.inference, "launch_sky_inference_service", _launch)

    result = runner.invoke(
        app,
        [
            "inference",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--gpu",
            "aws:1xl4@us-east-1",
            "--api-key",
            "test-key",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    request = captured["request"]
    assert request.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert request.use_spot is True
    assert request.api_key == "test-key"
    assert request.port == 8000
    payload = json.loads(result.stdout)
    assert payload["base_url"] == "https://inference.example/v1"
    assert payload["api_key"] == "test-key"
    assert payload["provider"]["base_url"] == "https://inference.example/v1"


def test_inference_command_launches_sky_batch_job_with_spot_default(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"1","prompt":"hello"}\n', encoding="utf-8")

    def _launch(request):
        captured["request"] = request
        return SkyInferenceLaunch(
            name="agentflow-inference-qwen",
            job_ids=[42],
            request_id="request-1",
            detached=True,
            gpu=request.gpu,
            engine=request.engine,
            use_spot=request.use_spot,
        )

    monkeypatch.setattr(agentflow.inference, "launch_sky_inference_job", _launch)

    result = runner.invoke(
        app,
        [
            "inference",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--mode",
            "batch",
            "--gpu",
            "aws:8x8xb200@us-east-2",
            "--input",
            str(prompts),
            "--image-id",
            "ami-explicit",
            "--detach",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    request = captured["request"]
    assert request.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert request.use_spot is True
    assert request.gpu.infra == "aws/us-east-2"
    assert request.gpu.num_nodes == 8
    assert request.gpu.accelerators == "B200:8"
    assert request.image_id == "ami-explicit"
    payload = json.loads(result.stdout)
    assert payload["job_ids"] == [42]
    assert payload["gpu"]["num_nodes"] == 8


def test_inference_command_allows_no_spot(monkeypatch):
    captured: dict[str, object] = {}

    def _launch(request):
        captured["request"] = request
        return SkyInferenceLaunch(
            name="agentflow-inference-qwen",
            job_ids=[7],
            request_id="request-2",
            detached=True,
            gpu=request.gpu,
            engine=request.engine,
            use_spot=request.use_spot,
        )

    monkeypatch.setattr(agentflow.inference, "launch_sky_inference_job", _launch)

    result = runner.invoke(
        app,
        [
            "inference",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--mode",
            "batch",
            "--gpu",
            "aws:1xl4@us-east-1",
            "--prompt",
            "hello",
            "--no-spot",
            "--detach",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert captured["request"].use_spot is False
    assert "Spot: False" in result.stdout


def test_inference_command_rejects_bad_gpu_selector():
    result = runner.invoke(app, ["inference", "model", "--gpu", "aws:8x"])

    assert result.exit_code != 0
    assert "Invalid value for --gpu" in result.stderr


def test_inference_command_rejects_workers_without_pool():
    result = runner.invoke(app, ["inference", "model", "--mode", "batch", "--gpu", "1xl4", "--workers", "2"])

    assert result.exit_code != 0
    assert "Invalid value for --workers" in result.stderr


def test_inference_command_rejects_batch_inputs_in_service_mode():
    result = runner.invoke(app, ["inference", "model", "--gpu", "1xl4", "--prompt", "hi"])

    assert result.exit_code != 0
    assert "`--prompt`" in result.stderr
    assert "`--mode batch`" in result.stderr


def test_render_doctor_summary_appends_bash_startup_summary_suffix():
    report = DoctorReport(
        status="ok",
        checks=[
            DoctorCheck(
                name="bash_login_startup",
                status="ok",
                detail="startup ready",
                context=_bash_startup_context(
                    "~/.profile -> ~/.bashrc",
                    startup_files={
                        "~/.bash_profile": "missing",
                        "~/.bash_login": "missing",
                        "~/.profile": "present",
                    },
                ),
            )
        ],
    )

    assert _render_doctor_summary(report) == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready (startup=~/.profile -> ~/.bashrc, files=~/.bash_profile=missing, ~/.bash_login=missing, ~/.profile=present)"
    )


def test_doctor_command_json_preserves_bash_startup_context(monkeypatch):
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="ok",
            checks=[
                DoctorCheck(
                    name="bash_login_startup",
                    status="ok",
                    detail="startup ready",
                    context={
                        "login_file": "~/.profile",
                        "startup_chain": ["~/.profile", "~/.bashrc"],
                        "startup_summary": "~/.profile -> ~/.bashrc",
                        "startup_files": {
                            "~/.bash_profile": "missing",
                            "~/.bash_login": "missing",
                            "~/.profile": "present",
                        },
                        "startup_files_summary": "~/.bash_profile=missing, ~/.bash_login=missing, ~/.profile=present",
                        "bashrc_reachable": True,
                        "bashrc_exists": True,
                    },
                )
            ],
        ),
    )

    result = runner.invoke(app, ["doctor", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["pipeline"] == {
        "auto_preflight": {
            "enabled": True,
            "reason": "path matches the bundled real-agent smoke pipeline.",
            "matches": [
                {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
            ],
            "match_summary": [
                "codex_plan (codex) via `target.bootstrap`",
                "claude_review (claude) via `target.bootstrap`",
            ],
        }
    }
    assert next(check for check in payload["checks"] if check["name"] == "bash_login_startup") == {
        "name": "bash_login_startup",
        "status": "ok",
        "detail": "startup ready",
        "context": {
            "login_file": "~/.profile",
            "startup_chain": ["~/.profile", "~/.bashrc"],
            "startup_summary": "~/.profile -> ~/.bashrc",
            "startup_files": {
                "~/.bash_profile": "missing",
                "~/.bash_login": "missing",
                "~/.profile": "present",
            },
            "startup_files_summary": "~/.bash_profile=missing, ~/.bash_login=missing, ~/.profile=present",
            "bashrc_reachable": True,
            "bashrc_exists": True,
        },
    }


def test_validate_resolves_working_dir_relative_to_pipeline_file(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipelines"
    pipeline_dir.mkdir()
    workdir = pipeline_dir / "workspace"
    workdir.mkdir()
    task_dir = workdir / "task"
    task_dir.mkdir()
    pipeline_path = pipeline_dir / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'cli', "working_dir": 'workspace', "nodes": [{"id": 'alpha', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "cwd": 'task'}}]}),
        encoding="utf-8",
    )
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    result = runner.invoke(app, ["validate", str(pipeline_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["working_dir"] == str(workdir.resolve())
    assert payload["nodes"][0]["target"]["cwd"] == str(task_dir.resolve())


def test_validate_applies_local_target_defaults_and_resolves_relative_cwd(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipelines"
    pipeline_dir.mkdir()
    workdir = pipeline_dir / "workspace"
    workdir.mkdir()
    default_task_dir = workdir / "shared-task"
    default_task_dir.mkdir()
    pipeline_path = pipeline_dir / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'cli-local-defaults', "working_dir": 'workspace', "local_target_defaults": {"shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": ['command -v kimi >/dev/null 2>&1', 'kimi'], "cwd": 'shared-task'}, "nodes": [{"id": 'alpha', "agent": 'codex', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    result = runner.invoke(app, ["validate", str(pipeline_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["local_target_defaults"]["cwd"] == str(default_task_dir.resolve())
    assert payload["nodes"][0]["target"]["shell"] == "bash"
    assert payload["nodes"][0]["target"]["cwd"] == str(default_task_dir.resolve())


def test_validate_reports_pipeline_validation_error_without_traceback(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'cli-invalid-bash', "working_dir": '.', "nodes": [{"id": 'alpha', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash --rcfile="$HOME/.bashrc" -ic \'kimi && {command}\'', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", str(pipeline_path)])

    assert result.exit_code == 1
    assert f"Failed to load pipeline `{pipeline_path}`:" in result.stderr
    assert "unsupported bash long option" in result.stderr
    assert "--rcfile=..." in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_rejects_incompatible_kimi_bootstrap_shorthand(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'cli-invalid-kimi-bootstrap', "working_dir": '.', "nodes": [{"id": 'alpha', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi', "shell": 'sh'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", str(pipeline_path)])

    assert result.exit_code == 1
    assert f"Failed to load pipeline `{pipeline_path}`:" in result.stderr
    assert "`target.bootstrap: kimi` requires bash-style shell bootstrap" in result.stderr
    assert "`target.shell` resolves to `sh`" in result.stderr
    assert "Traceback" not in result.stderr


def test_inspect_command_outputs_launch_summary(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-demo', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "model": 'gpt-5', "prompt": 'Reply with exactly: codex ok', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}, {"id": 'review', "agent": 'claude', "depends_on": ['plan'], "prompt": 'Review this: {{ nodes.plan.output }}'}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Pipeline: inspect-demo" in result.stdout
    assert "Auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap." in result.stdout
    assert "Auto preflight matches: plan (codex) via `target.shell_init`" in result.stdout
    assert "Note: Dependency references use placeholder node outputs" in result.stdout
    assert "- plan [codex/local]" in result.stdout
    assert "Model: gpt-5" in result.stdout
    assert "Mode: tools=read_only, capture=final" in result.stdout
    assert "Bootstrap: shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, init=kimi" in result.stdout
    assert "Startup files: ~/.bash_profile=missing, ~/.bash_login=missing, ~/.profile=present" in result.stdout
    assert "Launch: bash -l -i -c 'kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'" in result.stdout
    assert "Runtime files:" not in result.stdout
    assert "Prompt: Review this: <inspect placeholder for nodes.plan.output>" in result.stdout


def test_inspect_command_summary_infers_login_and_interactive_from_shell_wrapper(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-wrapper-flags', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lic 'kimi && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert (
        "Bootstrap: shell=bash -lic 'kimi && {command}', login=true, startup=~/.profile -> ~/.bashrc, interactive=true"
        in result.stdout
    )


def test_inspect_command_node_filter_omits_unrelated_placeholder_note(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-node-filter', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'Reply with exactly: codex ok'}, {"id": 'review', "agent": 'claude', "depends_on": ['plan'], "prompt": 'Review this: {{ nodes.plan.output }}'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "plan", "--output", "summary"])

    assert result.exit_code == 0
    assert "- plan [codex/local]" in result.stdout
    assert "- review [claude/local]" not in result.stdout
    assert "Note: Dependency references use placeholder node outputs" not in result.stdout


def test_inspect_command_omits_placeholder_note_for_plain_text_nodes_reference(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-plain-text-nodes', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'Explain why the string nodes.plan.output should stay literal in this doc example.'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Prompt: Explain why the string nodes.plan.output should stay literal in this doc example." in result.stdout
    assert "Note: Dependency references use placeholder node outputs" not in result.stdout


def test_inspect_defaults_to_json_output_when_not_tty(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-auto-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["inspect", str(pipeline_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["name"] == "inspect-auto-json"
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": False,
        "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
        "matches": [],
        "match_summary": [],
    }
    assert payload["nodes"][0]["id"] == "review"


def test_inspect_defaults_to_summary_output_on_tty(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-auto-summary', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["inspect", str(pipeline_path)])

    assert result.exit_code == 0
    assert "Pipeline: inspect-auto-summary" in result.stdout
    assert "- review [claude/local]" in result.stdout


def test_inspect_command_supports_json_output_and_redacts_env(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'Reply with exactly: claude ok', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["name"] == "inspect-json"
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [
            {
                "node_id": "review",
                "agent": "claude",
                "trigger": "target.shell_init",
            }
        ],
        "match_summary": ["review (claude) via `target.shell_init`"],
    }
    assert [node["id"] for node in payload["nodes"]] == ["review"]
    assert payload["nodes"][0]["tools"] == "read_only"
    assert payload["nodes"][0]["capture"] == "final"
    assert payload["nodes"][0]["resolved_provider"] == {
        "name": "kimi",
        "base_url": "https://api.kimi.com/coding/",
        "api_key_env": "ANTHROPIC_API_KEY",
        "wire_api": None,
        "headers": {},
        "env": {},
    }
    assert payload["nodes"][0]["prepared"]["env"]["ANTHROPIC_API_KEY"] == "<redacted>"
    assert payload["nodes"][0]["launch"]["env"]["ANTHROPIC_API_KEY"] == "<redacted>"
    assert payload["nodes"][0]["launch"]["env"]["ANTHROPIC_BASE_URL"] == "https://api.kimi.com/coding/"


def test_inspect_command_supports_json_summary_output(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-json-summary', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'Reply with exactly: claude ok', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == {
        "name": "inspect-json-summary",
        "working_dir": str(tmp_path.resolve()),
        "node_count": 1,
        "auto_preflight": "enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "auto_preflight_matches": ["review (claude) via `target.shell_init`"],
    }
    assert payload["nodes"] == [
        {
            "id": "review",
            "agent": "claude",
            "target": "local",
            "tools": "read_only",
            "capture": "final",
            "provider": "kimi, key=ANTHROPIC_API_KEY, url=https://api.kimi.com/coding/",
            "auth": "`ANTHROPIC_API_KEY` via `target.shell_init` (`kimi` helper)",
            "bootstrap": "shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, init=kimi",
            "bash_startup_files": {
                "~/.bash_profile": "missing",
                "~/.bash_login": "missing",
                "~/.profile": "present",
            },
            "prompt_preview": "Reply with exactly: claude ok",
            "prepared_command": "claude -p 'Reply with exactly: claude ok' --output-format stream-json --verbose --permission-mode bypassPermissions --tools Read,Glob,Grep,LS,NotebookRead,Task,TaskOutput,TodoRead,WebFetch,WebSearch",
            "launch": "bash -l -i -c 'kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'",
            "cwd": str(tmp_path.resolve()),
            "env_keys": ["AGENTFLOW_TARGET_COMMAND", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
            "warnings": ["Bash login startup reaches `~/.bashrc`, but that file does not exist."],
        }
    ]


def test_inspect_command_json_summary_warns_when_login_bash_has_no_user_startup_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-missing-login-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=none"
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup will not load any user file from `HOME` because `~/.bash_profile`, `~/.bash_login`, and `~/.profile` are all missing."
    ]


def test_inspect_command_json_summary_warns_when_login_bash_does_not_reach_bashrc(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text('export PATH="$HOME/bin:$PATH"\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-missing-bashrc-bridge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "preset=kimi, shell=bash, login=true, startup=~/.bash_profile, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    )
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup uses `~/.bash_profile`, but it does not reach `~/.bashrc`."
    ]


def test_inspect_command_json_summary_warns_when_login_bash_shadows_profile_bridge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text('export PATH="$HOME/bin:$PATH"\n', encoding="utf-8")
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("kimi(){ :; }\n", encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-shadowed-bashrc-bridge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "preset=kimi, shell=bash, login=true, startup=~/.bash_profile, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    )
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup uses `~/.bash_profile`, so `~/.profile` will never run even though it references "
        "`~/.bashrc`; reference `~/.bashrc` or `~/.profile` from `~/.bash_profile`."
    ]
    recommendation = build_bash_login_shell_bridge_recommendation(home=home)
    assert recommendation is not None
    assert payload["nodes"][0]["shell_bridge"] == recommendation.as_dict()


def test_inspect_command_json_summary_warns_when_login_auth_bridge_is_shadowed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text('export PATH="$HOME/bin:$PATH"\n', encoding="utf-8")
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("export ANTHROPIC_API_KEY=from-bashrc\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-shadowed-login-auth-bridge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "expects `ANTHROPIC_API_KEY` via current environment, `node.env`, `provider.env`, or local shell bootstrap"
    )
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=~/.bash_profile"
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup uses `~/.bash_profile`, so `~/.profile` will never run even though it references "
        "`~/.bashrc`; reference `~/.bashrc` or `~/.profile` from `~/.bash_profile`."
    ]
    recommendation = build_bash_login_shell_bridge_recommendation(home=home)
    assert recommendation is not None
    assert payload["nodes"][0]["shell_bridge"] == recommendation.as_dict()


def test_inspect_command_json_summary_warns_when_login_bash_startup_is_unreadable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    login_file = home / ".bash_profile"
    login_file.write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    login_file.chmod(0)
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-unreadable-bash-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "preset=kimi, shell=bash, login=true, startup=~/.bash_profile, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    )
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup uses `~/.bash_profile`, but AgentFlow could not read `~/.bash_profile` while "
        "checking whether login shells reach `~/.bashrc`: Permission denied."
    ]


def test_inspect_command_json_summary_accepts_login_bash_startup_with_undecodable_bytes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_bytes(b'\xff\xfeif [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n')
    (home / ".bashrc").write_text("kimi(){ :; }\n", encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-undecodable-bash-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "preset=kimi, shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    )
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_json_summary_warns_when_login_bash_reaches_missing_bashrc(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-missing-bashrc-file', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "preset=kimi, shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    )
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup reaches `~/.bashrc`, but that file does not exist."
    ]


def test_inspect_command_summary_includes_shell_bridge_for_shadowed_profile_bridge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text('export PATH="$HOME/bin:$PATH"\n', encoding="utf-8")
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("kimi(){ :; }\n", encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-shadowed-bashrc-bridge-summary', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    recommendation = build_bash_login_shell_bridge_recommendation(home=home)
    assert recommendation is not None
    assert (
        f"  Shell bridge suggestion for `{recommendation.target}` from `{recommendation.source}`:"
        in result.stdout
    )
    assert f"  Reason: {recommendation.reason}" in result.stdout
    assert '  if [ -f "$HOME/.profile" ]; then' in result.stdout
    assert '    . "$HOME/.profile"' in result.stdout
    assert "  fi" in result.stdout


def test_inspect_command_supports_kimi_bootstrap_shorthand(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-bootstrap-shorthand', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'Reply with exactly: claude ok', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == {
        "name": "inspect-bootstrap-shorthand",
        "working_dir": str(tmp_path.resolve()),
        "node_count": 1,
        "auto_preflight": "enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "auto_preflight_matches": ["review (claude) via `target.bootstrap`"],
    }
    assert payload["nodes"] == [
        {
            "id": "review",
            "agent": "claude",
            "target": "local",
            "tools": "read_only",
            "capture": "final",
            "provider": "kimi, key=ANTHROPIC_API_KEY, url=https://api.kimi.com/coding/",
            "auth": "`ANTHROPIC_API_KEY` via `target.bootstrap` (`kimi` helper)",
            "bootstrap": "preset=kimi, shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, init=command -v kimi >/dev/null 2>&1 && kimi",
            "bash_startup_files": {
                "~/.bash_profile": "missing",
                "~/.bash_login": "missing",
                "~/.profile": "present",
            },
            "prompt_preview": "Reply with exactly: claude ok",
            "prepared_command": "claude -p 'Reply with exactly: claude ok' --output-format stream-json --verbose --permission-mode bypassPermissions --tools Read,Glob,Grep,LS,NotebookRead,Task,TaskOutput,TodoRead,WebFetch,WebSearch",
            "launch": "bash -l -i -c 'command -v kimi >/dev/null 2>&1 && kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'",
            "cwd": str(tmp_path.resolve()),
            "env_keys": ["AGENTFLOW_TARGET_COMMAND", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
            "warnings": ["Bash login startup reaches `~/.bashrc`, but that file does not exist."],
        }
    ]


def test_inspect_command_merges_extra_shell_init_with_kimi_bootstrap_shorthand(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-bootstrap-shell-init-merge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi', "shell_init": ['export EXTRA_FLAG=1']}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight_matches"] == ["review (claude) via `target.bootstrap`"]
    assert payload["nodes"] == [
        {
            "id": "review",
            "agent": "claude",
            "target": "local",
            "tools": "read_only",
            "capture": "final",
            "provider": "kimi, key=ANTHROPIC_API_KEY, url=https://api.kimi.com/coding/",
            "auth": "`ANTHROPIC_API_KEY` via `target.bootstrap` (`kimi` helper)",
            "bootstrap": "preset=kimi, shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, init=export EXTRA_FLAG=1 && command -v kimi >/dev/null 2>&1 && kimi",
            "bash_startup_files": {
                "~/.bash_profile": "missing",
                "~/.bash_login": "missing",
                "~/.profile": "present",
            },
            "prompt_preview": "hi",
            "prepared_command": "claude -p hi --output-format stream-json --verbose --permission-mode bypassPermissions --tools Read,Glob,Grep,LS,NotebookRead,Task,TaskOutput,TodoRead,WebFetch,WebSearch",
            "launch": "bash -l -i -c 'export EXTRA_FLAG=1 && command -v kimi >/dev/null 2>&1 && kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'",
            "cwd": str(tmp_path.resolve()),
            "env_keys": ["AGENTFLOW_TARGET_COMMAND", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
            "warnings": ["Bash login startup reaches `~/.bashrc`, but that file does not exist."],
        }
    ]


def test_inspect_command_summary_notes_when_launch_env_overrides_current_base_url(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-base-url-override', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["launch_env_overrides"] == [
        {
            "key": "ANTHROPIC_BASE_URL",
            "current_value": "https://open.bigmodel.cn/api/anthropic",
            "launch_value": "https://api.kimi.com/coding/",
            "source": "provider.base_url",
        }
    ]
    assert payload["nodes"][0]["notes"] == [
        "Launch env overrides current `ANTHROPIC_BASE_URL` from `https://open.bigmodel.cn/api/anthropic` to `https://api.kimi.com/coding/` via `provider.base_url`.",
        "Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.shell_init` (`kimi` helper).",
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_warns_when_local_launch_inherits_current_base_url(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["launch_env_inheritances"] == [
        {
            "key": "ANTHROPIC_BASE_URL",
            "current_value": "https://open.bigmodel.cn/api/anthropic",
            "source": "current environment",
        }
    ]
    assert payload["nodes"][0]["warnings"] == [
        "Launch inherits current `ANTHROPIC_BASE_URL` value `https://open.bigmodel.cn/api/anthropic`; configure `provider` or `node.env` explicitly if you want Claude routing pinned for this node."
    ]


def test_inspect_command_summary_warns_when_explicit_claude_provider_leaves_base_url_inherited(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-explicit-claude-provider-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": {"name": 'anthropic', "api_key_env": 'ANTHROPIC_API_KEY'}, "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["launch_env_inheritances"] == [
        {
            "key": "ANTHROPIC_BASE_URL",
            "current_value": "https://open.bigmodel.cn/api/anthropic",
            "source": "current environment",
        }
    ]
    assert payload["nodes"][0]["warnings"] == [
        "Launch inherits current `ANTHROPIC_BASE_URL` value `https://open.bigmodel.cn/api/anthropic`; configure `provider` or `node.env` explicitly if you want Claude routing pinned for this node."
    ]


def test_inspect_command_summary_warns_when_explicit_codex_provider_leaves_base_url_inherited(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-explicit-codex-provider-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "provider": {"name": 'openai', "api_key_env": 'OPENAI_API_KEY', "wire_api": 'responses'}, "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://oai-relay.ctf.so/openai")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["launch_env_inheritances"] == [
        {
            "key": "OPENAI_BASE_URL",
            "current_value": "https://oai-relay.ctf.so/openai",
            "source": "current environment",
        }
    ]
    assert payload["nodes"][0]["warnings"] == [
        "Launch inherits current `OPENAI_BASE_URL` value `https://oai-relay.ctf.so/openai`; configure `provider` or `node.env` explicitly if you want Codex routing pinned for this node."
    ]


def test_inspect_command_summary_uses_node_home_for_base_url_bootstrap_detection(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()
    (launch_home / ".profile").write_text(
        "export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/\n",
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-home-base-url-bootstrap', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "env": {"HOME": str(launch_home)}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "launch_env_inheritances" not in payload["nodes"][0]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_notes_when_node_env_clears_current_base_url(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-base-url-cleared', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"OPENAI_BASE_URL": ''}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://oai-relay.ctf.so/openai")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "launch_env_inheritances" not in payload["nodes"][0]
    assert payload["nodes"][0]["launch_env_overrides"] == [
        {
            "key": "OPENAI_BASE_URL",
            "current_value": "https://oai-relay.ctf.so/openai",
            "launch_value": "",
            "source": "node.env",
        }
    ]
    assert payload["nodes"][0]["notes"] == [
        "Launch env clears current `OPENAI_BASE_URL` value `https://oai-relay.ctf.so/openai` via `node.env`."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_skips_current_base_url_inheritance_for_container_targets(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-base-url-inheritance-container', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'container', "image": 'python:3.11-slim'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "launch_env_inheritances" not in payload["nodes"][0]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_redacts_sensitive_launch_env_override_details(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-secret-override', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"OPENAI_API_KEY": 'node-secret'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "current-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["launch_env_overrides"] == [
        {"key": "OPENAI_API_KEY", "redacted": True, "source": "node.env"}
    ]
    assert payload["nodes"][0]["notes"] == [
        "Launch env overrides current `OPENAI_API_KEY` for this node via `node.env`."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_reports_bootstrap_auth_override_for_current_secret(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-bootstrap-auth-override', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "current-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap_env_overrides"] == [
        {"key": "ANTHROPIC_API_KEY", "redacted": True, "source": "target.bootstrap", "helper": "kimi"}
    ]
    assert payload["nodes"][0]["notes"] == [
        "Launch env overrides current `ANTHROPIC_BASE_URL` from `https://open.bigmodel.cn/api/anthropic` to `https://api.kimi.com/coding/` via `provider.base_url`.",
        "Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper)."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_keeps_bootstrap_auth_override_when_kimi_base_url_already_matches(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-bootstrap-auth-override-kimi-base-url', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "current-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "launch_env_overrides" not in payload["nodes"][0]
    assert payload["nodes"][0]["bootstrap_env_overrides"] == [
        {"key": "ANTHROPIC_API_KEY", "redacted": True, "source": "target.bootstrap", "helper": "kimi"}
    ]
    assert payload["nodes"][0]["notes"] == [
        "Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper)."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_reports_bootstrap_auth_override_for_launch_secret(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-bootstrap-auth-override-launch-secret', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "env": {"ANTHROPIC_API_KEY": 'launch-secret'}, "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap_env_overrides"] == [
        {
            "key": "ANTHROPIC_API_KEY",
            "redacted": True,
            "origin": "launch_env",
            "source": "target.bootstrap",
            "helper": "kimi",
        }
    ]
    assert payload["nodes"][0]["notes"] == [
        "Local shell bootstrap overrides launch `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper)."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_redacts_inline_shell_bootstrap_secrets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-secret-shell-init', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": ['export ANTHROPIC_API_KEY=super-secret-inline', 'kimi']}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    json_result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json"])

    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["nodes"][0]["target"]["shell_init"] == [
        "export ANTHROPIC_API_KEY=<redacted>",
        "kimi",
    ]
    assert payload["nodes"][0]["launch"]["command"] == [
        "bash",
        "-l",
        "-i",
        "-c",
        'export ANTHROPIC_API_KEY=<redacted> && kimi && eval "$AGENTFLOW_TARGET_COMMAND"',
    ]
    assert payload["nodes"][0]["launch"]["command_text"] == (
        "bash -l -i -c 'export ANTHROPIC_API_KEY=<redacted> && kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'"
    )

    summary_result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json-summary"])

    assert summary_result.exit_code == 0
    summary_payload = json.loads(summary_result.stdout)
    assert summary_payload["nodes"][0]["bootstrap"] == (
        "shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, init=export ANTHROPIC_API_KEY=<redacted> && kimi"
    )
    assert summary_payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell_init`"
    assert summary_payload["nodes"][0]["launch"] == (
        "bash -l -i -c 'export ANTHROPIC_API_KEY=<redacted> && kimi && eval \"$AGENTFLOW_TARGET_COMMAND\"'"
    )


def test_inspect_command_summary_shows_codex_login_fallback(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-codex-auth', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "Codex CLI login or `OPENAI_API_KEY` via current environment"


def test_inspect_command_summary_mentions_codex_kimi_bootstrap_with_openai_env(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-codex-kimi-env', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "`OPENAI_API_KEY` via current environment; `target.shell_init` (`kimi` helper) also runs before launch"
    )


def test_inspect_command_summary_mentions_codex_kimi_bootstrap_for_login_fallback(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-codex-kimi-login', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "Codex CLI login via `target.shell_init` (`kimi` helper) or `OPENAI_API_KEY` via current environment"
    )


def test_inspect_command_summary_treats_shell_prefix_openai_key_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-codex-shell-prefix-openai-key', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'env OPENAI_API_KEY=test-shell-key bash -c'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`OPENAI_API_KEY` via `target.shell`"


def test_inspect_command_summary_does_not_treat_empty_shell_prefix_openai_key_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-codex-shell-prefix-empty-openai-key', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'env OPENAI_API_KEY= bash -c'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "Codex CLI login or `OPENAI_API_KEY` via current environment"


def test_inspect_command_summary_treats_shell_prefix_provider_key_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-shell-prefix-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'env ANTHROPIC_API_KEY=test-shell-key bash -c'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell`"


def test_inspect_command_summary_treats_sourced_shell_file_as_auth_source(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".anthropic.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-sourced-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={home} bash -lc 'source ~/.anthropic.env && {{command}}'"}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell`"


def test_inspect_command_summary_treats_bash_env_file_as_auth_source(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-bash-env-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={home} BASH_ENV=$HOME/auth.env bash -c '{{command}}'"}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell`"


def test_inspect_command_summary_ignores_bash_env_file_for_interactive_bash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-bash-env-interactive-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={home} BASH_ENV=$HOME/auth.env bash -ic '{{command}}'"}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "expects `ANTHROPIC_API_KEY` via current environment, `node.env`, `provider.env`, or local shell bootstrap"
    )


def test_inspect_command_summary_ignores_bash_env_file_for_structured_interactive_bash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-bash-env-structured-interactive-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "env": {"BASH_ENV": '$HOME/auth.env'}, "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={home} bash", "shell_interactive": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "expects `ANTHROPIC_API_KEY` via current environment, `node.env`, `provider.env`, or local shell bootstrap"
    )


def test_inspect_command_summary_treats_node_env_bash_env_file_as_auth_source(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-node-env-bash-env-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "env": {"BASH_ENV": '$HOME/auth.env'}, "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={home} bash -c '{{command}}'"}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell`"


def test_inspect_command_summary_treats_login_shell_startup_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-login-startup-provider-key', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentflow.local_shell.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via local bash login startup files"


def test_inspect_command_summary_uses_target_cwd_for_relative_login_startup_sources(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f .bashrc ]; then . .bashrc; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("export ANTHROPIC_API_KEY=from-relative-bashrc\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-relative-login-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "cwd": str(home)}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via local bash login startup files"
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=~/.profile -> ~/.bashrc"
    assert payload["nodes"][0]["cwd"] == str(home.resolve())


def test_inspect_command_json_summary_recommends_shell_bridge_for_relative_login_source_outside_launch_cwd(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f .bashrc ]; then . .bashrc; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("export ANTHROPIC_API_KEY=from-relative-bashrc\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-relative-login-startup-shell-bridge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=~/.profile"
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup uses `~/.profile`, but it does not reach `~/.bashrc`."
    ]
    recommendation = build_bash_login_shell_bridge_recommendation(home=home, cwd=tmp_path)
    assert recommendation is not None
    assert payload["nodes"][0]["shell_bridge"] == recommendation.as_dict()


def test_inspect_command_summary_uses_launch_env_for_login_startup_sources(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    auth_file = tmp_path / "anthropic.env"
    auth_file.write_text("export ANTHROPIC_API_KEY=from-launch-env-file\n", encoding="utf-8")
    (home / ".profile").write_text(
        'if [ -n "${AGENTFLOW_KIMI_ENV_FILE:-}" ]; then . "$AGENTFLOW_KIMI_ENV_FILE"; fi\n',
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-claude-env-gated-login-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "env": {"AGENTFLOW_KIMI_ENV_FILE": str(auth_file)}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via local bash login startup files"
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=~/.profile"


def test_inspect_command_summary_reports_default_claude_auth_requirement_without_explicit_provider(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-default-claude-provider-auth', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == (
        "expects `ANTHROPIC_API_KEY` via current environment, `node.env`, `provider.env`, or local shell bootstrap"
    )


def test_inspect_command_summary_treats_custom_kimi_provider_shell_bootstrap_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-custom-kimi-provider', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": {"name": 'kimi-proxy', "base_url": 'https://api.kimi.com/coding/', "api_key_env": 'ANTHROPIC_API_KEY'}, "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell_init` (`kimi` helper)"


def test_inspect_command_summary_treats_custom_kimi_provider_env_base_url_shell_bootstrap_as_auth_source(
    tmp_path,
    monkeypatch,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-custom-kimi-provider-env-base-url', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": {"name": 'kimi-proxy', "api_key_env": 'ANTHROPIC_API_KEY', "env": {"ANTHROPIC_BASE_URL": 'https://api.kimi.com/coding/'}}, "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell_init` (`kimi` helper)"


def test_inspect_command_summary_treats_anthropic_provider_kimi_bootstrap_as_auth_source(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-anthropic-provider-kimi-bootstrap', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.bootstrap` (`kimi` helper)"


def test_inspect_command_summary_reports_kimi_bootstrap_base_url_override_for_anthropic_provider(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-anthropic-provider-kimi-bootstrap-base-url-override', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap_env_overrides"] == [
        {
            "key": "ANTHROPIC_BASE_URL",
            "current_value": "https://api.anthropic.com",
            "bootstrap_value": "https://api.kimi.com/coding/",
            "origin": "launch_env",
            "source": "target.bootstrap",
            "helper": "kimi",
        }
    ]
    assert payload["nodes"][0]["notes"] == [
        "Local shell bootstrap overrides launch `ANTHROPIC_BASE_URL` from `https://api.anthropic.com` to `https://api.kimi.com/coding/` via `target.bootstrap` (`kimi` helper)."
    ]
    assert not payload["nodes"][0].get("warnings")


def test_inspect_command_summary_prefers_kimi_helper_auth_over_node_env(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-helper-over-node-env', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "env": {"ANTHROPIC_API_KEY": 'node-secret'}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "current-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `target.shell_init` (`kimi` helper)"


def test_inspect_command_reports_disabled_auto_preflight_for_plain_pipeline(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-no-preflight', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'Reply with exactly: codex ok'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == (
        "disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap."
    )


def test_inspect_command_handles_symlinked_login_file_outside_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    (dotfiles / "bash_profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (home / ".bash_profile").symlink_to(dotfiles / "bash_profile")
    (home / ".bashrc").write_text("export READY=1\n", encoding="utf-8")

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-symlinked-login-file', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=~/.bash_profile -> ~/.bashrc"


def test_inspect_command_uses_node_env_home_for_login_startup_summary(tmp_path, monkeypatch):
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    (host_home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (host_home / ".bashrc").write_text("export READY=1\n", encoding="utf-8")
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-home-env-login-startup', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"HOME": str(launch_home)}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: host_home)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == "shell=bash, login=true, startup=none"


def test_inspect_command_reports_auto_preflight_match_sources(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-preflight-matches', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}, {"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lic 'kimi && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"]["matches"] == [
        {
            "node_id": "plan",
            "agent": "codex",
            "trigger": "target.shell_init",
        },
        {
            "node_id": "review",
            "agent": "claude",
            "trigger": "target.shell",
        },
    ]
    assert payload["pipeline"]["auto_preflight"]["match_summary"] == [
        "plan (codex) via `target.shell_init`",
        "review (claude) via `target.shell`",
    ]


def test_inspect_command_ignores_non_helper_kimi_substrings_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-substrings', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'printf kimi-ready && {command}'", "shell_init": 'echo kimi-ready'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": False,
        "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
        "matches": [],
        "match_summary": [],
    }


def test_inspect_command_ignores_plain_text_kimi_tokens_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-plain-text', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'echo kimi && {command}'", "shell_init": 'echo kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": False,
        "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
        "matches": [],
        "match_summary": [],
    }


def test_inspect_command_supports_shell_init_command_lists(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    monkeypatch.setattr("agentflow.local_shell.Path.home", lambda: home)

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-shell-init-list', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": ['command -v kimi >/dev/null 2>&1', 'kimi']}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Auto preflight matches: review (claude) via `target.shell_init`" in result.stdout
    assert (
        "Bootstrap: shell=bash, login=true, startup=~/.profile -> ~/.bashrc, interactive=true, "
        "init=command -v kimi >/dev/null 2>&1 && kimi"
    ) in result.stdout


def test_inspect_command_ignores_kimi_probe_commands_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-probes', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'command -v kimi >/dev/null && {command}'", "shell_init": 'type kimi >/dev/null 2>&1'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": False,
        "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
        "matches": [],
        "match_summary": [],
    }


def test_inspect_command_warns_when_kimi_shell_init_is_not_interactive(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-warning', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Warning: `shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable." in result.stdout
    assert "Set `target.shell_interactive: true` or use `bash -lic`." in result.stdout


def test_inspect_command_json_summary_warns_when_kimi_shell_init_disables_login_startup(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-noprofile', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash --noprofile -lic '{command}'", "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["bootstrap"] == (
        "shell=bash --noprofile -lic '{command}', login=true, startup=disabled (--noprofile), "
        "interactive=true, init=kimi"
    )
    assert payload["nodes"][0]["warnings"] == [
        "Bash login startup is disabled by `--noprofile`, so login shells will not load `~/.bash_profile`, `~/.bash_login`, or `~/.profile`.",
        "`shell_init: kimi` uses bash with `--noprofile`, so login startup files never reach `~/.bashrc`. Remove `--noprofile`, source the helper explicitly, or export provider variables directly.",
    ]
    assert "shell_bridge" not in payload["nodes"][0]


def test_inspect_command_accepts_interactive_bash_rcfile_wrapper(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-rcfile', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash --rcfile ~/.bashrc -ic '{command}'", "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0].get("warnings") is None


def test_inspect_command_accepts_bash_env_kimi_wrapper(tmp_path):
    shell_env = tmp_path / "shell.env"
    shell_env.write_text("kimi(){ :; }\n", encoding="utf-8")

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-bash-env', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env BASH_ENV={shell_env} bash -c 'kimi && {{command}}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == {
        "name": "inspect-kimi-bash-env",
        "working_dir": str(tmp_path.resolve()),
        "node_count": 1,
        "auto_preflight": "enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "auto_preflight_matches": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0].get("warnings") is None


def test_inspect_command_accepts_bash_env_wrapper_that_sources_helper_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".agentflow-kimi").write_text("kimi(){ :; }\n", encoding="utf-8")
    (home / "shell.env").write_text('source "$HOME/.agentflow-kimi"\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-bash-env-source', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "env BASH_ENV=$HOME/shell.env bash -c 'kimi && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == {
        "name": "inspect-kimi-bash-env-source",
        "working_dir": str(tmp_path.resolve()),
        "node_count": 1,
        "auto_preflight": "enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "auto_preflight_matches": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0].get("warnings") is None


def test_inspect_command_json_summary_includes_kimi_shell_init_warning(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-warning-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["warnings"] == [
        "`shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`."
    ]


def test_inspect_command_warns_when_explicit_kimi_shell_wrapper_is_not_interactive(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-wrapper-warning', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'kimi && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Warning: `target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable." in result.stdout
    assert "Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`." in result.stdout


def test_inspect_command_detects_eval_style_kimi_wrapper_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": "inspect-kimi-eval-wrapper", "working_dir": ".", "nodes": [{"id": "review", "agent": "claude", "prompt": "hi", "target": {"kind": "local", "shell": "bash -lc 'eval \"$(kimi)\" && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.shell"}],
        "match_summary": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0]["warnings"] == [
        "`target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`."
    ]


def test_inspect_command_detects_backtick_eval_kimi_wrapper_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-backtick-eval-wrapper', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'eval `kimi` && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.shell"}],
        "match_summary": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0]["warnings"] == [
        "`target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`."
    ]


def test_inspect_command_detects_export_kimi_wrapper_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-export-wrapper', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'export $(kimi) && {command}'"}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.shell"}],
        "match_summary": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0]["warnings"] == [
        "`target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`."
    ]


def test_inspect_command_detects_env_var_eval_kimi_wrapper_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-env-eval-wrapper', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash -lc \'KIMI_ENV="$(kimi)" && eval "$KIMI_ENV" && {command}\''}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.shell"}],
        "match_summary": ["review (claude) via `target.shell`"],
    }
    assert payload["nodes"][0]["warnings"] == [
        "`target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`."
    ]


def test_inspect_command_warns_when_bash_env_points_to_guarded_home_bashrc(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(
        "case $- in\n    *i*) ;;\n      *) return;;\nesac\n\nkimi(){ :; }\n",
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-bashenv-bashrc-guard', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'env BASH_ENV=$HOME/.bashrc bash -c', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(agentflow.local_shell.Path, "home", classmethod(lambda cls: home))

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["warnings"] == [
        "`shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`."
    ]


def test_inspect_command_detects_kimi_agent_in_auto_preflight(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-agent-bootstrap', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
        "matches": [{"node_id": "review", "agent": "kimi", "trigger": "target.shell_init"}],
        "match_summary": ["review (kimi) via `target.shell_init`"],
    }
    assert payload["nodes"][0]["warnings"] == [
        "`shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`."
    ]


def test_inspect_command_warns_when_kimi_shell_init_uses_non_bash_shell(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-non-bash-kimi', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'sh', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["warnings"] == [
        "`shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. "
        "Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, "
        "use `bash -lic`, or export provider variables directly."
    ]


def test_inspect_command_resolves_default_kimi_provider_in_json(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-default-provider', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["provider"] is None
    assert payload["nodes"][0]["resolved_provider"] == {
        "name": "moonshot",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "KIMI_API_KEY",
        "wire_api": None,
        "headers": {},
        "env": {},
    }


def test_inspect_command_surfaces_default_kimi_provider_in_json_summary(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-kimi-default-provider-summary', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["provider"] == "moonshot, key=KIMI_API_KEY, url=https://api.moonshot.ai/v1"


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_warn_when_kimi_shell_init_is_not_interactive(tmp_path, monkeypatch, command):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'kimi-preflight-warning', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("pipeline", pipeline.name),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert captured == {"pipeline": "kimi-preflight-warning"}
    payload = json.loads(result.stderr or result.stdout)
    assert payload["status"] == "warning"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "warning",
            "detail": "Node `review`: `shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`.",
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_fail_when_kimi_shell_init_uses_non_bash_shell(tmp_path, monkeypatch, command):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'kimi-preflight-non-bash', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'sh', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: pytest.fail("pipeline should not run"),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr or result.stdout)
    assert payload["status"] == "failed"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "failed",
            "detail": "Node `review`: `shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, use `bash -lic`, or export provider variables directly.",
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_accepts_kimi_shell_init_when_login_shell_already_loads_kimi(
    tmp_path,
    monkeypatch,
    command,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'codex-kimi-preflight-warning', "working_dir": '.', "nodes": [{"id": 'codex_plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        agentflow.cli,
        "probe_target_bash_startup_env_var",
        lambda *args, **kwargs: agentflow.local_shell.BashStartupEnvProbeResult(exported=False),
    )
    monkeypatch.setattr(agentflow.inspection, "target_bash_startup_exports_env_var", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (
            pytest.fail("codex auth probe should not run when the kimi shell bootstrap already warns")
            if "codex login status" in str((kwargs.get("env") or {}).get("AGENTFLOW_TARGET_COMMAND", ""))
            else subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("pipeline", pipeline.name),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert captured == {"pipeline": "codex-kimi-preflight-warning"}
    assert result.stderr == ""
    assert result.stdout == ""


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_skips_codex_auth_probe_when_kimi_shell_init_uses_non_bash_shell(
    tmp_path,
    monkeypatch,
    command,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'codex-kimi-preflight-non-bash', "working_dir": '.', "nodes": [{"id": 'codex_plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "shell": 'sh', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("codex auth probe should not run when the kimi shell bootstrap already fails"),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: pytest.fail("pipeline should not run"),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr or result.stdout)
    assert payload["status"] == "failed"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "failed",
            "detail": "Node `codex_plan`: `shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, use `bash -lic`, or export provider variables directly.",
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_warn_when_explicit_kimi_shell_wrapper_is_not_interactive(tmp_path, monkeypatch, command):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'kimi-wrapper-preflight-warning', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'kimi && {command}'"}}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("pipeline", pipeline.name),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert captured == {"pipeline": "kimi-wrapper-preflight-warning"}
    payload = json.loads(result.stderr)
    assert payload["status"] == "warning"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "warning",
            "detail": "Node `review`: `target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`.",
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_warn_when_eval_style_kimi_wrapper_is_not_interactive(tmp_path, monkeypatch, command):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": "kimi-eval-wrapper-preflight-warning", "working_dir": ".", "nodes": [{"id": "review", "agent": "claude", "prompt": "hi", "target": {"kind": "local", "shell": "bash -lc 'eval \"$(kimi)\" && {command}'"}}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("pipeline", pipeline.name),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert captured == {"pipeline": "kimi-eval-wrapper-preflight-warning"}
    payload = json.loads(result.stderr)
    assert payload["status"] == "warning"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "warning",
            "detail": "Node `review`: `target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`.",
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_preflight_warn_when_backtick_eval_kimi_wrapper_is_not_interactive(tmp_path, monkeypatch, command):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'kimi-backtick-wrapper-preflight-warning', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "target": {"kind": 'local', "shell": "bash -lc 'eval `kimi` && {command}'"}}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("pipeline", pipeline.name),
    )

    result = runner.invoke(app, [*command, str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert captured == {"pipeline": "kimi-backtick-wrapper-preflight-warning"}
    payload = json.loads(result.stderr)
    assert payload["status"] == "warning"
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "kimi_shell_bootstrap",
            "status": "warning",
            "detail": "Node `review`: `target.shell` uses `kimi` with bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Add `-i`, set `target.shell_interactive: true`, or use `bash -lic`.",
        },
    ]


def test_inspect_command_redacts_auth_and_header_style_env_keys(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-redaction', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "provider": {"name": 'kimi', "base_url": 'https://api.kimi.com/coding/', "api_key_env": 'ANTHROPIC_API_KEY', "env": {"UPSTREAM_AUTH_HEADER": 'Bearer top-secret'}, "headers": {"Authorization": 'Bearer top-secret'}}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "review", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    env = payload["nodes"][0]["launch"]["env"]
    assert env["ANTHROPIC_API_KEY"] == "<redacted>"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "<redacted>"
    assert env["UPSTREAM_AUTH_HEADER"] == "<redacted>"


def test_inspect_command_summary_shows_resolved_provider(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-provider', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "model": 'claude-sonnet-4-5', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Model: claude-sonnet-4-5" in result.stdout
    assert "Provider: kimi, key=ANTHROPIC_API_KEY, url=https://api.kimi.com/coding/" in result.stdout


def test_inspect_command_prefers_node_env_auth_source_over_provider_env(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-node-env-auth', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "env": {"ANTHROPIC_API_KEY": 'node-secret'}, "provider": {"name": 'kimi-proxy', "base_url": 'https://example.test/anthropic', "api_key_env": 'ANTHROPIC_API_KEY', "env": {"ANTHROPIC_API_KEY": 'provider-secret'}}}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"][0]["auth"] == "`ANTHROPIC_API_KEY` via `node.env`"


def test_inspect_command_surfaces_skills_and_mcp_names(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-integrations', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "skills": ['repo-map', 'release-notes'], "mcps": [{"name": 'github', "transport": 'streamable_http', "url": 'https://example.com/mcp'}, {"name": 'filesystem', "command": 'npx', "args": ['-y', '@modelcontextprotocol/server-filesystem', '.']}]}]}),
        encoding="utf-8",
    )

    summary_result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "summary"])

    assert summary_result.exit_code == 0
    assert "Skills: repo-map, release-notes" in summary_result.stdout
    assert "MCPs: github, filesystem" in summary_result.stdout

    json_summary_result = runner.invoke(app, ["inspect", str(pipeline_path), "--output", "json-summary"])

    assert json_summary_result.exit_code == 0
    payload = json.loads(json_summary_result.stdout)
    assert payload["nodes"][0]["skills"] == ["repo-map", "release-notes"]
    assert payload["nodes"][0]["mcps"] == ["github", "filesystem"]


def test_inspect_command_rejects_unknown_nodes(tmp_path):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'inspect-error', "working_dir": '.', "nodes": [{"id": 'alpha', "agent": 'codex', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(pipeline_path), "--node", "missing"])

    assert result.exit_code != 0
    assert "unknown node ids: ['missing']" in result.stderr


def test_serve_uses_runtime_env_vars(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("AGENTFLOW_RUNS_DIR", "/tmp/agentflow-env-runs")
    monkeypatch.setenv("AGENTFLOW_MAX_CONCURRENT_RUNS", "7")

    def fake_build_runtime(runs_dir: str, max_concurrent_runs: int):
        captured["runs_dir"] = runs_dir
        captured["max_concurrent_runs"] = max_concurrent_runs
        return object(), object()

    monkeypatch.setattr(agentflow.cli, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(agentflow.cli, "_create_web_app", lambda store, orchestrator: "fake-app")
    monkeypatch.setattr(agentflow.cli, "_serve_web_app", lambda app, host, port: captured.update({"app": app, "host": host, "port": port}))

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert captured == {
        "runs_dir": "/tmp/agentflow-env-runs",
        "max_concurrent_runs": 7,
        "app": "fake-app",
        "host": "127.0.0.1",
        "port": 8000,
    }


def test_run_uses_runtime_env_vars(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("AGENTFLOW_RUNS_DIR", "/tmp/agentflow-env-runs")
    monkeypatch.setenv("AGENTFLOW_MAX_CONCURRENT_RUNS", "9")

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-123")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return SimpleNamespace(
                status=SimpleNamespace(value="completed"),
                model_dump=lambda mode="json": {"id": run_id, "status": "completed"},
            )

    def fake_build_runtime(runs_dir: str, max_concurrent_runs: int):
        captured["runs_dir"] = runs_dir
        captured["max_concurrent_runs"] = max_concurrent_runs
        return object(), FakeOrchestrator()

    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)

    result = runner.invoke(app, ["run", "pipeline.yaml", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"id": "run-123", "status": "completed"}
    assert captured["runs_dir"] == "/tmp/agentflow-env-runs"
    assert captured["max_concurrent_runs"] == 9
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-123"
    assert captured["wait_timeout"] is None


def test_run_defaults_to_summary_on_tty(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="run-auto-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(
                run_id,
                pipeline_name="auto-summary-pipeline",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model="gpt-5-codex",
                        provider=None,
                    )
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    )
                },
            )

    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()))
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: not err)

    result = runner.invoke(app, ["run", "pipeline.yaml"])

    assert result.exit_code == 0
    assert "Run run-auto-summary: completed" in result.stdout
    assert "Pipeline: auto-summary-pipeline" in result.stdout
    assert "Run dir: .agentflow/runs/run-auto-summary" in result.stdout
    assert "- codex_plan [codex, model=gpt-5-codex]: completed (attempt 1, exit 0) - codex ok" in result.stdout


def test_run_supports_summary_output(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="run-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(
                run_id,
                pipeline_name="summary-pipeline",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model="gpt-5-codex",
                        provider=None,
                    )
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    )
                },
            )

    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()))
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["run", "pipeline.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert "Run run-summary: completed" in result.stdout
    assert "Pipeline: summary-pipeline" in result.stdout
    assert "Run dir: .agentflow/runs/run-summary" in result.stdout
    assert "- codex_plan [codex, model=gpt-5-codex]: completed (attempt 1, exit 0) - codex ok" in result.stdout


def test_run_supports_json_summary_output(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="run-json-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(
                run_id,
                pipeline_name="summary-pipeline",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model="gpt-5-codex",
                        provider=None,
                    )
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    )
                },
            )

    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()))
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["run", "pipeline.yaml", "--output", "json-summary"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "id": "run-json-summary",
        "status": "completed",
        "pipeline": {"name": "summary-pipeline"},
        "started_at": "2026-03-08T04:11:03+00:00",
        "finished_at": "2026-03-08T04:11:10+00:00",
        "duration": "7.0s",
        "duration_seconds": 7.0,
        "run_dir": ".agentflow/runs/run-json-summary",
        "nodes": [
            {
                "id": "codex_plan",
                "status": "completed",
                "agent": "codex",
                "model": "gpt-5-codex",
                "attempts": 1,
                "exit_code": 0,
                "preview": "codex ok",
            }
        ],
    }


def test_runs_supports_summary_output(monkeypatch):
    recent = _completed_run("run-list-recent", pipeline_name="recent-pipeline")
    older = _completed_run("run-list-older", pipeline_name="older-pipeline", status="running")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            list_runs=lambda: [recent, older],
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert "Runs: 2" in result.stdout
    assert "- run-list-recent: completed - recent-pipeline (7.0s)" in result.stdout
    assert "- run-list-older: running - older-pipeline (7.0s)" in result.stdout


def test_runs_defaults_to_first_twenty_records(monkeypatch):
    records = [_completed_run(f"run-{index:02d}", pipeline_name=f"pipeline-{index:02d}") for index in range(25)]

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            list_runs=lambda: records,
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert "Runs: 20 of 25" in result.stdout
    assert "run-00" in result.stdout
    assert "run-19" in result.stdout
    assert "run-20" not in result.stdout


def test_runs_limit_zero_shows_all_records(monkeypatch):
    records = [_completed_run(f"run-{index:02d}", pipeline_name=f"pipeline-{index:02d}") for index in range(25)]

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            list_runs=lambda: records,
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["runs", "--limit", "0"])

    assert result.exit_code == 0
    assert "Runs: 25" in result.stdout
    assert "run-24" in result.stdout


def test_runs_supports_json_summary_output(monkeypatch):
    recent = _completed_run("run-list-json", pipeline_name="json-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            list_runs=lambda: [recent],
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )

    result = runner.invoke(app, ["runs", "--output", "json-summary"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == [
        {
            "id": "run-list-json",
            "status": "completed",
            "pipeline": {"name": "json-pipeline"},
            "started_at": "2026-03-08T04:11:03+00:00",
            "finished_at": "2026-03-08T04:11:10+00:00",
            "duration": "7.0s",
            "duration_seconds": 7.0,
            "run_dir": ".agentflow/runs/run-list-json",
            "nodes": [],
        }
    ]


def test_runs_defaults_to_json_when_stdout_is_not_a_tty(monkeypatch):
    recent = _completed_run("run-list-json-auto", pipeline_name="json-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            list_runs=lambda: [recent],
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == [{"id": "run-list-json-auto", "status": "completed"}]


def test_show_outputs_summary_for_persisted_run(monkeypatch):
    record = _completed_run("run-show", pipeline_name="show-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            get_run=lambda run_id: record,
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["show", "run-show"])

    assert result.exit_code == 0
    assert "Run run-show: completed" in result.stdout
    assert "Pipeline: show-pipeline" in result.stdout
    assert "Run dir: .agentflow/runs/run-show" in result.stdout


def test_show_defaults_to_json_when_stdout_is_not_a_tty(monkeypatch):
    record = _completed_run("run-show-json", pipeline_name="show-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(
            get_run=lambda run_id: record,
            run_dir=lambda run_id: Path(runs_dir) / run_id,
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["show", "run-show-json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"id": "run-show-json", "status": "completed"}


def test_show_exits_for_missing_run(monkeypatch):
    def _missing(run_id: str):
        raise KeyError(run_id)

    monkeypatch.setattr(
        agentflow.cli,
        "_build_store",
        lambda runs_dir: SimpleNamespace(get_run=_missing),
    )

    result = runner.invoke(app, ["show", "missing-run"])

    assert result.exit_code == 1
    assert "Run `missing-run` not found in `.agentflow/runs`." in result.stderr


def test_cancel_outputs_summary_for_existing_run(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def cancel(self, run_id: str):
            captured["run_id"] = run_id
            return _completed_run("run-cancel", pipeline_name="cancel-pipeline", status="cancelling")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["cancel", "run-cancel"])

    assert result.exit_code == 0
    assert captured["run_id"] == "run-cancel"
    assert "Run run-cancel: cancelling" in result.stdout
    assert "Pipeline: cancel-pipeline" in result.stdout


def test_cancel_defaults_to_json_when_stdout_is_not_a_tty(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def cancel(self, run_id: str):
            captured["run_id"] = run_id
            return _completed_run("run-cancel-json", pipeline_name="cancel-pipeline", status="cancelling")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["cancel", "run-cancel-json"])

    assert result.exit_code == 0
    assert captured["run_id"] == "run-cancel-json"
    assert json.loads(result.stdout) == {"id": "run-cancel-json", "status": "cancelling"}


def test_cancel_exits_for_missing_run(monkeypatch):
    class FakeOrchestrator:
        async def cancel(self, run_id: str):
            raise KeyError(run_id)

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )

    result = runner.invoke(app, ["cancel", "missing-run"])

    assert result.exit_code == 1
    assert "Run `missing-run` not found in `.agentflow/runs`." in result.stderr


def test_rerun_supports_json_summary_output(monkeypatch):
    captured: dict[str, object] = {}

    queued_run = SimpleNamespace(
        id="run-rerun-new",
        status=SimpleNamespace(value="queued"),
        pipeline=SimpleNamespace(name="rerun-pipeline", nodes=[]),
        started_at=None,
        finished_at=None,
        nodes={},
        model_dump=lambda mode="json": {"id": "run-rerun-new", "status": "queued"},
    )

    class FakeOrchestrator:
        async def rerun(self, run_id: str):
            captured["run_id"] = run_id
            return queued_run

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run("run-rerun-new", pipeline_name="rerun-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )

    result = runner.invoke(app, ["rerun", "run-old", "--output", "json-summary"])

    assert result.exit_code == 0
    assert captured["run_id"] == "run-old"
    assert captured["wait_run_id"] == "run-rerun-new"
    assert captured["wait_timeout"] is None
    assert json.loads(result.stdout) == {
        "id": "run-rerun-new",
        "status": "completed",
        "pipeline": {"name": "rerun-pipeline"},
        "started_at": "2026-03-08T04:11:03+00:00",
        "finished_at": "2026-03-08T04:11:10+00:00",
        "duration": "7.0s",
        "duration_seconds": 7.0,
        "run_dir": ".agentflow/runs/run-rerun-new",
        "nodes": [],
    }


def test_rerun_defaults_to_json_when_stdout_is_not_a_tty(monkeypatch):
    captured: dict[str, object] = {}

    queued_run = SimpleNamespace(
        id="run-rerun-json",
        status=SimpleNamespace(value="queued"),
        pipeline=SimpleNamespace(name="rerun-pipeline", nodes=[]),
        started_at=None,
        finished_at=None,
        nodes={},
        model_dump=lambda mode="json": {"id": "run-rerun-json", "status": "queued"},
    )

    class FakeOrchestrator:
        async def rerun(self, run_id: str):
            captured["run_id"] = run_id
            return queued_run

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run("run-rerun-json", pipeline_name="rerun-pipeline")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["rerun", "run-old"])

    assert result.exit_code == 0
    assert captured["run_id"] == "run-old"
    assert captured["wait_run_id"] == "run-rerun-json"
    assert captured["wait_timeout"] is None
    assert json.loads(result.stdout) == {"id": "run-rerun-json", "status": "completed"}


def test_rerun_exits_for_missing_run(monkeypatch):
    class FakeOrchestrator:
        async def rerun(self, run_id: str):
            raise KeyError(run_id)

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )

    result = runner.invoke(app, ["rerun", "missing-run"])

    assert result.exit_code == 1
    assert "Run `missing-run` not found in `.agentflow/runs`." in result.stderr


def test_run_skips_preflight_for_custom_pipeline_in_auto_mode(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-custom")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-run")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: (_ for _ in ()).throw(AssertionError("doctor should not run")))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"id": "run-custom", "status": "completed"}
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-custom"
    assert captured["wait_timeout"] is None


def test_run_auto_runs_preflight_for_custom_pipeline_with_kimi_shell_init(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-custom-kimi")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-run")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=True),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-custom-kimi"
    assert captured["wait_timeout"] is None


def test_run_show_preflight_prints_successful_summary_to_stderr(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-show-preflight")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-run")

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr("agentflow.doctor.subprocess.run", _codex_ready_and_auth_subprocess(_CODEX_AUTH_VIA_API_KEY_AND_LOGIN_STATUS_EXIT_CODE))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_interactive=True, shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml", "--output", "summary", "--show-preflight"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; "
        "`codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via "
        "`OPENAI_API_KEY` + `codex login status`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )
    assert "Run run-show-preflight: completed" in result.stdout
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-show-preflight"
    assert captured["wait_timeout"] is None


def test_run_auto_preflight_stops_when_local_codex_auth_is_unavailable(monkeypatch):
    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_run(*args, **kwargs):
        env = kwargs.get("env") or {}
        target_command = env.get("AGENTFLOW_TARGET_COMMAND", "")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1 if "subprocess.run" in target_command and "OPENAI_API_KEY" in target_command else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime should not build when preflight fails")),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert (result.stderr or result.stdout) == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_auth: failed - Node `codex_plan` (codex) cannot authenticate local Codex after the node shell bootstrap; `codex login status` fails and `OPENAI_API_KEY` is not set in the current environment, `node.env`, or `provider.env`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )


def test_run_auto_preflight_stops_when_local_codex_is_unavailable_after_shell_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime should not build when preflight fails")),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_ready: failed - Node `codex_plan` (codex) cannot launch local Codex after the node shell bootstrap; `codex --version` fails in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )


def test_run_auto_preflight_stops_when_local_claude_is_unavailable_after_shell_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr(subprocess, "run", _completed_subprocess(returncode=1))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime should not build when preflight fails")),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: failed - Node `claude_review` (claude) cannot launch local Claude after the node shell bootstrap; `claude --version` fails in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_run_auto_runs_preflight_for_custom_pipeline_with_kimi_agent(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-custom-kimi-agent")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-agent-run")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="kimi"),
                env={"KIMI_API_KEY": "inline-secret"},
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-custom-kimi-agent"
    assert captured["wait_timeout"] is None


def test_run_auto_runs_preflight_for_custom_pipeline_with_shell_init_command_list(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-custom-kimi-shell-init-list")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-shell-init-list")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init=["command -v kimi >/dev/null 2>&1", "kimi"]),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-custom-kimi-shell-init-list"
    assert captured["wait_timeout"] is None


def test_run_auto_ignores_non_helper_kimi_substrings(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-custom-non-kimi")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-non-kimi-run")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: (_ for _ in ()).throw(AssertionError("doctor should not run")))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash -lc 'printf kimi-ready'", shell_init="echo kimi-ready"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-custom-non-kimi"
    assert captured["wait_timeout"] is None


def test_run_runs_preflight_for_explicit_bundled_pipeline_path(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-explicit-default")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _doctor_report()

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", bundled_path, "--output", "json"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == bundled_path
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-explicit-default"
    assert captured["wait_timeout"] is None


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["smoke"],
    ],
)
def test_run_and_smoke_bundled_preflight_applies_pipeline_shell_checks(monkeypatch, command):
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.py").resolve())

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="ok",
            checks=[DoctorCheck(name="kimi_shell_helper", status="ok", detail="ready")],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: pytest.fail("pipeline should not run"),
    )

    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="sh",
                    shell_init="kimi",
                    shell_login=False,
                    shell_interactive=False,
                    cwd=None,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)

    args = [*command, bundled_path, "--output", "json"] if command[0] == "run" else [*command, "--output", "json"]
    if command[0] == "smoke":
        args = [*command, "--output", "json"]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
            {
                "name": "kimi_shell_bootstrap",
                "status": "failed",
                "detail": "Node `review`: `shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, use `bash -lic`, or export provider variables directly.",
            },
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {
                        "node_id": "review",
                        "agent": "claude",
                        "trigger": "target.shell_init",
                    }
                ],
                "match_summary": ["review (claude) via `target.shell_init`"],
            }
        },
    }


def test_run_can_disable_preflight_for_bundled_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-no-preflight")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: (_ for _ in ()).throw(AssertionError("doctor should not run")))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", bundled_path, "--preflight", "never"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == bundled_path
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-no-preflight"
    assert captured["wait_timeout"] is None


def test_run_stops_when_detected_preflight_fails(monkeypatch):
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))
    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (_ for _ in ()).throw(AssertionError("runtime should not build")))

    result = runner.invoke(app, ["run", bundled_path, "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [{"name": "kimi_shell_helper", "status": "failed", "detail": "missing"}],
    }


def test_smoke_uses_bundled_pipeline_by_default(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("AGENTFLOW_RUNS_DIR", "/tmp/agentflow-smoke-runs")
    monkeypatch.setenv("AGENTFLOW_MAX_CONCURRENT_RUNS", "5")

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-123")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(
                run_id,
                pipeline_name="local-real-agents-kimi-smoke",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model=None,
                        provider=None,
                    ),
                    SimpleNamespace(
                        id="claude_review",
                        agent=SimpleNamespace(value="claude"),
                        model=None,
                        provider="kimi",
                    ),
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    ),
                    "claude_review": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="claude ok",
                        output="claude ok",
                        stderr_lines=[],
                    ),
                },
            )

    def fake_build_runtime(runs_dir: str, max_concurrent_runs: int):
        captured["runs_dir"] = runs_dir
        captured["max_concurrent_runs"] = max_concurrent_runs
        return object(), FakeOrchestrator()

    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), fake_build_runtime(runs_dir, max_concurrent_runs)[1]))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "Run smoke-123: completed" in result.stdout
    assert "Pipeline: local-real-agents-kimi-smoke" in result.stdout
    assert "- codex_plan [codex]: completed (attempt 1, exit 0) - codex ok" in result.stdout
    assert "- claude_review [claude, provider=kimi]: completed (attempt 1, exit 0) - claude ok" in result.stdout
    assert captured["loaded_path"] == "examples/local-real-agents-kimi-smoke.yaml"
    assert captured["runs_dir"] == "/tmp/agentflow-smoke-runs"
    assert captured["max_concurrent_runs"] == 5
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-123"
    assert captured["wait_timeout"] is None


def test_render_run_summary_includes_provider_account_diagnosis_for_api_errors():
    record = _completed_run(
        "smoke-provider-error",
        pipeline_name="local-real-agents-kimi-smoke",
        status="failed",
        pipeline_nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                model=None,
                provider="kimi",
            )
        ],
        nodes={
            "claude_review": SimpleNamespace(
                status=SimpleNamespace(value="failed"),
                current_attempt=1,
                attempts=[SimpleNamespace(number=1)],
                exit_code=1,
                final_response='API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}',
                output='API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}',
                stderr_lines=[],
                stdout_lines=[],
            )
        },
    )

    rendered = agentflow.cli._render_run_summary(record, run_dir=".agentflow/runs/smoke-provider-error")

    assert "Run smoke-provider-error: failed" in rendered
    assert (
        "- claude_review [claude, provider=kimi]: failed (attempt 1, exit 1) - "
        'API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}'
    ) in rendered
    assert (
        "  Diagnosis: Claude-on-Kimi reached the provider, but the request was rejected with a "
        "membership/billing-style API error. The local shell bootstrap is likely working; "
        "check the upstream provider account state."
    ) in rendered


def test_build_run_summary_includes_provider_account_diagnosis_for_api_errors():
    record = _completed_run(
        "smoke-provider-error-json",
        pipeline_name="local-real-agents-kimi-smoke",
        status="failed",
        pipeline_nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                model=None,
                provider="kimi",
            )
        ],
        nodes={
            "claude_review": SimpleNamespace(
                status=SimpleNamespace(value="failed"),
                current_attempt=1,
                attempts=[SimpleNamespace(number=1)],
                exit_code=1,
                final_response='API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}',
                output='API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}',
                stderr_lines=[],
                stdout_lines=[],
            )
        },
    )

    summary = agentflow.cli._build_run_summary(record, run_dir=".agentflow/runs/smoke-provider-error-json")

    assert summary["nodes"] == [
        {
            "id": "claude_review",
            "status": "failed",
            "agent": "claude",
            "provider": "kimi",
            "attempts": 1,
            "exit_code": 1,
            "preview": 'API Error: 402 {"error":{"type":"invalid_request_error","message":"membership required"}}',
            "diagnosis": (
                "Claude-on-Kimi reached the provider, but the request was rejected with a "
                "membership/billing-style API error. The local shell bootstrap is likely working; "
                "check the upstream provider account state."
            ),
        }
    ]


def test_check_local_uses_bundled_pipeline_by_default(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("AGENTFLOW_RUNS_DIR", "/tmp/agentflow-check-local-runs")
    monkeypatch.setenv("AGENTFLOW_MAX_CONCURRENT_RUNS", "4")
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)
    _mock_local_readiness_info(monkeypatch)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-123")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    def fake_build_runtime(runs_dir: str, max_concurrent_runs: int):
        captured["runs_dir"] = runs_dir
        captured["max_concurrent_runs"] = max_concurrent_runs
        return object(), FakeOrchestrator()

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            fake_build_runtime(runs_dir, max_concurrent_runs)[1],
        ),
    )

    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
        ]
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline run/smoke auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline run/smoke auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )
    assert "Run check-local-123: completed" in result.stdout
    assert captured["loaded_path"] == bundled_path
    assert captured["runs_dir"] == "/tmp/agentflow-check-local-runs"
    assert captured["max_concurrent_runs"] == 4
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "check-local-123"
    assert captured["wait_timeout"] is None


def test_check_local_accepts_wrapper_preflight_flags_for_cli_parity(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-wrapper-flags")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    fake_pipeline = SimpleNamespace(nodes=[])
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)

    result = runner.invoke(app, ["check-local", "--preflight", "auto", "--show-preflight"])

    assert result.exit_code == 0
    assert "Doctor: ok" in result.stderr
    assert "Pipeline run/smoke auto preflight: enabled - path matches the bundled real-agent smoke pipeline." in result.stderr
    assert "Run check-local-wrapper-flags: completed" in result.stdout
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "check-local-wrapper-flags"


def test_check_local_rejects_preflight_never(monkeypatch):
    monkeypatch.setattr(
        agentflow.cli,
        "_doctor_report_for_path",
        lambda path: (_ for _ in ()).throw(AssertionError("doctor should not run when `--preflight never` is rejected")),
    )

    result = runner.invoke(app, ["check-local", "--preflight", "never"])

    assert result.exit_code == 2


def test_check_local_reuses_preflight_loaded_pipeline(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-reuse")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")

    loaded_pipelines = [object(), object()]

    def fake_load(path: str):
        captured["loaded_path"] = path
        captured["load_calls"] = int(captured.get("load_calls", 0)) + 1
        pipeline = loaded_pipelines.pop(0)
        captured.setdefault("loaded_pipelines", []).append(pipeline)
        return pipeline

    monkeypatch.setattr(agentflow.cli, "_load_pipeline", fake_load)

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "examples/local-real-agents-kimi-smoke.yaml"
    assert captured["load_calls"] == 1
    assert captured["submitted_pipeline"] is captured["loaded_pipelines"][0]
    assert len(loaded_pipelines) == 1


def test_check_local_defaults_to_json_when_not_tty(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="check-local-auto-json")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"}],
        "pipeline": {
            "auto_preflight_scope": "run/smoke",
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            }
        },
    }
    assert json.loads(result.stdout) == {"id": "check-local-auto-json", "status": "completed"}


def test_check_local_auto_uses_json_when_stdout_is_redirected_but_stderr_is_tty(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="check-local-auto-mixed-streams")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: err)
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"}],
        "pipeline": {
            "auto_preflight_scope": "run/smoke",
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            }
        },
    }
    assert json.loads(result.stdout) == {"id": "check-local-auto-mixed-streams", "status": "completed"}


def test_check_local_defaults_to_summary_on_tty(monkeypatch):
    _mock_local_readiness_info(monkeypatch)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="check-local-auto-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline run/smoke auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
    )
    assert "Run check-local-auto-summary: completed" in result.stdout


def test_check_local_uses_json_doctor_output_when_run_output_is_json(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="check-local-json")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["check-local", "--output", "json", "--shell-bridge"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"}],
        "pipeline": {
            "auto_preflight_scope": "run/smoke",
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            },
        },
        "shell_bridge": _shell_bridge_recommendation().as_dict(),
    }
    assert json.loads(result.stdout) == {"id": "check-local-json", "status": "completed"}


def test_check_local_uses_json_summary_doctor_output_when_run_output_is_json_summary(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="check-local-json-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(
                run_id,
                pipeline_name="local-real-agents-kimi-smoke",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model="gpt-5-codex",
                        provider=None,
                    )
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    )
                },
            )

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[
                DoctorCheck(
                    name="bash_login_startup",
                    status="warning",
                    detail="missing bridge",
                    context=_bash_startup_context("~/.profile -> ~/.bashrc"),
                ),
                DoctorCheck(
                    name="kimi_shell_helper",
                    status="ok",
                    detail="ready",
                    context={"path": "/tmp/kimi"},
                ),
            ],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["check-local", "--output", "json-summary", "--shell-bridge"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "counts": {"ok": 1, "warning": 1, "failed": 0},
        "checks": [
            {
                "name": "bash_login_startup",
                "status": "warning",
                "detail": "missing bridge",
                "startup_summary": "~/.profile -> ~/.bashrc",
            },
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        ],
        "pipeline": {
            "auto_preflight_scope": "run/smoke",
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            }
        },
        "shell_bridge": _shell_bridge_recommendation().as_dict(),
    }
    assert json.loads(result.stdout) == {
        "id": "check-local-json-summary",
        "status": "completed",
        "pipeline": {"name": "local-real-agents-kimi-smoke"},
        "started_at": "2026-03-08T04:11:03+00:00",
        "finished_at": "2026-03-08T04:11:10+00:00",
        "duration": "7.0s",
        "duration_seconds": 7.0,
        "run_dir": ".agentflow/runs/check-local-json-summary",
        "nodes": [
            {
                "id": "codex_plan",
                "status": "completed",
                "agent": "codex",
                "model": "gpt-5-codex",
                "attempts": 1,
                "exit_code": 0,
                "preview": "codex ok",
            }
        ],
    }


def test_check_local_stops_when_doctor_fails(monkeypatch):
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())

    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader({}, object()))
    monkeypatch.setattr(agentflow.cli, "_build_runtime", lambda runs_dir, max_concurrent_runs: (_ for _ in ()).throw(AssertionError("runtime should not build")))

    result = runner.invoke(app, ["check-local"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Doctor: failed\n"
        "- kimi_shell_helper: failed - missing\n"
        "Pipeline run/smoke auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
    )
    assert result.stdout == ""


def test_doctor_custom_non_kimi_pipeline_skips_bundled_smoke_prereqs(monkeypatch):
    fake_pipeline = SimpleNamespace(nodes=[])

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing kimi"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)

    result = runner.invoke(app, ["doctor", "examples/custom.yaml", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [],
        "pipeline": {
            "auto_preflight": {
                "enabled": False,
                "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
                "matches": [],
                "match_summary": [],
            }
        },
    }


def test_check_local_custom_non_kimi_pipeline_skips_bundled_smoke_prereqs(monkeypatch):
    captured: dict[str, object] = {}
    fake_pipeline = SimpleNamespace(nodes=[])
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-custom")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            return _completed_run(run_id, pipeline_name="custom")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing kimi"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )

    result = runner.invoke(app, ["check-local", "examples/custom.yaml"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "Pipeline run/smoke auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )
    assert "Run check-local-custom: completed" in result.stdout
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "check-local-custom"


def test_check_local_custom_kimi_pipeline_reports_successful_local_agent_probes(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-custom-kimi")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            return _completed_run(run_id, pipeline_name="custom-kimi")

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            ),
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )

    result = runner.invoke(app, ["check-local", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline run/smoke auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline run/smoke auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )
    assert "Run check-local-custom-kimi: completed" in result.stdout
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "check-local-custom-kimi"


def test_smoke_runs_when_bundled_preflight_warns(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-warning")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="claude", status="warning", detail="bootstrap-only")],
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "Run smoke-warning: completed" in result.stdout
    assert result.stderr == (
        "Doctor: warning\n"
        "- claude: warning - bootstrap-only\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
    )
    assert captured["loaded_path"] == "examples/local-real-agents-kimi-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-warning"
    assert captured["wait_timeout"] is None


def test_smoke_keeps_expected_launch_env_override_out_of_preflight_warning_state(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "bundled-smoke.json"
    pipeline_path.write_text(
        json.dumps({"name": 'bundled-smoke', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-launch-env-warning")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="bundled-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: str(pipeline_path))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "Run smoke-launch-env-warning: completed" in result.stdout
    assert result.stderr == ""
    assert getattr(captured["submitted_pipeline"], "name", None) == "bundled-smoke"


def test_smoke_show_preflight_prints_successful_summary_to_stderr(monkeypatch):
    captured: dict[str, object] = {}
    _mock_local_readiness_info(monkeypatch)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-show-preflight")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "--show-preflight"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )
    assert "Run smoke-show-preflight: completed" in result.stdout
    assert captured["loaded_path"] == "examples/local-real-agents-kimi-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-show-preflight"
    assert captured["wait_timeout"] is None


def test_smoke_show_preflight_keeps_json_stdout_machine_readable(monkeypatch):
    _mock_local_readiness_info(monkeypatch)

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-show-preflight-json")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(
        agentflow.cli,
        "_load_pipeline",
        lambda path: SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    id="codex_plan",
                    agent=SimpleNamespace(value="codex"),
                    target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
                ),
                SimpleNamespace(
                    id="claude_review",
                    agent=SimpleNamespace(value="claude"),
                    target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
                ),
            ]
        ),
    )

    result = runner.invoke(app, ["smoke", "--output", "json", "--show-preflight"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )
    assert json.loads(result.stdout) == {"id": "smoke-show-preflight-json", "status": "completed"}


def test_smoke_warn_preflight_includes_shell_bridge_summary_when_available(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-warning-shell-bridge")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: warning\n"
        "- bash_login_startup: warning - missing bridge\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Shell bridge suggestion for `~/.bash_profile` from `~/.profile`:\n"
        "Reason: Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.\n"
        "if [ -f \"$HOME/.profile\" ]; then\n"
        "  . \"$HOME/.profile\"\n"
        "fi\n"
    )


def test_smoke_warn_preflight_honors_json_output(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-warning-json")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="claude", status="warning", detail="bootstrap-only")],
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["smoke", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "claude", "status": "warning", "detail": "bootstrap-only"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            }
        },
    }


def test_run_auto_warn_preflight_uses_json_when_stdout_is_redirected_but_stderr_is_tty(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="run-warning-mixed-streams")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            return _completed_run(run_id, pipeline_name="custom-kimi-run")

    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: err)
    _reject_bundled_smoke_doctor(monkeypatch)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env=None,
                target=SimpleNamespace(kind="local", shell="bash", shell_interactive=True, shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["run", "custom-run.yaml"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
                "matches": [
                    {
                        "node_id": "codex_plan",
                        "agent": "codex",
                        "trigger": "target.shell_init",
                    }
                ],
                "match_summary": ["codex_plan (codex) via `target.shell_init`"],
            }
        },
    }
    assert json.loads(result.stdout) == {"id": "run-warning-mixed-streams", "status": "completed"}
    assert captured["loaded_path"] == "custom-run.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "run-warning-mixed-streams"


def test_smoke_warn_preflight_includes_shell_bridge_json_when_available(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-warning-json-shell-bridge")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge")],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["smoke", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "checks": [{"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [],
                "match_summary": [],
            }
        },
        "shell_bridge": _shell_bridge_recommendation().as_dict(),
    }


def test_smoke_skips_preflight_for_custom_pipeline_in_auto_mode(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: (_ for _ in ()).throw(AssertionError("doctor should not run")))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert "Run smoke-custom: completed" in result.stdout
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom"
    assert captured["wait_timeout"] is None


def test_smoke_auto_runs_preflight_for_custom_pipeline_with_kimi_shell_init(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-kimi-shell-init")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-smoke")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-kimi-shell-init"
    assert captured["wait_timeout"] is None


def test_smoke_show_preflight_reports_pipeline_specific_readiness_for_custom_kimi_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-show-preflight")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-smoke")

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", lambda: _custom_kimi_preflight_report())
    monkeypatch.setattr("agentflow.doctor.subprocess.run", _codex_ready_and_auth_subprocess(_CODEX_AUTH_VIA_API_KEY_AND_LOGIN_STATUS_EXIT_CODE))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_interactive=True, shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml", "--output", "summary", "--show-preflight"])

    assert result.exit_code == 0
    assert result.stderr == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; "
        "`codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via "
        "`OPENAI_API_KEY` + `codex login status`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )
    assert "Run smoke-custom-show-preflight: completed" in result.stdout
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-show-preflight"
    assert captured["wait_timeout"] is None


def test_smoke_failed_preflight_for_custom_kimi_pipeline_includes_shell_bridge_when_available(monkeypatch):
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("bundled smoke doctor should not run for custom kimi preflight")),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[
                DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge"),
                DoctorCheck(name="kimi_shell_helper", status="failed", detail="missing"),
            ],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=True),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: fake_pipeline)

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 1
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: warning - missing bridge\n"
        "- kimi_shell_helper: failed - missing\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: review (claude) via `target.shell_init`\n"
        "Shell bridge suggestion for `~/.bash_profile` from `~/.profile`:\n"
        "Reason: Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.\n"
        "if [ -f \"$HOME/.profile\" ]; then\n"
        "  . \"$HOME/.profile\"\n"
        "fi\n"
    )


def test_smoke_auto_runs_preflight_for_custom_claude_kimi_provider_without_host_anthropic_key(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-claude-kimi-provider")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-claude-kimi-provider-smoke")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=True),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-claude-kimi-provider"
    assert captured["wait_timeout"] is None


def test_smoke_auto_runs_preflight_for_custom_pipeline_with_kimi_agent(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-kimi-agent")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-agent-smoke")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="kimi"),
                env={"KIMI_API_KEY": "inline-secret"},
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-kimi-agent"
    assert captured["wait_timeout"] is None


def test_smoke_auto_runs_preflight_for_custom_pipeline_with_explicit_kimi_shell_wrapper(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-kimi-wrapper")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-shell-wrapper")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash -lic 'kimi && {command}'", shell_init=None),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-kimi-wrapper"
    assert captured["wait_timeout"] is None


def test_smoke_auto_runs_preflight_for_custom_pipeline_with_backtick_eval_kimi_shell_wrapper(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-custom-kimi-backtick-wrapper")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="custom-kimi-backtick-shell-wrapper")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return _custom_kimi_preflight_report()

    _reject_bundled_smoke_doctor(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash -lic 'eval `kimi` && {command}'", shell_init=None),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "custom-smoke.yaml"])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-custom-kimi-backtick-wrapper"
    assert captured["wait_timeout"] is None


def test_smoke_runs_preflight_for_explicit_bundled_pipeline_path(monkeypatch):
    captured: dict[str, object] = {}
    doctor_calls = 0

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-explicit-default")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    def fake_doctor_report():
        nonlocal doctor_calls
        doctor_calls += 1
        return DoctorReport(
            status="warning",
            checks=[DoctorCheck(name="claude", status="warning", detail="bootstrap-only")],
        )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", fake_doctor_report)
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", bundled_path])

    assert result.exit_code == 0
    assert doctor_calls == 1
    assert result.stderr == (
        "Doctor: warning\n"
        "- claude: warning - bootstrap-only\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
    )
    assert captured["loaded_path"] == bundled_path
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-explicit-default"
    assert captured["wait_timeout"] is None


@pytest.mark.parametrize("command", ["run", "smoke"])
def test_run_and_smoke_force_preflight_for_custom_pipeline_use_pipeline_specific_checks(monkeypatch, command):
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell=None,
                    shell_login=False,
                    shell_interactive=False,
                    shell_init=None,
                    cwd=None,
                ),
            )
        ]
    )
    captured: dict[str, object] = {}
    codex_readiness_pipelines: list[object] = []

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("bundled smoke doctor should not run for custom forced preflight")),
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_codex_readiness_checks",
        lambda pipeline: codex_readiness_pipelines.append(pipeline) or [],
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda pipeline, runs_dir, max_concurrent_runs, output: captured.setdefault("submitted_pipeline", pipeline),
    )

    result = runner.invoke(app, [command, "custom-smoke.yaml", "--preflight", "always"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert codex_readiness_pipelines == [fake_pipeline]


def test_smoke_can_disable_bundled_preflight(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="smoke-no-preflight")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            captured["wait_timeout"] = timeout
            return _completed_run(run_id, pipeline_name="local-real-agents-kimi-smoke")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: (_ for _ in ()).throw(AssertionError("doctor should not run")))
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    fake_pipeline = object()
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["smoke", "--preflight", "never"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "examples/local-real-agents-kimi-smoke.yaml"
    assert captured["submitted_pipeline"] is fake_pipeline
    assert captured["wait_run_id"] == "smoke-no-preflight"
    assert captured["wait_timeout"] is None


def test_smoke_supports_json_output(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-json")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(run_id)

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["smoke", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"id": "smoke-json", "status": "completed"}


def test_smoke_supports_json_summary_output(monkeypatch):
    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            return SimpleNamespace(id="smoke-json-summary")

        async def wait(self, run_id: str, timeout: float | None = None):
            return _completed_run(
                run_id,
                pipeline_name="local-real-agents-kimi-smoke",
                pipeline_nodes=[
                    SimpleNamespace(
                        id="codex_plan",
                        agent=SimpleNamespace(value="codex"),
                        model=None,
                        provider=None,
                    )
                ],
                nodes={
                    "codex_plan": SimpleNamespace(
                        status=SimpleNamespace(value="completed"),
                        current_attempt=1,
                        attempts=[SimpleNamespace(number=1)],
                        exit_code=0,
                        final_response="codex ok",
                        output="codex ok",
                        stderr_lines=[],
                    )
                },
            )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id), FakeOrchestrator()),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: object())

    result = runner.invoke(app, ["smoke", "--output", "json-summary"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "id": "smoke-json-summary",
        "status": "completed",
        "pipeline": {"name": "local-real-agents-kimi-smoke"},
        "started_at": "2026-03-08T04:11:03+00:00",
        "finished_at": "2026-03-08T04:11:10+00:00",
        "duration": "7.0s",
        "duration_seconds": 7.0,
        "run_dir": ".agentflow/runs/smoke-json-summary",
        "nodes": [
            {
                "id": "codex_plan",
                "status": "completed",
                "agent": "codex",
                "attempts": 1,
                "exit_code": 0,
                "preview": "codex ok",
            }
        ],
    }


def test_doctor_outputs_json_report(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])

    result = runner.invoke(app, ["doctor", "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [{"name": "kimi_shell_helper", "status": "ok", "detail": "ready"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                    {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
                ],
                "match_summary": [
                    "codex_plan (codex) via `target.bootstrap`",
                    "claude_review (claude) via `target.bootstrap`",
                ],
            }
        },
    }


def test_doctor_outputs_json_summary_report(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="warning",
            checks=[
                DoctorCheck(
                    name="bash_login_startup",
                    status="warning",
                    detail="missing bridge",
                    context=_bash_startup_context("~/.profile -> ~/.bashrc"),
                ),
                DoctorCheck(
                    name="kimi_shell_helper",
                    status="ok",
                    detail="ready",
                    context={"path": "/tmp/kimi"},
                ),
            ],
        ),
    )

    result = runner.invoke(app, ["doctor", "--output", "json-summary"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "warning",
        "counts": {"ok": 1, "warning": 1, "failed": 0},
        "checks": [
            {
                "name": "bash_login_startup",
                "status": "warning",
                "detail": "missing bridge",
                "startup_summary": "~/.profile -> ~/.bashrc",
            },
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                    {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
                ],
                "match_summary": [
                    "codex_plan (codex) via `target.bootstrap`",
                    "claude_review (claude) via `target.bootstrap`",
                ],
            }
        },
    }


def test_doctor_defaults_to_json_report(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [{"name": "kimi_shell_helper", "status": "ok", "detail": "ready"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                    {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
                ],
                "match_summary": [
                    "codex_plan (codex) via `target.bootstrap`",
                    "claude_review (claude) via `target.bootstrap`",
                ],
            }
        },
    }


def test_doctor_defaults_to_summary_report_on_tty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.bootstrap`, claude_review (claude) via `target.bootstrap`\n"
    )


def test_doctor_supports_summary_output(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])

    result = runner.invoke(app, ["doctor", "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.bootstrap`, claude_review (claude) via `target.bootstrap`\n"
    )


def test_doctor_without_path_includes_bundled_local_readiness_info(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _mock_local_readiness_info(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    bundled_path = str((Path.cwd() / "examples/local-real-agents-kimi-smoke.yaml").resolve())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: bundled_path)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_login=True, shell_interactive=True, shell_init="kimi"),
            ),
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))
    monkeypatch.setattr(agentflow.cli, "_stream_supports_tty_summary", lambda *, err: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == bundled_path
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_augments_report_for_kimi_shell_bootstrap_warning(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=False),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: warning\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- kimi_shell_bootstrap: warning - Node `claude_review`: `shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_augments_report_for_kimi_shell_bootstrap_noprofile_warning(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(
                    kind="local",
                    shell="bash --noprofile -lic '{command}'",
                    shell_init="kimi",
                    shell_interactive=False,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: warning\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- kimi_shell_bootstrap: warning - Node `claude_review`: `shell_init: kimi` uses bash with `--noprofile`, so login startup files never reach `~/.bashrc`. Remove `--noprofile`, source the helper explicitly, or export provider variables directly.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_fails_for_non_bash_kimi_shell_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                target=SimpleNamespace(kind="local", shell="sh", shell_init="kimi", shell_interactive=False),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- kimi_shell_bootstrap: failed - Node `claude_review`: `shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, use `bash -lic`, or export provider variables directly.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_augments_report_for_kimi_agent_bootstrap_warning(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="kimi_review",
                agent=SimpleNamespace(value="kimi"),
                env={"KIMI_API_KEY": "inline-secret"},
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=False),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: warning\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- kimi_shell_bootstrap: warning - Node `kimi_review`: `shell_init: kimi` uses bash without interactive startup; helpers from `~/.bashrc` are usually unavailable. Set `target.shell_interactive: true` or use `bash -lic`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: kimi_review (kimi) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_fails_when_kimi_node_is_missing_kimi_api_key(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="kimi_review",
                agent=SimpleNamespace(value="kimi"),
                provider=None,
                env={},
                target=SimpleNamespace(kind="local"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `kimi_review` (kimi) requires `KIMI_API_KEY` for provider `moonshot`, but it is not set in the current environment, `node.env`, or `provider.env`.\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: kimi_review (kimi) via `agent`\n"
    )


def test_doctor_with_pipeline_path_accepts_claude_kimi_provider_credentials_from_kimi_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=True),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_claude_anthropic_provider_credentials_from_kimi_bootstrap(
    tmp_path,
    monkeypatch,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-anthropic-provider-kimi-bootstrap', "working_dir": '.', "nodes": [{"id": 'claude_review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_init": 'kimi', "shell_interactive": True}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- bootstrap_env_override: ok - Node `claude_review`: Local shell bootstrap overrides launch `ANTHROPIC_BASE_URL` from `https://api.anthropic.com` to `https://api.kimi.com/coding/` via `target.shell_init` (`kimi` helper).\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_json_includes_kimi_bootstrap_base_url_override_for_anthropic_provider(
    tmp_path,
    monkeypatch,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-anthropic-provider-kimi-bootstrap-json', "working_dir": '.', "nodes": [{"id": 'claude_review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_init": 'kimi', "shell_interactive": True}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [
            {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
            {
                "name": "claude_ready",
                "status": "ok",
                "detail": "Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
            },
            {
                "name": "bootstrap_env_override",
                "status": "ok",
                "detail": "Node `claude_review`: Local shell bootstrap overrides launch `ANTHROPIC_BASE_URL` from `https://api.anthropic.com` to `https://api.kimi.com/coding/` via `target.shell_init` (`kimi` helper).",
                "context": {
                    "node_id": "claude_review",
                    "key": "ANTHROPIC_BASE_URL",
                    "current_value": "https://api.anthropic.com",
                    "bootstrap_value": "https://api.kimi.com/coding/",
                    "origin": "launch_env",
                    "source": "target.shell_init",
                    "helper": "kimi",
                },
            },
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
                "matches": [
                    {
                        "node_id": "claude_review",
                        "agent": "claude",
                        "trigger": "target.shell_init",
                    }
                ],
                "match_summary": ["claude_review (claude) via `target.shell_init`"],
            }
        },
    }


def test_doctor_with_pipeline_path_fails_when_local_claude_is_unavailable_after_shell_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess(returncode=1))
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: failed - Node `claude_review` (claude) cannot launch local Claude after the node shell bootstrap; `claude --version` fails in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_fails_when_local_codex_auth_is_unavailable(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_run(*args, **kwargs):
        env = kwargs.get("env") or {}
        target_command = env.get("AGENTFLOW_TARGET_COMMAND", "")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1 if "subprocess.run" in target_command and "OPENAI_API_KEY" in target_command else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_auth: failed - Node `codex_plan` (codex) cannot authenticate local Codex after the node shell bootstrap; `codex login status` fails and `OPENAI_API_KEY` is not set in the current environment, `node.env`, or `provider.env`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_local_codex_login_status_from_shell_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_local_codex_login_for_explicit_openai_provider(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider="openai",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=False,
                    shell_interactive=False,
                    shell_init=None,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "explicit-openai.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "explicit-openai.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_uses_codex_auth_failure_for_explicit_openai_provider(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        command = args[0]
        if isinstance(command, list):
            target_command = " ".join(str(part) for part in command)
        else:
            target_command = str(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1 if "subprocess.run" in target_command and "OPENAI_API_KEY" in target_command else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider="openai",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=False,
                    shell_interactive=False,
                    shell_init=None,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "explicit-openai.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "explicit-openai.yaml"
    assert "provider_credentials" not in result.stdout
    assert result.stdout == (
        "Doctor: failed\n"
        "- codex_auth: failed - Node `codex_plan` (codex) cannot authenticate local Codex after the node shell bootstrap; `codex login status` fails and `OPENAI_API_KEY` is not set in the current environment, `node.env`, or `provider.env`.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_custom_kimi_pipeline_reports_successful_local_agent_probes(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="codex_plan",
                agent=SimpleNamespace(value="codex"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            ),
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="kimi",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            ),
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- codex_ready: ok - Node `codex_plan` (codex) can launch local Codex after the node shell bootstrap; `codex --version` succeeds in the prepared local shell.\n"
        "- codex_auth: ok - Node `codex_plan` (codex) can authenticate local Codex after the node shell bootstrap via `codex login status` or `OPENAI_API_KEY`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.shell_init`, claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_shell_init_export(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_init=["export ANTHROPIC_API_KEY=test-shell-key"],
                    shell_interactive=True,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_split_shell_init_export(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_init=["ANTHROPIC_API_KEY=test-shell-key", "export ANTHROPIC_API_KEY"],
                    shell_interactive=True,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_custom_kimi_provider_credentials_from_shell_init_helper(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider=ProviderConfig(
                    name="kimi-proxy",
                    base_url="https://api.kimi.com/coding/",
                    api_key_env="ANTHROPIC_API_KEY",
                    env={},
                ),
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_init="kimi",
                    shell_login=True,
                    shell_interactive=True,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_custom_kimi_provider_env_base_url_credentials_from_shell_init_helper(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider=ProviderConfig(
                    name="kimi-proxy",
                    api_key_env="ANTHROPIC_API_KEY",
                    env={"ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/"},
                ),
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_init="kimi",
                    shell_login=True,
                    shell_interactive=True,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_shell_wrapper_export(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash -lc 'export ANTHROPIC_API_KEY=test-shell-key && {command}'",
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_shell_prefix_env_wrapper(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="env ANTHROPIC_API_KEY=test-shell-key bash -c",
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_split_shell_wrapper_export(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell="bash -lc 'ANTHROPIC_API_KEY=test-shell-key && export ANTHROPIC_API_KEY && {command}'",
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_sourced_shell_wrapper(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    home = tmp_path / "home"
    home.mkdir()
    (home / ".anthropic.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell=f"env HOME={home} bash -lc 'source ~/.anthropic.env && {{command}}'",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_bash_env_file(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                target=SimpleNamespace(
                    kind="local",
                    shell=f"env HOME={home} BASH_ENV=$HOME/auth.env bash -c '{{command}}'",
                    cwd=None,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_rejects_provider_credentials_from_interactive_bash_env_file(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell=f"env HOME={home} BASH_ENV=$HOME/auth.env bash -ic '{{command}}'",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout.startswith(
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `claude_review` (claude) requires `ANTHROPIC_API_KEY` for provider "
        "`anthropic`, but it is not set in the current environment, `node.env`, or `provider.env`.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_node_env_bash_env_file(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.env").write_text("export ANTHROPIC_API_KEY=test-shell-key\n", encoding="utf-8")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={"BASH_ENV": "$HOME/auth.env"},
                target=SimpleNamespace(
                    kind="local",
                    shell=f"env HOME={home} bash -c '{{command}}'",
                    cwd=None,
                ),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_login_shell_startup(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_requires_default_claude_credentials_without_explicit_provider(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider=None,
                env={},
                executable=None,
                target=SimpleNamespace(kind="local"),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout.startswith(
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `claude_review` (claude) requires `ANTHROPIC_API_KEY` for provider "
        "`anthropic`, but it is not set in the current environment, `node.env`, or `provider.env`.\n"
    )


def test_doctor_with_pipeline_path_fails_when_node_env_clears_current_provider_api_key(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={"ANTHROPIC_API_KEY": ""},
                executable=None,
                target=SimpleNamespace(kind="local"),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `claude_review` (claude) requires `ANTHROPIC_API_KEY` for provider "
        "`anthropic`, but the launch env clears the current environment value via `node.env`.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local "
        "Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_fails_when_provider_env_clears_current_provider_api_key(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider=ProviderConfig(
                    name="anthropic",
                    base_url="https://api.anthropic.com",
                    api_key_env="ANTHROPIC_API_KEY",
                    env={"ANTHROPIC_API_KEY": ""},
                ),
                env={},
                executable=None,
                target=SimpleNamespace(kind="local"),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `claude_review` (claude) requires `ANTHROPIC_API_KEY` for provider "
        "`anthropic`, but the launch env clears the current environment value via `provider.env`.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local "
        "Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_fails_when_shell_wrapper_clears_current_provider_api_key(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell=f"env -i PATH={os.environ.get('PATH', '/usr/bin:/bin')} bash",
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- provider_credentials: failed - Node `claude_review` (claude) requires `ANTHROPIC_API_KEY` for provider "
        "`anthropic`, but `target.shell` overrides or clears the current environment value before launch.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local "
        "Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_reports_when_launch_env_clears_current_provider_api_key_before_kimi_bootstrap(
    tmp_path,
    monkeypatch,
):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-kimi-bootstrap-key-clear', "working_dir": '.', "nodes": [{"id": 'claude_review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "env": {"ANTHROPIC_API_KEY": ''}, "target": {"kind": 'local', "shell": 'bash', "shell_init": 'kimi', "shell_interactive": True}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `claude_review` (claude) can launch local Claude after the node shell "
        "bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- launch_env_override: ok - Node `claude_review`: Launch env clears current `ANTHROPIC_API_KEY` for this "
        "node via `node.env`.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: claude_review (claude) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_warns_when_login_startup_provider_probe_times_out(monkeypatch):
    captured: dict[str, object] = {}

    def _timeout(*args, **kwargs):
        command = list(args[0])
        if command == ["bash", "-lc", 'test -n "${ANTHROPIC_API_KEY:-}"']:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _timeout)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: warning\n"
        "- provider_credentials_probe: warning - Node `claude_review` (claude) could not confirm `ANTHROPIC_API_KEY` "
        "for provider `anthropic` from local bash startup because the probe timed out after 5s. Fix the shell startup "
        "or increase `AGENTFLOW_BASH_STARTUP_PROBE_TIMEOUT_SECONDS`.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_uses_target_cwd_for_relative_login_startup_sources(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text('if [ -f .bashrc ]; then . .bashrc; fi\n', encoding="utf-8")
    (home / ".bashrc").write_text("export ANTHROPIC_API_KEY=from-relative-bashrc\n", encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-relative-login-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "cwd": str(home)}}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "provider_credentials" not in result.stdout
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_login_shell_startup_via_node_env_home(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()
    observed_commands: list[list[str]] = []
    observed_envs: list[dict[str, str]] = []

    def _run(*args, **kwargs):
        command = list(args[0])
        observed_commands.append(command)
        observed_envs.append(dict(kwargs.get("env") or {}))
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0 if kwargs.get("env", {}).get("HOME") == str(launch_home) else 1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={"HOME": str(launch_home)},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert ["bash", "-lc", 'test -n "${ANTHROPIC_API_KEY:-}"'] in observed_commands
    assert any(env.get("HOME") == str(launch_home) for env in observed_envs)
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_login_startup_env_bridge(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "home"
    home.mkdir()
    auth_file = tmp_path / "anthropic.env"
    auth_file.write_text("export ANTHROPIC_API_KEY=from-launch-env-file\n", encoding="utf-8")
    (home / ".profile").write_text(
        'if [ -n "${AGENTFLOW_KIMI_ENV_FILE:-}" ]; then . "$AGENTFLOW_KIMI_ENV_FILE"; fi\n',
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-env-gated-login-startup', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "env": {"AGENTFLOW_KIMI_ENV_FILE": str(auth_file)}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    real_subprocess_run = subprocess.run

    def _run(*args, **kwargs):
        command = list(args[0])
        if command == ["bash", "-lc", 'test -n "${ANTHROPIC_API_KEY:-}"']:
            return real_subprocess_run(*args, **kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "provider_credentials" not in result.stdout
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_shell_wrapper_login_bridge(monkeypatch):
    captured: dict[str, object] = {}
    observed_commands: list[list[str]] = []
    observed_envs: list[dict[str, str]] = []

    def _run(*args, **kwargs):
        command = list(args[0])
        env = dict(kwargs.get("env") or {})
        observed_commands.append(command)
        observed_envs.append(env)
        if command == ["/opt/custom/bash", "-lc", 'test -n "${ANTHROPIC_API_KEY:-}"']:
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=0 if env.get("AGENTFLOW_KIMI_ENV_FILE") == "/tmp/from-shell-wrapper" else 1,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="env AGENTFLOW_KIMI_ENV_FILE=/tmp/from-shell-wrapper /opt/custom/bash",
                    shell_login=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert ["/opt/custom/bash", "-lc", 'test -n "${ANTHROPIC_API_KEY:-}"'] in observed_commands
    assert any(env.get("AGENTFLOW_KIMI_ENV_FILE") == "/tmp/from-shell-wrapper" for env in observed_envs)
    assert "provider_credentials" not in result.stdout
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_interactive_login_shell_startup(monkeypatch):
    captured: dict[str, object] = {}
    observed_commands: list[list[str]] = []

    def _run(*args, **kwargs):
        command = list(args[0])
        observed_commands.append(command)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0 if command[:2] == ["bash", "-lic"] or command[:4] == ["bash", "-l", "-i", "-c"] else 1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert ["bash", "-lic", 'test -n "${ANTHROPIC_API_KEY:-}"'] in observed_commands
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_provider_credentials_from_interactive_shell_startup(monkeypatch):
    captured: dict[str, object] = {}
    observed_commands: list[list[str]] = []

    def _run(*args, **kwargs):
        command = list(args[0])
        observed_commands.append(command)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0 if command[:2] == ["bash", "-ic"] or command[:3] == ["bash", "-i", "-c"] else 1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="claude_review",
                agent=SimpleNamespace(value="claude"),
                provider="anthropic",
                env={},
                executable=None,
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_interactive=True,
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert ["bash", "-ic", 'test -n "${ANTHROPIC_API_KEY:-}"'] in observed_commands
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_accepts_kimi_api_key_from_node_env(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="kimi_review",
                agent=SimpleNamespace(value="kimi"),
                provider=None,
                env={"KIMI_API_KEY": "inline-secret"},
                target=SimpleNamespace(kind="local"),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_ready: ok - Node `kimi_review` (kimi) found the Kimi CLI after the node shell bootstrap; "
        "`kimi --version` succeeds in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: kimi_review (kimi) via `agent`\n"
    )


def test_doctor_with_pipeline_path_fails_when_kimi_cli_is_unavailable(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess(returncode=1))
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="kimi_review",
                agent=SimpleNamespace(value="kimi"),
                provider=None,
                env={"KIMI_API_KEY": "inline-secret"},
                executable="python-kimi",
                target=SimpleNamespace(
                    kind="local",
                    shell="bash",
                    shell_login=True,
                    shell_interactive=True,
                    shell_init="kimi",
                    cwd=None,
                ),
            )
        ],
        working_path=Path.cwd(),
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "summary"])

    assert result.exit_code == 1
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- kimi_ready: failed - Node `kimi_review` (kimi) cannot find the Kimi CLI after the node shell bootstrap; `python-kimi --version` fails in the prepared local shell.\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: kimi_review (kimi) via `target.shell_init`\n"
    )


def test_doctor_with_pipeline_path_supports_json_output(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    fake_pipeline = SimpleNamespace(nodes=[])
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "json"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [],
        "pipeline": {
            "auto_preflight": {
                "enabled": False,
                "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
                "matches": [],
                "match_summary": [],
            }
        },
    }


def test_doctor_with_pipeline_path_supports_json_summary_output(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    fake_pipeline = SimpleNamespace(nodes=[])
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "json-summary"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert json.loads(result.stdout) == {
        "status": "ok",
        "counts": {"ok": 0, "warning": 0, "failed": 0},
        "checks": [],
        "pipeline": {
            "auto_preflight": {
                "enabled": False,
                "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
                "matches": [],
                "match_summary": [],
            }
        },
    }


def test_doctor_with_pipeline_path_reports_auto_preflight_metadata_in_json(monkeypatch):
    captured: dict[str, object] = {}

    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _completed_subprocess())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    fake_pipeline = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="review",
                agent=SimpleNamespace(value="claude"),
                target=SimpleNamespace(kind="local", shell="bash", shell_init="kimi", shell_interactive=True),
            )
        ]
    )
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", _capture_pipeline_loader(captured, fake_pipeline))

    result = runner.invoke(app, ["doctor", "custom-smoke.yaml", "--output", "json"])

    assert result.exit_code == 0
    assert captured["loaded_path"] == "custom-smoke.yaml"
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [
            {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
            {
                "name": "claude_ready",
                "status": "ok",
                "detail": "Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
            },
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.",
                "matches": [
                    {
                        "node_id": "review",
                        "agent": "claude",
                        "trigger": "target.shell_init",
                    }
                ],
                "match_summary": ["review (claude) via `target.shell_init`"],
            }
        },
    }


def test_doctor_with_pipeline_path_reports_expected_launch_env_override_as_ok(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-base-url-override', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout.startswith(
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- launch_env_override: ok - Node `review`: Launch env uses configured `ANTHROPIC_BASE_URL` value `https://api.kimi.com/coding/` instead of current `https://open.bigmodel.cn/api/anthropic` via `provider.base_url`.\n"
    )


def test_doctor_with_pipeline_path_reports_bootstrap_auth_override_as_ok(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-bootstrap-auth-override', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout.startswith(
        "Doctor: ok\n"
        "- bash_login_startup: ok - startup ready\n"
        "- kimi_shell_helper: ok - ready\n"
        "- claude_ready: ok - Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.\n"
        "- launch_env_override: ok - Node `review`: Launch env uses configured `ANTHROPIC_BASE_URL` value `https://api.kimi.com/coding/` instead of current `https://open.bigmodel.cn/api/anthropic` via `provider.base_url`.\n"
        "- bootstrap_env_override: ok - Node `review`: Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper).\n"
    )


def test_doctor_with_pipeline_path_keeps_bootstrap_auth_override_when_kimi_base_url_already_matches(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-bootstrap-auth-override-kimi-base-url', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "claude_ready",
            "status": "ok",
            "detail": "Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
        },
        {
            "name": "bootstrap_env_override",
            "status": "ok",
            "detail": "Node `review`: Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper).",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_API_KEY",
                "redacted": True,
                "source": "target.bootstrap",
                "helper": "kimi",
            },
        },
    ]


def test_doctor_with_pipeline_path_json_includes_launch_env_override_context(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-base-url-override-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "claude_ready",
            "status": "ok",
            "detail": "Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
        },
        {
            "name": "launch_env_override",
            "status": "ok",
            "detail": "Node `review`: Launch env uses configured `ANTHROPIC_BASE_URL` value `https://api.kimi.com/coding/` instead of current `https://open.bigmodel.cn/api/anthropic` via `provider.base_url`.",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_BASE_URL",
                "current_value": "https://open.bigmodel.cn/api/anthropic",
                "launch_value": "https://api.kimi.com/coding/",
                "source": "provider.base_url",
            },
        },
        {
            "name": "bootstrap_env_override",
            "status": "ok",
            "detail": "Node `review`: Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.shell_init` (`kimi` helper).",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_API_KEY",
                "redacted": True,
                "source": "target.shell_init",
                "helper": "kimi",
            },
        },
    ]


def test_doctor_with_pipeline_path_json_includes_bootstrap_auth_override_context(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-bootstrap-auth-override-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "claude_ready",
            "status": "ok",
            "detail": "Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
        },
        {
            "name": "launch_env_override",
            "status": "ok",
            "detail": "Node `review`: Launch env uses configured `ANTHROPIC_BASE_URL` value `https://api.kimi.com/coding/` instead of current `https://open.bigmodel.cn/api/anthropic` via `provider.base_url`.",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_BASE_URL",
                "current_value": "https://open.bigmodel.cn/api/anthropic",
                "launch_value": "https://api.kimi.com/coding/",
                "source": "provider.base_url",
            },
        },
        {
            "name": "bootstrap_env_override",
            "status": "ok",
            "detail": "Node `review`: Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper).",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_API_KEY",
                "redacted": True,
                "source": "target.bootstrap",
                "helper": "kimi",
            },
        },
    ]


def test_doctor_with_pipeline_path_reports_bootstrap_auth_override_for_launch_secret(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-bootstrap-auth-override-launch-secret', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "env": {"ANTHROPIC_API_KEY": 'launch-secret'}, "target": {"kind": 'local', "bootstrap": 'kimi'}}]}),
        encoding="utf-8",
    )
    _mock_custom_kimi_preflight(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checks"] == [
        {"name": "bash_login_startup", "status": "ok", "detail": "startup ready"},
        {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
        {
            "name": "claude_ready",
            "status": "ok",
            "detail": "Node `review` (claude) can launch local Claude after the node shell bootstrap; `claude --version` succeeds in the prepared local shell.",
        },
        {
            "name": "bootstrap_env_override",
            "status": "ok",
            "detail": "Node `review`: Local shell bootstrap overrides launch `ANTHROPIC_API_KEY` for this node via `target.bootstrap` (`kimi` helper).",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_API_KEY",
                "redacted": True,
                "origin": "launch_env",
                "source": "target.bootstrap",
                "helper": "kimi",
            },
        },
    ]


def test_doctor_with_custom_kimi_pipeline_uses_pipeline_specific_preflight(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-custom-kimi', "working_dir": '.', "local_target_defaults": {"bootstrap": 'kimi'}, "nodes": [{"id": 'review', "agent": 'claude', "executable": 'custom-claude', "provider": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[
                DoctorCheck(name="codex", status="failed", detail="default codex smoke check should be ignored"),
                DoctorCheck(name="claude", status="failed", detail="default claude smoke check should be ignored"),
            ],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_local_kimi_bootstrap_doctor_report", _doctor_report)
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        "Pipeline auto preflight matches: review (claude) via `target.bootstrap`\n"
    )


def test_doctor_with_local_kimi_provider_pipeline_reports_auto_preflight_enabled(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-provider-kimi-auto-preflight', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_claude_readiness_info_checks",
        lambda pipeline: [DoctorCheck(name="claude_ready", status="ok", detail="ready")],
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- claude_ready: ok - ready\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: review (claude) via `provider`\n"
    )


def test_doctor_with_local_kimi_agent_pipeline_reports_auto_preflight_enabled(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-kimi-agent-auto-preflight', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_kimi_readiness_info_checks",
        lambda pipeline: [DoctorCheck(name="kimi_ready", status="ok", detail="ready")],
    )
    monkeypatch.setenv("KIMI_API_KEY", "super-secret")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_ready: ok - ready\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: review (kimi) via `agent`\n"
    )


def test_run_with_local_kimi_provider_pipeline_auto_preflight_runs_pipeline_checks(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'run-provider-kimi-auto-preflight', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("bundled smoke doctor should not run")),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("kimi bootstrap doctor should not run")),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_claude_readiness_checks",
        lambda pipeline: [DoctorCheck(name="claude_ready", status="failed", detail="missing local claude")],
    )
    monkeypatch.setattr(agentflow.cli, "_run_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run should not start")))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")

    result = runner.invoke(app, ["run", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 1
    assert result.stdout == (
        "Doctor: failed\n"
        "- claude_ready: failed - missing local claude\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: review (claude) via `provider`\n"
    )


def test_run_with_custom_kimi_provider_api_key_env_pipeline_auto_preflight_runs_pipeline_checks(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'run-provider-kimi-custom-key-auto-preflight', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": {"name": 'kimi-proxy', "base_url": 'https://api.kimi.com/coding/', "api_key_env": 'KIMI_PROXY_KEY'}, "prompt": 'hi'}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("bundled smoke doctor should not run")),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: (_ for _ in ()).throw(AssertionError("kimi bootstrap doctor should not run")),
    )
    monkeypatch.setattr(
        agentflow.cli,
        "build_pipeline_local_claude_readiness_checks",
        lambda pipeline: [DoctorCheck(name="claude_ready", status="failed", detail="missing local claude")],
    )
    monkeypatch.setattr(agentflow.cli, "_run_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run should not start")))
    monkeypatch.setenv("KIMI_PROXY_KEY", "super-secret")

    result = runner.invoke(app, ["run", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 1
    assert result.stdout == (
        "Doctor: failed\n"
        "- claude_ready: failed - missing local claude\n"
        "Pipeline auto preflight: enabled - local Kimi-backed nodes require pipeline-specific readiness checks.\n"
        "Pipeline auto preflight matches: review (claude) via `provider`\n"
    )


def test_doctor_with_pipeline_path_warns_when_local_launch_inherits_current_base_url(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: warning\n"
        "- launch_env_inheritance: warning - Node `review`: Launch inherits current `ANTHROPIC_BASE_URL` value `https://open.bigmodel.cn/api/anthropic`; configure `provider` or `node.env` explicitly if you want Claude routing pinned for this node.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_warns_when_explicit_claude_provider_leaves_base_url_inherited(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-explicit-claude-provider-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": {"name": 'anthropic', "api_key_env": 'ANTHROPIC_API_KEY'}, "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: warning\n"
        "- launch_env_inheritance: warning - Node `review`: Launch inherits current `ANTHROPIC_BASE_URL` value `https://open.bigmodel.cn/api/anthropic`; configure `provider` or `node.env` explicitly if you want Claude routing pinned for this node.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_warns_when_explicit_codex_provider_leaves_base_url_inherited(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-explicit-codex-provider-base-url-inheritance', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "provider": {"name": 'openai', "api_key_env": 'OPENAI_API_KEY', "wire_api": 'responses'}, "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://oai-relay.ctf.so/openai")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: warning\n"
        "- launch_env_inheritance: warning - Node `plan`: Launch inherits current `OPENAI_BASE_URL` value `https://oai-relay.ctf.so/openai`; configure `provider` or `node.env` explicitly if you want Codex routing pinned for this node.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_uses_node_home_for_base_url_bootstrap_detection(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()
    (launch_home / ".profile").write_text(
        "export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/\n",
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-home-base-url-bootstrap', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi', "env": {"HOME": str(launch_home)}, "target": {"kind": 'local', "shell": 'bash', "shell_login": True}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_reports_when_node_env_clears_current_base_url(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-base-url-cleared', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"OPENAI_BASE_URL": ''}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_codex_auth_checks", lambda pipeline: [])
    monkeypatch.setenv("OPENAI_BASE_URL", "https://oai-relay.ctf.so/openai")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- launch_env_override: ok - Node `plan`: Launch env clears current `OPENAI_BASE_URL` value `https://oai-relay.ctf.so/openai` via `node.env`.\n"
        "Pipeline auto preflight: disabled - path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.\n"
    )


def test_doctor_with_pipeline_path_json_includes_launch_env_inheritance_context(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-base-url-inheritance-json', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "prompt": 'hi'}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_claude_readiness_checks", lambda pipeline: [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checks"] == [
        {
            "name": "launch_env_inheritance",
            "status": "warning",
            "detail": "Node `review`: Launch inherits current `ANTHROPIC_BASE_URL` value `https://open.bigmodel.cn/api/anthropic`; configure `provider` or `node.env` explicitly if you want Claude routing pinned for this node.",
            "context": {
                "node_id": "review",
                "key": "ANTHROPIC_BASE_URL",
                "current_value": "https://open.bigmodel.cn/api/anthropic",
                "source": "current environment",
            },
        }
    ]


def test_doctor_without_path_reports_bundled_smoke_override_as_ok(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "bundled-smoke.json"
    pipeline_path.write_text(
        json.dumps({"name": 'bundled-smoke', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'bash', "shell_login": True, "shell_interactive": True, "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: str(pipeline_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")

    result = runner.invoke(app, ["doctor", "--output", "summary"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "- launch_env_override: ok - Node `review`: Launch env uses configured `ANTHROPIC_BASE_URL` value `https://api.kimi.com/coding/` instead of current `https://open.bigmodel.cn/api/anthropic` via `provider.base_url`.\n"
        "- bootstrap_env_override: ok - Node `review`: Local shell bootstrap overrides current `ANTHROPIC_API_KEY` for this node via `target.shell_init` (`kimi` helper).\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: review (claude) via `target.shell_init`\n"
    )


def test_doctor_without_path_applies_bundled_pipeline_shell_checks(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "bundled-smoke.json"
    pipeline_path.write_text(
        json.dumps({"name": 'bundled-smoke', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'kimi', "prompt": 'hi', "target": {"kind": 'local', "shell": 'sh', "shell_init": 'kimi'}}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="ok",
            checks=[DoctorCheck(name="kimi_shell_helper", status="ok", detail="ready")],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: str(pipeline_path))

    result = runner.invoke(app, ["doctor", "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
            {
                "name": "kimi_shell_bootstrap",
                "status": "failed",
                "detail": "Node `review`: `shell_init: kimi` requires bash-style shell bootstrap, but `target.shell` resolves to `sh`. Use `shell: bash` with `target.shell_login: true` and `target.shell_interactive: true`, use `bash -lic`, or export provider variables directly.",
            },
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "review", "agent": "claude", "trigger": "target.shell_init"},
                ],
                "match_summary": ["review (claude) via `target.shell_init`"],
            }
        },
    }


def test_doctor_can_include_shell_bridge_in_json_output(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_bash_login_shell_bridge_recommendation",
        lambda: ShellBridgeRecommendation(
            target="~/.bash_profile",
            source="~/.profile",
            snippet='if [ -f "$HOME/.profile" ]; then\n  . "$HOME/.profile"\nfi\n',
            reason="Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.",
        ),
    )

    result = runner.invoke(app, ["doctor", "--output", "json", "--shell-bridge"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "ok",
        "checks": [{"name": "kimi_shell_helper", "status": "ok", "detail": "ready"}],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                    {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
                ],
                "match_summary": [
                    "codex_plan (codex) via `target.bootstrap`",
                    "claude_review (claude) via `target.bootstrap`",
                ],
            }
        },
        "shell_bridge": {
            "target": "~/.bash_profile",
            "source": "~/.profile",
            "snippet": 'if [ -f "$HOME/.profile" ]; then\n  . "$HOME/.profile"\nfi\n',
            "reason": "Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.",
        },
    }


def test_doctor_can_include_shell_bridge_in_summary_output(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(
        agentflow.cli,
        "build_bash_login_shell_bridge_recommendation",
        lambda: ShellBridgeRecommendation(
            target="~/.bash_profile",
            source="~/.profile",
            snippet='if [ -f "$HOME/.profile" ]; then\n  . "$HOME/.profile"\nfi\n',
            reason="Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.",
        ),
    )

    result = runner.invoke(app, ["doctor", "--output", "summary", "--shell-bridge"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.bootstrap`, claude_review (claude) via `target.bootstrap`\n"
        "Shell bridge suggestion for `~/.bash_profile` from `~/.profile`:\n"
        "Reason: Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.\n"
        "if [ -f \"$HOME/.profile\" ]; then\n"
        "  . \"$HOME/.profile\"\n"
        "fi\n"
    )


def test_doctor_shell_bridge_summary_reports_when_no_fix_is_needed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(agentflow.cli, "_pipeline_launch_inspection_nodes", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", lambda: None)

    result = runner.invoke(app, ["doctor", "--output", "summary", "--shell-bridge"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Doctor: ok\n"
        "- kimi_shell_helper: ok - ready\n"
        "Pipeline auto preflight: enabled - path matches the bundled real-agent smoke pipeline.\n"
        "Pipeline auto preflight matches: codex_plan (codex) via `target.bootstrap`, claude_review (claude) via `target.bootstrap`\n"
        "Shell bridge suggestion: not needed\n"
    )


def test_doctor_fails_when_launch_inspection_errors(monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report())
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: _bundled_kimi_smoke_pipeline())
    monkeypatch.setattr(
        agentflow.inspection,
        "build_launch_inspection",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = runner.invoke(app, ["doctor", "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {"name": "kimi_shell_helper", "status": "ok", "detail": "ready"},
            {
                "name": "launch_inspection",
                "status": "failed",
                "detail": (
                    "AgentFlow could not inspect the resolved local launch plan for preflight safety checks: "
                    "RuntimeError: boom."
                ),
            },
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "path matches the bundled real-agent smoke pipeline.",
                "matches": [
                    {"node_id": "codex_plan", "agent": "codex", "trigger": "target.bootstrap"},
                    {"node_id": "claude_review", "agent": "claude", "trigger": "target.bootstrap"},
                ],
                "match_summary": [
                    "codex_plan (codex) via `target.bootstrap`",
                    "claude_review (claude) via `target.bootstrap`",
                ],
            }
        },
    }


def test_doctor_with_pipeline_path_warns_for_custom_home_login_startup(tmp_path, monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-noisy-shell-bridge', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"OPENAI_BASE_URL": ''}, "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "status": "warning",
        "counts": {"ok": 0, "warning": 1, "failed": 0},
        "checks": [
            {
                "name": "bash_login_startup",
                "status": "warning",
                "detail": (
                    f"Node `plan` uses bash login startup from `{custom_home.resolve()}`: "
                    "Bash login startup will not load any user file from `HOME` because "
                    "`~/.bash_profile`, `~/.bash_login`, and `~/.profile` are all missing."
                ),
            }
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": False,
                "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
                "matches": [],
                "match_summary": [],
            }
        },
        "shell_bridge": build_bash_login_shell_bridge_recommendation(home=custom_home).as_dict(),
    }


def test_check_local_warns_for_custom_home_login_startup(tmp_path, monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    captured: dict[str, object] = {}

    class FakeOrchestrator:
        async def submit(self, pipeline: object):
            captured["submitted_pipeline"] = pipeline
            return SimpleNamespace(id="check-local-custom-home")

        async def wait(self, run_id: str, timeout: float | None = None):
            captured["wait_run_id"] = run_id
            return _completed_run(run_id, pipeline_name="doctor-noisy-shell-bridge")

    monkeypatch.setattr(
        agentflow.cli,
        "_build_runtime",
        lambda runs_dir, max_concurrent_runs: (
            SimpleNamespace(run_dir=lambda run_id: Path(runs_dir) / run_id),
            FakeOrchestrator(),
        ),
    )

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-noisy-shell-bridge', "working_dir": '.', "nodes": [{"id": 'plan', "agent": 'codex', "prompt": 'hi', "env": {"OPENAI_BASE_URL": ''}, "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check-local", str(pipeline_path), "--output", "json-summary"])

    assert result.exit_code == 0
    assert json.loads(result.stderr) == {
        "status": "warning",
        "counts": {"ok": 0, "warning": 1, "failed": 0},
        "checks": [
            {
                "name": "bash_login_startup",
                "status": "warning",
                "detail": (
                    f"Node `plan` uses bash login startup from `{custom_home.resolve()}`: "
                    "Bash login startup will not load any user file from `HOME` because "
                    "`~/.bash_profile`, `~/.bash_login`, and `~/.profile` are all missing."
                ),
            }
        ],
        "pipeline": {
            "auto_preflight_scope": "run/smoke",
            "auto_preflight": {
                "enabled": False,
                "reason": "path does not match the bundled smoke pipeline and no local Codex/Claude/Kimi node uses `kimi` bootstrap.",
                "matches": [],
                "match_summary": [],
            }
        },
        "shell_bridge": build_bash_login_shell_bridge_recommendation(home=custom_home).as_dict(),
    }
    assert json.loads(result.stdout) == {
        "id": "check-local-custom-home",
        "status": "completed",
        "pipeline": {"name": "doctor-noisy-shell-bridge"},
        "started_at": "2026-03-08T04:11:03+00:00",
        "finished_at": "2026-03-08T04:11:10+00:00",
        "duration": "7.0s",
        "duration_seconds": 7.0,
        "run_dir": ".agentflow/runs/check-local-custom-home",
        "nodes": [],
    }
    assert captured["wait_run_id"] == "check-local-custom-home"


def test_doctor_with_pipeline_path_auto_includes_shell_bridge_when_auth_depends_on_login_startup(tmp_path, monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-shell-bridge-needed-for-auth', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True, "shell_interactive": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["pipeline"]["auto_preflight"] == {
        "enabled": True,
        "reason": "local Codex/Claude/Kimi nodes depend on shell startup for auth.",
        "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.bash_startup"}],
        "match_summary": ["review (claude) via `target.bash_startup`"],
    }
    assert payload["checks"] == [
        {
            "name": "provider_credentials",
            "status": "failed",
            "detail": (
                "Node `review` (claude) requires `ANTHROPIC_API_KEY` for provider `anthropic`, but it is not set "
                "in the current environment, `node.env`, or `provider.env`."
            ),
        }
    ]
    assert payload["shell_bridge"] == build_bash_login_shell_bridge_recommendation(home=custom_home).as_dict()


def test_run_auto_preflight_stops_when_auth_depends_on_login_startup(tmp_path, monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run should not start")),
    )

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'run-shell-startup-auth-preflight', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True, "shell_interactive": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {
                "name": "provider_credentials",
                "status": "failed",
                "detail": (
                    "Node `review` (claude) requires `ANTHROPIC_API_KEY` for provider `anthropic`, but it is not set "
                    "in the current environment, `node.env`, or `provider.env`."
                ),
            }
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": "local Codex/Claude/Kimi nodes depend on shell startup for auth.",
                "matches": [{"node_id": "review", "agent": "claude", "trigger": "target.bash_startup"}],
                "match_summary": ["review (claude) via `target.bash_startup`"],
            }
        },
        "shell_bridge": build_bash_login_shell_bridge_recommendation(home=custom_home).as_dict(),
    }


def test_run_auto_preflight_fails_closed_when_launch_inspection_errors(tmp_path, monkeypatch):
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentflow.cli,
        "_run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run should not start")),
    )
    monkeypatch.setattr(
        agentflow.inspection,
        "build_launch_inspection",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'run-shell-startup-auth-inspection-error', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True, "shell_interactive": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(pipeline_path), "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {
                "name": "launch_inspection",
                "status": "failed",
                "detail": (
                    "AgentFlow could not inspect the resolved local launch plan for preflight safety checks: "
                    "RuntimeError: boom."
                ),
            }
        ],
        "pipeline": {
            "auto_preflight": {
                "enabled": True,
                "reason": (
                    "AgentFlow could not inspect local launch details while deciding whether shell-startup "
                    "auth preflight is required."
                ),
                "matches": [],
                "match_summary": [],
            }
        },
    }


def test_doctor_with_pipeline_path_uses_pipeline_shell_bridge_for_custom_home(tmp_path, monkeypatch):
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    (host_home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (host_home / ".bashrc").write_text("export PATH=\"$HOME/bin:$PATH\"\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _disable_local_readiness_probes(monkeypatch)

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'doctor-custom-home-shell-bridge', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'claude', "provider": 'anthropic', "prompt": 'hi', "target": {"kind": 'local', "shell": f"env HOME={custom_home} bash", "shell_login": True, "shell_interactive": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", str(pipeline_path), "--output", "json", "--shell-bridge"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["shell_bridge"] == build_bash_login_shell_bridge_recommendation(home=custom_home).as_dict()
    assert payload["checks"] == [
        {
            "name": "provider_credentials",
            "status": "failed",
            "detail": (
                "Node `review` (claude) requires `ANTHROPIC_API_KEY` for provider `anthropic`, but it is not set "
                "in the current environment, `node.env`, or `provider.env`."
            ),
        }
    ]


def test_doctor_command_does_not_import_web_stack(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import builtins
import importlib
import json
import os
from types import SimpleNamespace

from typer.testing import CliRunner
from agentflow.doctor import DoctorCheck, DoctorReport

os.environ.pop("ANTHROPIC_BASE_URL", None)

original_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in {\"agentflow.app\", \"fastapi\"}:
        raise ModuleNotFoundError(name)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
cli_module = importlib.import_module(\"agentflow.cli\")
cli_module.build_local_smoke_doctor_report = lambda: DoctorReport(
    status=\"ok\",
    checks=[DoctorCheck(name=\"kimi_shell_helper\", status=\"ok\", detail=\"ready\")],
)
cli_module._load_pipeline = lambda path: SimpleNamespace(nodes=[])
cli_module._pipeline_launch_inspection_nodes = lambda pipeline: []
result = CliRunner().invoke(cli_module.app, [\"doctor\", \"--output\", \"json\"])
print(json.dumps({\"exit_code\": result.exit_code, \"stdout\": result.stdout}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=repo_root,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "exit_code": 0,
        "stdout": json.dumps({
            "status": "ok",
            "checks": [{"name": "kimi_shell_helper", "status": "ok", "detail": "ready"}],
            "pipeline": {
                "auto_preflight": {
                    "enabled": True,
                    "reason": "path matches the bundled real-agent smoke pipeline.",
                    "matches": [],
                    "match_summary": [],
                }
            },
        }, indent=2) + "\n",
    }


def test_run_command_executes_local_kimi_node_when_pipeline_lives_outside_repo(tmp_path, monkeypatch):
    pipeline_path = tmp_path / "kimi-only.json"
    pipeline_path.write_text(
        json.dumps({"name": 'kimi-only', "working_dir": '.', "nodes": [{"id": 'review', "agent": 'kimi', "prompt": 'Reply with exactly: kimi ok\n', "timeout_seconds": 30, "success_criteria": [{"kind": 'output_contains', "value": 'kimi ok'}]}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTFLOW_KIMI_MOCK_RESPONSE", "kimi ok")
    monkeypatch.setenv("KIMI_API_KEY", "super-secret")
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_checks", lambda pipeline: [])
    monkeypatch.setattr(agentflow.cli, "build_pipeline_local_kimi_readiness_info_checks", lambda pipeline: [])

    result = runner.invoke(app, ["run", str(pipeline_path), "--output", "summary"])

    assert result.exit_code == 0
    assert "Run " in result.stdout
    assert "Pipeline: kimi-only" in result.stdout
    assert "review [kimi]: completed" in result.stdout
    assert "kimi ok" in result.stdout


def test_smoke_stops_when_bundled_preflight_fails(monkeypatch):
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 1
    assert result.stdout == "Doctor: failed\n- kimi_shell_helper: failed - missing\n"


def test_smoke_failed_preflight_includes_shell_bridge_when_available(monkeypatch):
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[
                DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge"),
                DoctorCheck(name="kimi_shell_helper", status="failed", detail="missing"),
            ],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 1
    assert result.stdout == (
        "Doctor: failed\n"
        "- bash_login_startup: warning - missing bridge\n"
        "- kimi_shell_helper: failed - missing\n"
        "Shell bridge suggestion for `~/.bash_profile` from `~/.profile`:\n"
        "Reason: Bash login shells use `~/.bash_profile`, so `~/.profile` never runs.\n"
        "if [ -f \"$HOME/.profile\" ]; then\n"
        "  . \"$HOME/.profile\"\n"
        "fi\n"
    )


def test_smoke_failed_preflight_uses_pipeline_shell_bridge_for_custom_home(tmp_path, monkeypatch):
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    (host_home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    (host_home / ".bashrc").write_text("export PATH=\"$HOME/bin:$PATH\"\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    _reject_bundled_smoke_doctor(monkeypatch)
    _disable_local_readiness_probes(monkeypatch)
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_kimi_bootstrap_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[DoctorCheck(name="kimi_shell_helper", status="failed", detail="broken helper")],
        ),
    )

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps({"name": 'smoke-custom-home-shell-bridge', "working_dir": '.', "nodes": [{"id": 'codex_plan', "agent": 'codex', "prompt": 'hi', "target": {"kind": 'local', "bootstrap": 'kimi', "shell": f"env HOME={custom_home} bash", "shell_login": True, "shell_interactive": True}}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["smoke", str(pipeline_path)])

    assert result.exit_code == 1
    recommendation = build_bash_login_shell_bridge_recommendation(home=custom_home)
    assert recommendation is not None
    assert result.stdout == (
        "Doctor: failed\n"
        "- kimi_shell_helper: failed - broken helper\n"
        f"Pipeline auto preflight: enabled - local Codex/Claude/Kimi nodes use a `kimi` shell bootstrap.\n"
        f"Pipeline auto preflight matches: codex_plan (codex) via `target.bootstrap`\n"
        f"Shell bridge suggestion for `{recommendation.target}` from `{recommendation.source}`:\n"
        f"Reason: {recommendation.reason}\n"
        f"{recommendation.snippet}"
    )


def test_smoke_failed_preflight_honors_json_output(monkeypatch):
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))

    result = runner.invoke(app, ["smoke", "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [{"name": "kimi_shell_helper", "status": "failed", "detail": "missing"}],
    }


def test_smoke_failed_preflight_includes_shell_bridge_json_when_available(monkeypatch):
    monkeypatch.setattr(
        agentflow.cli,
        "build_local_smoke_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[
                DoctorCheck(name="bash_login_startup", status="warning", detail="missing bridge"),
                DoctorCheck(name="kimi_shell_helper", status="failed", detail="missing"),
            ],
        ),
    )
    monkeypatch.setattr(agentflow.cli, "build_bash_login_shell_bridge_recommendation", _shell_bridge_recommendation)
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))

    result = runner.invoke(app, ["smoke", "--output", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "checks": [
            {"name": "bash_login_startup", "status": "warning", "detail": "missing bridge"},
            {"name": "kimi_shell_helper", "status": "failed", "detail": "missing"},
        ],
        "shell_bridge": _shell_bridge_recommendation().as_dict(),
    }


def test_smoke_failed_preflight_honors_json_summary_output(monkeypatch):
    monkeypatch.setattr(agentflow.cli, "build_local_smoke_doctor_report", lambda: _doctor_report(status="failed", detail="missing"))
    monkeypatch.setattr(agentflow.cli, "default_smoke_pipeline_path", lambda: "examples/local-real-agents-kimi-smoke.yaml")
    monkeypatch.setattr(agentflow.cli, "_load_pipeline", lambda path: (_ for _ in ()).throw(AssertionError("pipeline should not load")))

    result = runner.invoke(app, ["smoke", "--output", "json-summary"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "counts": {"ok": 0, "warning": 0, "failed": 1},
        "checks": [{"name": "kimi_shell_helper", "status": "failed", "detail": "missing"}],
    }
