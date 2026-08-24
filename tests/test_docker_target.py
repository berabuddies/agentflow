from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentflow.agents.codex import CodexAdapter
from agentflow.agents.util import SyncAdapter
from agentflow.inspection import build_launch_inspection
from agentflow.loader import load_pipeline_from_text
from agentflow.prepared import ExecutionPaths, PreparedExecution, build_execution_paths
from agentflow.runners.container import ContainerRunner
from agentflow.runners.docker import DockerRunner
from agentflow.runners.registry import RunnerRegistry
from agentflow.specs import (
    ContainerTarget,
    DockerMount,
    DockerNetworkPolicy,
    DockerTarget,
    NodeSpec,
    PipelineSpec,
)


def _node(target: dict[str, object]) -> NodeSpec:
    return NodeSpec.model_validate(
        {
            "id": "docker-node",
            "agent": "codex",
            "prompt": "run in Docker",
            "target": target,
        }
    )


def _paths(tmp_path: Path) -> ExecutionPaths:
    return ExecutionPaths(
        host_workdir=tmp_path,
        host_runtime_dir=tmp_path / ".runtime",
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=tmp_path / "agentflow-app",
    )


def _option_values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


@pytest.mark.parametrize(
    ("shorthand", "expected_mode", "expected_name", "expected_network"),
    [
        ("none", "none", None, "none"),
        ("bridge", "bridge", None, "bridge"),
        ("host", "host", None, "host"),
        ("agentflow-egress", "custom", "agentflow-egress", "agentflow-egress"),
    ],
)
def test_docker_network_policy_accepts_string_shorthand(
    shorthand: str,
    expected_mode: str,
    expected_name: str | None,
    expected_network: str,
):
    policy = DockerNetworkPolicy.model_validate(shorthand)

    assert policy.mode == expected_mode
    assert policy.name == expected_name
    assert policy.docker_network == expected_network


def test_docker_target_schema_normalizes_compatibility_shorthands():
    node = _node(
        {
            "kind": "docker",
            "image": "agentflow-agents:test",
            "app_mount": "/agentflow-app",
            "network": "restricted-egress",
            "mount_docker_socket": True,
            "mounts": [
                {"source": " ./fixtures ", "target": " /fixtures ", "read_only": True},
            ],
        }
    )

    assert isinstance(node.target, DockerTarget)
    assert node.target.image == "agentflow-agents:test"
    assert node.target.network_policy == DockerNetworkPolicy(
        mode="custom",
        name="restricted-egress",
    )
    assert node.target.mount_docker_daemon is True
    assert node.target.mounts == [
        DockerMount(source="./fixtures", target="/fixtures", read_only=True),
    ]


def test_docker_target_secure_defaults_do_not_expose_host_app_or_credentials():
    target = DockerTarget()

    assert target.image == "agentflow-agents:latest"
    assert target.app_mount is None
    assert target.user == "host"
    assert target.inherit_credentials is False
    assert target.privileged is False
    assert target.mount_docker_daemon is False
    assert target.dind is False
    assert target.network_policy.mode == "bridge"


@pytest.mark.parametrize(
    ("target_patch", "expected_message"),
    [
        ({"dind": True}, r"dind.*requires.*privileged"),
        (
            {"dind": True, "privileged": True, "mount_docker_daemon": True},
            r"dind.*mount_docker_daemon.*mutually exclusive",
        ),
        (
            {"dind": True, "privileged": True, "extra_args": ["--read-only"]},
            r"dind.*cannot be combined.*read-only",
        ),
        (
            {"docker_daemon_socket": "/run/user/1000/docker.sock"},
            r"docker_daemon_socket.*requires.*mount_docker_daemon",
        ),
        (
            {"network_policy": {"mode": "custom"}},
            r"network_policy\.name.*required.*custom",
        ),
        (
            {"network_policy": {"mode": "bridge", "aliases": ["agent"]}},
            r"network_policy\.aliases.*requires.*custom",
        ),
        (
            {"mounts": [{"source": ".", "target": "relative"}]},
            r"mounts.*target.*absolute container path",
        ),
        (
            {
                "mounts": [
                    {"source": "first", "target": "/data"},
                    {"source": "second", "target": "/data"},
                ]
            },
            r"duplicate Docker mount targets.*data",
        ),
        (
            {
                "mounts": [
                    {"source": "first", "target": "/data"},
                    {"source": "second", "target": "/data/nested"},
                ]
            },
            r"mounts.*must not overlap.*data/nested",
        ),
        (
            {"mounts": [{"source": ".", "target": "/workspace"}]},
            r"cannot overlap AgentFlow-managed mount targets.*workspace",
        ),
        (
            {"mounts": [{"source": ".", "target": "/workspace/cache"}]},
            r"cannot overlap AgentFlow-managed mount targets.*workspace/cache.*workspace",
        ),
        (
            {"workdir_mount": "/", "runtime_mount": "/agentflow-runtime"},
            r"managed Docker mount targets must not overlap.*agentflow-runtime",
        ),
        ({"workdir_mount": "//workspace"}, r"workdir_mount.*single leading slash"),
        ({"image": "--privileged"}, r"image.*not a command-line option"),
        ({"extra_args": ["alpine:3.20"]}, r"extra_args.*positional values"),
        ({"extra_args": ["--pid=host"]}, r"extra_args.*unsupported.*pid"),
        (
            {"extra_args": ["--network=host"]},
            r"extra_args.*cannot set.*network",
        ),
    ],
)
def test_docker_target_rejects_unsafe_or_conflicting_configuration(
    target_patch: dict[str, object],
    expected_message: str,
):
    with pytest.raises(ValidationError, match=expected_message):
        _node({"kind": "docker", **target_patch})


def test_docker_runner_plan_contains_structured_isolation_controls(tmp_path: Path):
    socket_path = tmp_path / "run" / "docker.sock"
    auth_path = tmp_path / "home" / ".codex" / "auth.json"
    node = _node(
        {
            "kind": "docker",
            "image": "agentflow-agents:test",
            "app_mount": "/agentflow-app",
            "mounts": [
                {
                    "source": str(tmp_path / "fixtures"),
                    "target": "/fixtures",
                    "read_only": True,
                },
                {"source": str(tmp_path / "cache"), "target": "/cache"},
            ],
            "network_policy": {
                "mode": "custom",
                "name": "restricted-egress",
                "aliases": ["agent"],
                "dns": ["1.1.1.1"],
                "add_hosts": {"internal.test": "192.0.2.10"},
            },
            "privileged": True,
            "mount_docker_daemon": True,
            "inherit_credentials": True,
            "docker_daemon_socket": str(socket_path),
            "extra_args": ["--cpus", "2"],
            "entrypoint": "/bin/sh",
        }
    )
    prepared = PreparedExecution(
        command=["codex", "exec", "inspect the repo"],
        env={"OPENAI_API_KEY": "secret", "PYTHONPATH": "/extra"},
        cwd="/workspace/task",
        trace_kind="codex",
        runtime_files={"codex_home/config.toml": "model = 'gpt-5'\n"},
        runtime_symlinks={"codex_home/auth.json": str(auth_path)},
        stdin="prompt on stdin\n",
    )

    plan = DockerRunner().plan_execution(node, prepared, _paths(tmp_path))

    assert plan.kind == "docker"
    assert plan.command is not None
    command = plan.command
    assert command[:2] == ["docker", "run"]
    assert "--rm" not in command
    assert "--interactive" in command
    assert "--privileged" in command
    assert _option_values(command, "--network") == ["restricted-egress"]
    assert _option_values(command, "--network-alias") == ["agent"]
    assert _option_values(command, "--dns") == ["1.1.1.1"]
    assert _option_values(command, "--add-host") == ["internal.test=192.0.2.10"]
    assert _option_values(command, "--workdir") == ["/workspace/task"]
    assert _option_values(command, "--entrypoint") == ["/bin/sh"]
    assert "--cpus" in command
    assert command[command.index("--cpus") + 1] == "2"

    mount_specs = set(_option_values(command, "--mount"))
    assert f"type=bind,src={tmp_path},dst=/workspace" in mount_specs
    assert (
        f"type=bind,src={tmp_path / '.runtime'},dst=/agentflow-runtime" in mount_specs
    )
    assert (
        f"type=bind,src={tmp_path / 'agentflow-app'},dst=/agentflow-app,readonly"
        in mount_specs
    )
    assert (
        f"type=bind,src={tmp_path / 'fixtures'},dst=/fixtures,readonly" in mount_specs
    )
    assert f"type=bind,src={tmp_path / 'cache'},dst=/cache" in mount_specs
    assert (
        f"type=bind,src={auth_path},dst=/agentflow-runtime/codex_home/auth.json,readonly"
        in mount_specs
    )
    assert f"type=bind,src={socket_path},dst=/var/run/docker.sock" in mount_specs

    assert _option_values(command, "--env") == []
    assert _option_values(command, "--env-file") == [
        str(tmp_path / ".runtime" / ".agentflow" / "docker-env.list")
    ]
    assert "secret" not in command
    assert plan.env == {}
    image_index = command.index("agentflow-agents:test")
    assert command[image_index + 1 :] == ["codex", "exec", "inspect the repo"]

    assert plan.cwd == str(tmp_path)
    assert plan.stdin == "prompt on stdin\n"
    assert plan.runtime_files == [
        ".agentflow/docker-env.list",
        "codex_home/auth.json",
        "codex_home/config.toml",
    ]
    assert plan.payload is not None
    assert plan.payload["privileged"] is True
    assert plan.payload["mount_docker_daemon"] is True
    assert plan.payload["inherit_credentials"] is True
    assert plan.payload["dind"] is False
    assert set(plan.payload["env_keys"]) >= {
        "AGENTFLOW_DOCKER_CONTAINER_NAME",
        "AGENTFLOW_HOST_RUNTIME_DIR",
        "AGENTFLOW_HOST_WORKDIR",
        "DOCKER_HOST",
        "HOME",
        "OPENAI_API_KEY",
        "PYTHONPATH",
    }


def test_docker_runner_plan_enables_dind_without_host_socket_mount(tmp_path: Path):
    node = _node(
        {
            "kind": "docker",
            "image": "agentflow-agents:dind",
            "privileged": True,
            "dind": True,
            "network_policy": "none",
        }
    )
    prepared = PreparedExecution(
        command=["docker", "info"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
        stdin=None,
    )

    plan = DockerRunner().plan_execution(node, prepared, _paths(tmp_path))

    assert plan.command is not None
    command = plan.command
    assert "--privileged" in command
    assert "--interactive" not in command
    assert _option_values(command, "--network") == ["none"]
    assert all(
        "dst=/var/run/docker.sock" not in mount
        for mount in _option_values(command, "--mount")
    )
    assert _option_values(command, "--env") == []
    assert _option_values(command, "--env-file") == [
        str(tmp_path / ".runtime" / ".agentflow" / "docker-env.list")
    ]
    assert "--user" not in command
    assert plan.stdin is None
    assert plan.payload is not None
    assert plan.payload["dind"] is True
    assert plan.payload["mount_docker_daemon"] is False
    assert plan.payload["effective_user"] == f"{os.getuid()}:{os.getgid()}"
    assert plan.payload["docker_launch_user"] == "image-default"
    assert set(plan.payload["env_keys"]) >= {
        "AGENTFLOW_DIND",
        "AGENTFLOW_RUN_GID",
        "AGENTFLOW_RUN_UID",
        "DOCKER_HOST",
        "DOCKER_TLS_CERTDIR",
        "HOME",
    }


def test_docker_runner_env_file_keeps_every_value_out_of_argv(tmp_path: Path):
    node = _node({"kind": "docker"})
    prepared = PreparedExecution(
        command=["bash", "-c", "true"],
        env={
            "DATABASE_URL": "postgres://user:secret@db/example",
            "SENTRY_DSN": "https://secret@sentry.invalid/1",
        },
        cwd="/workspace",
        trace_kind="shell",
    )
    runner = DockerRunner()

    docker_prepared, _, container_env = runner._docker_prepared(
        node, prepared, _paths(tmp_path)
    )

    command_text = "\x00".join(docker_prepared.command)
    assert "postgres://" not in command_text
    assert "secret@sentry" not in command_text
    assert docker_prepared.env == {}
    assert docker_prepared.runtime_files[".agentflow/docker-env.list"] == (
        "DATABASE_URL=postgres://user:secret@db/example\n"
        "SENTRY_DSN=https://secret@sentry.invalid/1\n"
        "HOME=/agentflow-runtime/home\n"
    )
    assert set(container_env) == {"DATABASE_URL", "SENTRY_DSN", "HOME"}


def test_docker_runner_rejects_multiline_environment_values(tmp_path: Path):
    node = _node({"kind": "docker"})
    prepared = PreparedExecution(
        command=["bash", "-c", "true"],
        env={"PRIVATE_KEY": "first line\nsecond line"},
        cwd="/workspace",
        trace_kind="shell",
    )

    with pytest.raises(ValueError, match=r"PRIVATE_KEY.*single-line"):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


@pytest.mark.parametrize("invalid_name", ["#TOKEN", " FOO", "FOO BAR", "9TOKEN"])
def test_docker_runner_rejects_nonportable_environment_names(
    tmp_path: Path, invalid_name: str
):
    node = _node({"kind": "docker"})
    prepared = PreparedExecution(
        command=["bash", "-c", "true"],
        env={invalid_name: "value"},
        cwd="/workspace",
        trace_kind="shell",
    )

    with pytest.raises(ValueError, match=r"environment variable name must match"):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_docker_runner_resolves_named_unix_docker_context(tmp_path: Path, monkeypatch):
    docker_config = tmp_path / "docker-config"
    metadata = docker_config / "contexts" / "meta" / "context-id" / "meta.json"
    metadata.parent.mkdir(parents=True)
    socket_path = tmp_path / "colima.sock"
    (docker_config / "config.json").write_text(
        json.dumps({"currentContext": "colima"}),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "Name": "colima",
                "Endpoints": {"docker": {"Host": f"unix://{socket_path}"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    node = _node({"kind": "docker", "mount_docker_daemon": True})
    prepared = PreparedExecution(
        command=["docker", "info"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    plan = DockerRunner().plan_execution(node, prepared, _paths(tmp_path))

    assert any(
        mount == f"type=bind,src={socket_path},dst=/var/run/docker.sock"
        for mount in _option_values(plan.command or [], "--mount")
    )


def test_docker_runner_rejects_non_unix_active_daemon_for_socket_mount(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker.example:2376")
    node = _node({"kind": "docker", "mount_docker_daemon": True})
    prepared = PreparedExecution(
        command=["docker", "info"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    with pytest.raises(ValueError, match=r"require a local Unix Docker endpoint"):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_docker_runner_validates_daemon_mount_is_a_unix_socket(
    tmp_path: Path, monkeypatch
):
    socket_path = tmp_path / "docker.sock"
    node = _node(
        {
            "kind": "docker",
            "mount_docker_daemon": True,
            "docker_daemon_socket": str(socket_path),
        }
    )
    paths = _paths(tmp_path)
    runner = DockerRunner()

    socket_path.write_text("not a socket", encoding="utf-8")
    with pytest.raises(ValueError, match=r"not a Unix socket"):
        runner._validate_docker_daemon_socket(node.target, paths)

    socket_path.unlink()
    monkeypatch.chdir(tmp_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as daemon_socket:
        daemon_socket.bind(socket_path.name)
        runner._validate_docker_daemon_socket(node.target, paths)


def test_docker_runner_rejects_adapter_host_files_without_explicit_opt_in(
    tmp_path: Path,
):
    node = _node({"kind": "docker"})
    prepared = PreparedExecution(
        command=["codex", "exec", "hi"],
        env={},
        cwd="/workspace",
        trace_kind="codex",
        runtime_symlinks={"codex_home/auth.json": str(tmp_path / "auth.json")},
    )

    with pytest.raises(ValueError, match=r"inherit_credentials: true"):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_docker_runner_rejects_writable_alias_of_read_only_workspace(tmp_path: Path):
    node = _node(
        {
            "kind": "docker",
            "workdir_read_only": True,
            "mounts": [{"source": str(tmp_path), "target": "/workspace-alias"}],
        }
    )
    prepared = PreparedExecution(
        command=["bash", "-c", "true"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    with pytest.raises(
        ValueError, match=r"read-write.*overlaps.*read-only managed workspace"
    ):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_docker_runner_requires_daemon_opt_in_for_explicit_socket_parent_mount(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'docker.sock'}")
    node = _node(
        {
            "kind": "docker",
            "mounts": [{"source": str(tmp_path), "target": "/host-directory"}],
        }
    )
    prepared = PreparedExecution(
        command=["bash", "-c", "true"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    with pytest.raises(
        ValueError, match=r"exposes the active daemon socket.*mount_docker_daemon"
    ):
        DockerRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_sync_adapter_rejects_docker_target_before_building_invalid_ssh_command(
    tmp_path: Path,
):
    node = NodeSpec.model_validate(
        {
            "id": "sync",
            "agent": "sync",
            "prompt": "full",
            "target": {"kind": "docker"},
        }
    )

    with pytest.raises(ValueError, match=r"sync nodes require.*ssh.*ec2.*ecs"):
        SyncAdapter().prepare(node, "full", _paths(tmp_path))


def test_codex_docker_credentials_are_opt_in(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text("model = 'gpt-5'\n", encoding="utf-8")
    (codex_home / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    paths = _paths(tmp_path)

    isolated = CodexAdapter().prepare(_node({"kind": "docker"}), "hi", paths)
    inherited = CodexAdapter().prepare(
        _node({"kind": "docker", "inherit_credentials": True}),
        "hi",
        paths,
    )

    assert isolated.runtime_symlinks == {}
    assert inherited.runtime_symlinks == {
        "codex_home/config.toml": str(codex_home / "config.toml"),
        "codex_home/auth.json": str(codex_home / "auth.json"),
    }
    assert inherited.env["CODEX_HOME"] == "/agentflow-runtime/codex_home"


def test_runtime_materialization_uses_private_permissions(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runner = DockerRunner()

    runner.materialize_runtime_files(
        runtime_dir,
        {"codex_home/config.toml": "api_key = 'secret'\n"},
    )

    assert runtime_dir.stat().st_mode & 0o777 == 0o700
    assert (runtime_dir / "codex_home").stat().st_mode & 0o777 == 0o700
    assert (runtime_dir / "codex_home" / "config.toml").stat().st_mode & 0o777 == 0o600


def test_build_execution_paths_uses_container_paths_for_docker_target(tmp_path: Path):
    pipeline_workdir = tmp_path / "repo"
    target = DockerTarget(
        workdir_mount="/repo",
        runtime_mount="/run/agentflow",
    )

    paths = build_execution_paths(
        base_dir=tmp_path / "runs",
        pipeline_workdir=pipeline_workdir,
        run_id="run-1",
        node_id="plan",
        node_target=target,
        create_runtime_dir=False,
    )

    assert paths.host_workdir == pipeline_workdir
    assert paths.host_runtime_dir == tmp_path / "runs" / "run-1" / "runtime" / "plan"
    assert paths.target_workdir == "/repo"
    assert paths.target_runtime_dir == "/run/agentflow"


def test_loader_resolves_docker_mount_sources_relative_to_pipeline_workdir(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    absolute_source = tmp_path / "absolute-source"
    monkeypatch.setenv("HOME", str(home))

    pipeline = load_pipeline_from_text(
        json.dumps(
            {
                "name": "docker-mounts",
                "working_dir": "repo",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "codex",
                        "prompt": "hi",
                        "target": {
                            "kind": "docker",
                            "mounts": [
                                {"source": "fixtures", "target": "/fixtures"},
                                {"source": "~/cache", "target": "/cache"},
                                {"source": str(absolute_source), "target": "/absolute"},
                            ],
                        },
                    }
                ],
            }
        ),
        base_dir=tmp_path,
    )

    assert isinstance(pipeline.nodes[0].target, DockerTarget)
    assert [mount.source for mount in pipeline.nodes[0].target.mounts] == [
        str((tmp_path / "repo" / "fixtures").resolve()),
        str((home / "cache").resolve()),
        str(absolute_source.resolve()),
    ]


def test_launch_inspection_summarizes_docker_and_warns_about_host_isolation_bypasses(
    tmp_path: Path,
):
    pipeline = PipelineSpec.model_validate(
        {
            "name": "inspect-risky-docker",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "inspect",
                    "agent": "python",
                    "prompt": "print('ok')",
                    "target": {
                        "kind": "docker",
                        "image": "agentflow-agents:test",
                        "privileged": True,
                        "mount_docker_daemon": True,
                        "network_policy": "host",
                    },
                }
            ],
        }
    )

    report = build_launch_inspection(pipeline, runs_dir=str(tmp_path / ".agentflow"))
    node_plan = report["nodes"][0]

    assert node_plan["launch"]["kind"] == "docker"
    assert (
        node_plan["launch"]["payload_summary"] == "docker image=agentflow-agents:test"
    )
    assert len(node_plan["warnings"]) == 3
    assert any(
        "daemon socket" in warning and "root-level control" in warning
        for warning in node_plan["warnings"]
    )
    assert any(
        "Privileged Docker execution" in warning for warning in node_plan["warnings"]
    )
    assert any(
        "host networking" in warning and "not network-isolated" in warning
        for warning in node_plan["warnings"]
    )


def test_launch_inspection_does_not_claim_docker_inherits_ambient_auth(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    pipeline = PipelineSpec.model_validate(
        {
            "name": "inspect-docker-auth",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "inspect",
                    "agent": "codex",
                    "prompt": "hi",
                    "target": {"kind": "docker"},
                }
            ],
        }
    )

    report = build_launch_inspection(pipeline, runs_dir=str(tmp_path / ".agentflow"))

    assert (
        "current environment and host CLI homes are not inherited"
        in report["nodes"][0]["auth"]
    )


def test_legacy_container_target_still_uses_container_runner(tmp_path: Path):
    node = _node(
        {
            "kind": "container",
            "image": "python:3.12",
            "extra_args": ["--network", "host"],
        }
    )
    prepared = PreparedExecution(
        command=["python", "-V"],
        env={},
        cwd="/workspace",
        trace_kind="python",
    )
    registry = RunnerRegistry()

    assert isinstance(node.target, ContainerTarget)
    assert not isinstance(node.target, DockerTarget)
    assert isinstance(registry.get("container"), ContainerRunner)
    assert isinstance(registry.get("docker"), DockerRunner)

    plan = registry.get("container").plan_execution(node, prepared, _paths(tmp_path))
    assert plan.kind == "container"
    assert plan.command is not None
    assert "--network" in plan.command
    assert plan.command[plan.command.index("--network") + 1] == "host"


@pytest.mark.asyncio
async def test_docker_runner_uses_container_exit_when_client_stalls(
    tmp_path: Path, monkeypatch
):
    node = _node({"kind": "docker"})
    node.timeout_seconds = 5
    paths = _paths(tmp_path)
    docker_prepared = PreparedExecution(
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        env={},
        cwd=str(tmp_path),
        trace_kind="python",
    )
    runner = DockerRunner()
    removals: list[tuple[str, str]] = []
    states: list[dict[str, object] | None] = [
        None,
        {"Status": "running", "ExitCode": 0},
        {"Status": "exited", "ExitCode": 23},
    ]

    async def fake_remove(engine: str, container_name: str) -> None:
        removals.append((engine, container_name))

    async def fake_inspect(
        _engine: str, _container_name: str
    ) -> dict[str, object] | None:
        return states.pop(0) if states else {"Status": "exited", "ExitCode": 23}

    monkeypatch.setattr(runner, "_force_remove_container", fake_remove)
    monkeypatch.setattr(runner, "_inspect_container_state", fake_inspect)
    monkeypatch.setattr(
        runner,
        "_docker_prepared",
        lambda _node, _prepared, _paths: (docker_prepared, [], {}),
    )

    output: list[tuple[str, str]] = []

    async def on_output(stream: str, text: str) -> None:
        output.append((stream, text))

    started = time.monotonic()
    result = await runner.execute(node, docker_prepared, paths, on_output, lambda: False)

    assert time.monotonic() - started < 4
    assert result.exit_code == 23
    assert result.timed_out is False
    assert result.cancelled is False
    assert ("stderr", "Cancelled by user") not in output
    assert len(removals) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [None, False, "0", -1, 256])
async def test_docker_runner_fails_closed_for_invalid_container_exit_code(
    monkeypatch, exit_code: object
):
    runner = DockerRunner()

    async def fake_inspect(
        _engine: str, _container_name: str
    ) -> dict[str, object]:
        return {"Status": "exited", "ExitCode": exit_code}

    monkeypatch.setattr(runner, "_inspect_container_state", fake_inspect)

    assert await runner._wait_for_container_exit("docker", "agentflow-test") == 125
