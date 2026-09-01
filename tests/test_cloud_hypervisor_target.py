from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentflow.agents.codex import CodexAdapter
from agentflow.cloud_hypervisor_guest import (
    GuestRequestError,
    _command_for_identity,
    _configure_dhcp,
    _configure_network,
    _execute_request,
    _validate_mounts,
    _validate_request,
)
from agentflow.inspection import build_launch_inspection
from agentflow.loader import load_pipeline_from_text
from agentflow.prepared import ExecutionPaths, PreparedExecution, build_execution_paths
from agentflow.runners.cloud_hypervisor import CloudHypervisorRunner
from agentflow.runners.registry import RunnerRegistry
from agentflow.specs import (
    CloudHypervisorMount,
    CloudHypervisorNetworkPolicy,
    CloudHypervisorTarget,
    NodeSpec,
    PipelineSpec,
)


def _node(
    target: dict[str, object],
    *,
    agent: str = "shell",
    env: dict[str, str] | None = None,
) -> NodeSpec:
    return NodeSpec.model_validate(
        {
            "id": "vm-node",
            "agent": agent,
            "prompt": "run in a VM",
            "env": env or {},
            "target": target,
        }
    )


def _paths(tmp_path: Path) -> ExecutionPaths:
    return ExecutionPaths(
        host_workdir=tmp_path / "workspace",
        host_runtime_dir=tmp_path / "runtime",
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=tmp_path / "app",
    )


def _option_values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def test_cloud_hypervisor_target_secure_defaults():
    target = CloudHypervisorTarget(kernel="./vmlinux", rootfs="./rootfs")

    assert target.cpus == 2
    assert target.memory_mib == 4096
    assert target.workdir_mount == "/workspace"
    assert target.runtime_mount == "/agentflow-runtime"
    assert target.app_mount is None
    assert target.user == "host"
    assert target.inherit_credentials is False
    assert target.network_policy.mode == "none"
    assert target.network_policy.num_queues == 2
    assert target.seccomp == "true"


@pytest.mark.parametrize(
    ("shorthand", "expected"),
    [
        ("none", CloudHypervisorNetworkPolicy(mode="none")),
        (
            "tap",
            CloudHypervisorNetworkPolicy(
                mode="tap",
                host_ip="192.168.249.1",
                host_mask="255.255.255.0",
                guest_address="192.168.249.2/24",
                gateway="192.168.249.1",
            ),
        ),
        (
            "agenttap0",
            CloudHypervisorNetworkPolicy(mode="tap", tap="agenttap0", dhcp=True),
        ),
    ],
)
def test_cloud_hypervisor_network_policy_shorthand(
    shorthand: str,
    expected: CloudHypervisorNetworkPolicy,
):
    assert CloudHypervisorNetworkPolicy.model_validate(shorthand) == expected


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"workdir_mount": "relative"}, r"workdir_mount.*absolute guest path"),
        ({"runtime_mount": "//runtime"}, r"runtime_mount.*single leading slash"),
        ({"runtime_mount": "/runtime:bad"}, r"runtime_mount.*colons"),
        (
            {"workdir_mount": "/", "runtime_mount": "/agentflow-runtime"},
            r"reserved system path|must not overlap",
        ),
        ({"workdir_mount": "/proc/work"}, r"reserved system path.*proc"),
        (
            {"mounts": [{"source": ".", "target": "/dev/data"}]},
            r"reserved system path.*dev",
        ),
        (
            {
                "mounts": [
                    {"source": "one", "target": "/inputs"},
                    {"source": "two", "target": "/inputs/nested"},
                ]
            },
            r"mount targets must not overlap.*inputs/nested",
        ),
        (
            {"mounts": [{"source": ".", "target": "/workspace/cache"}]},
            r"mount targets must not overlap.*workspace",
        ),
        ({"user": "daemon"}, r"user.*host.*numeric"),
        ({"user": "4294967295:1"}, r"user.*fit Linux 32-bit"),
        ({"init_path": "sbin/init"}, r"init_path.*absolute guest path"),
        (
            {"nss_wrapper_path": "libnss_wrapper.so"},
            r"nss_wrapper_path.*absolute guest path",
        ),
        ({"kernel_args": ["root=/dev/vda1"]}, r"kernel_args.*cannot override.*root"),
        ({"kernel_args": ["rw"]}, r"kernel_args.*cannot override.*root"),
        (
            {"kernel_args": ["quiet init=/bin/sh"]},
            r"kernel_args.*without whitespace",
        ),
        (
            {"network_policy": {"mode": "none", "tap": "tap0"}},
            r"network settings require.*tap",
        ),
        (
            {
                "network_policy": {
                    "mode": "tap",
                    "dhcp": True,
                    "guest_address": "192.0.2.2/24",
                }
            },
            r"dhcp.*guest_address.*mutually exclusive",
        ),
        (
            {"network_policy": {"mode": "tap", "gateway": "192.0.2.1"}},
            r"gateway.*requires.*guest_address",
        ),
        (
            {"network_policy": {"mode": "tap", "host_ip": "192.0.2.1"}},
            r"host_ip.*host_mask.*together",
        ),
        (
            {
                "network_policy": {
                    "mode": "tap",
                    "host_ip": "2001:db8::1",
                    "host_mask": "255.255.255.0",
                }
            },
            r"host_ip.*IPv4",
        ),
        (
            {"network_policy": {"mode": "tap", "num_queues": 3}},
            r"num_queues.*even",
        ),
    ],
)
def test_cloud_hypervisor_target_rejects_unsafe_or_conflicting_configuration(
    patch: dict[str, object],
    message: str,
):
    with pytest.raises(ValidationError, match=message):
        _node(
            {
                "kind": "cloud_hypervisor",
                "kernel": "vmlinux",
                "rootfs": "rootfs",
                **patch,
            }
        )


def test_cloud_hypervisor_target_schema_parses_structured_mounts_and_network():
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": " vmlinux ",
            "rootfs": " rootfs ",
            "mounts": [
                {"source": " ./fixtures ", "target": " /inputs ", "read_only": False}
            ],
            "network_policy": {
                "mode": "tap",
                "tap": "agenttap0",
                "mac": "02:00:00:00:00:01",
                "host_ip": "192.0.2.1",
                "host_mask": "255.255.255.0",
                "guest_address": "192.0.2.2/24",
                "gateway": "192.0.2.1",
                "dns": ["1.1.1.1"],
            },
        }
    )

    assert isinstance(node.target, CloudHypervisorTarget)
    assert node.target.kernel == "vmlinux"
    assert node.target.rootfs == "rootfs"
    assert node.target.mounts == [
        CloudHypervisorMount(source="./fixtures", target="/inputs", read_only=False)
    ]
    assert node.target.network_policy.tap == "agenttap0"
    assert node.target.network_policy.guest_address == "192.0.2.2/24"


def test_cloud_hypervisor_runner_plan_contains_vmm_vsock_and_virtiofs_configuration(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    for directory in (
        paths.host_workdir,
        paths.host_runtime_dir,
        paths.app_root,
        tmp_path / "rootfs",
        tmp_path / "inputs",
    ):
        directory.mkdir(parents=True)
    (tmp_path / "vmlinux").write_bytes(b"kernel")
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
            "cpus": 4,
            "memory_mib": 2048,
            "app_mount": "/agentflow-app",
            "mounts": [
                {
                    "source": str(tmp_path / "inputs"),
                    "target": "/inputs",
                    "read_only": True,
                }
            ],
            "network_policy": {
                "mode": "tap",
                "tap": "agenttap0",
                "host_ip": "192.0.2.1",
                "host_mask": "255.255.255.0",
                "guest_address": "192.0.2.2/24",
                "gateway": "192.0.2.1",
                "dns": ["1.1.1.1"],
                "num_queues": 2,
            },
        },
        env={"DATABASE_URL": "postgres://user:secret@example.invalid/db"},
    )
    prepared = PreparedExecution(
        command=["bash", "-c", "echo ready"],
        env=dict(node.env),
        cwd="/workspace",
        trace_kind="shell",
        stdin="input\n",
    )

    plan = CloudHypervisorRunner().plan_execution(node, prepared, paths)

    assert plan.kind == "cloud_hypervisor"
    assert plan.command is not None
    assert plan.command[0] == "cloud-hypervisor"
    assert _option_values(plan.command, "--kernel") == [str(tmp_path / "vmlinux")]
    assert _option_values(plan.command, "--cpus") == ["boot=4"]
    assert _option_values(plan.command, "--memory") == ["size=2048M,shared=on"]
    assert _option_values(plan.command, "--console") == ["off"]
    assert "root=/dev/root" in _option_values(plan.command, "--cmdline")[0]
    assert (
        "init=/usr/local/bin/agentflow-cloud-hypervisor-init"
        in _option_values(plan.command, "--cmdline")[0]
    )
    assert len(_option_values(plan.command, "--fs")) == 5
    assert _option_values(plan.command, "--net") == [
        "tap=agenttap0,ip=192.0.2.1,mask=255.255.255.0,num_queues=2"
    ]
    assert "postgres://" not in "\x00".join(plan.command)
    assert plan.env == {}
    assert plan.stdin is None
    assert plan.payload is not None
    assert plan.payload["env_keys"] == [
        "DATABASE_URL",
        "HOME",
        "LD_PRELOAD",
        "LOGNAME",
        "NSS_WRAPPER_GROUP",
        "NSS_WRAPPER_PASSWD",
        "PYTHONPATH",
        "USER",
    ]
    assert "secret" not in json.dumps(plan.payload)
    assert plan.payload["guest_user"] == f"{os.getuid()}:{os.getgid()}"
    assert plan.runtime_files == [".agentflow-nss/group", ".agentflow-nss/passwd"]

    virtiofsd_commands = plan.payload["virtiofsd_commands"]
    assert isinstance(virtiofsd_commands, list)
    assert len(virtiofsd_commands) == 5
    assert "--readonly" in virtiofsd_commands[0]
    assert "--readonly" not in virtiofsd_commands[1]
    assert "--readonly" not in virtiofsd_commands[2]
    assert "--readonly" in virtiofsd_commands[3]
    assert "--readonly" in virtiofsd_commands[4]
    assert any(
        argument.startswith("--translate-uid=map:")
        for argument in virtiofsd_commands[0]
    )


def test_cloud_hypervisor_runner_rejects_writable_alias_of_read_only_workspace(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    paths.host_workdir.mkdir(parents=True)
    paths.host_runtime_dir.mkdir()
    (tmp_path / "rootfs").mkdir()
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
            "workdir_read_only": True,
            "mounts": [
                {
                    "source": str(paths.host_workdir / "."),
                    "target": "/alias",
                    "read_only": False,
                }
            ],
        }
    )
    prepared = PreparedExecution(
        command=["true"], env={}, cwd="/workspace", trace_kind="shell"
    )

    with pytest.raises(ValueError, match=r"read-write.*overlaps.*read-only.*workspace"):
        CloudHypervisorRunner().plan_execution(node, prepared, paths)


def test_cloud_hypervisor_runner_rejects_read_only_alias_of_writable_workspace(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    paths.host_workdir.mkdir(parents=True)
    paths.host_runtime_dir.mkdir()
    (tmp_path / "rootfs").mkdir()
    fixtures = paths.host_workdir / "fixtures"
    fixtures.mkdir()
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
            "mounts": [
                {
                    "source": str(fixtures),
                    "target": "/inputs",
                    "read_only": True,
                }
            ],
        }
    )
    prepared = PreparedExecution(
        command=["true"], env={}, cwd="/workspace", trace_kind="shell"
    )

    with pytest.raises(ValueError, match=r"read-only.*overlaps.*writable.*workspace"):
        CloudHypervisorRunner().plan_execution(node, prepared, paths)


def test_cloud_hypervisor_runner_rejects_mixed_access_aliases_between_mounts(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    paths.host_workdir.mkdir(parents=True)
    paths.host_runtime_dir.mkdir()
    (tmp_path / "rootfs").mkdir()
    shared = tmp_path / "shared"
    (shared / "nested").mkdir(parents=True)
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
            "mounts": [
                {"source": str(shared), "target": "/inputs", "read_only": True},
                {
                    "source": str(shared / "nested"),
                    "target": "/outputs",
                    "read_only": False,
                },
            ],
        }
    )
    prepared = PreparedExecution(
        command=["true"], env={}, cwd="/workspace", trace_kind="shell"
    )

    with pytest.raises(ValueError, match=r"read-only and read-write.*must not overlap"):
        CloudHypervisorRunner().plan_execution(node, prepared, paths)


def test_cloud_hypervisor_runner_protects_rootfs_from_writable_workspace_alias(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    rootfs = paths.host_workdir / ".agentflow" / "cloud-hypervisor" / "rootfs"
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(rootfs),
        }
    )
    prepared = PreparedExecution(
        command=["true"], env={}, cwd="/workspace", trace_kind="shell"
    )

    with pytest.raises(ValueError, match=r"writable managed workspace.*rootfs"):
        CloudHypervisorRunner().plan_execution(node, prepared, paths)

    read_only_node = node.model_copy(
        update={"target": node.target.model_copy(update={"workdir_read_only": True})}
    )
    plan = CloudHypervisorRunner().plan_execution(read_only_node, prepared, paths)
    assert plan.kind == "cloud_hypervisor"


def test_cloud_hypervisor_runner_requires_explicit_credential_inheritance(
    tmp_path: Path,
):
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
        }
    )
    prepared = PreparedExecution(
        command=["codex", "exec", "hi"],
        env={},
        cwd="/workspace",
        trace_kind="codex",
        runtime_symlinks={"codex_home/auth.json": str(tmp_path / "auth.json")},
    )

    with pytest.raises(
        ValueError, match=r"do not expose.*credentials.*inherit_credentials"
    ):
        CloudHypervisorRunner().plan_execution(node, prepared, _paths(tmp_path))


def test_cloud_hypervisor_runtime_copies_inherited_credentials_privately(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    paths.host_workdir.mkdir(parents=True)
    paths.host_runtime_dir.mkdir()
    auth = tmp_path / "auth.json"
    auth.write_text('{"token":"secret"}', encoding="utf-8")
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": "vmlinux",
            "rootfs": "rootfs",
            "inherit_credentials": True,
        }
    )
    prepared = PreparedExecution(
        command=["true"],
        env={},
        cwd="/workspace",
        trace_kind="codex",
        runtime_files={"config/settings.json": "{}"},
        runtime_symlinks={"codex_home/auth.json": str(auth)},
    )
    runner = CloudHypervisorRunner()

    runner._prepare_runtime(node.target, prepared, paths)

    credential = paths.host_runtime_dir / "codex_home" / "auth.json"
    generated = paths.host_runtime_dir / "config" / "settings.json"
    assert credential.read_text(encoding="utf-8") == '{"token":"secret"}'
    assert credential.stat().st_mode & 0o777 == 0o600
    assert generated.stat().st_mode & 0o777 == 0o600
    assert paths.host_runtime_dir.stat().st_mode & 0o777 == 0o700
    assert not credential.is_symlink()


def test_cloud_hypervisor_console_log_replaces_guest_symlink_without_following(
    tmp_path: Path,
):
    paths = _paths(tmp_path)
    paths.host_runtime_dir.mkdir(parents=True)
    runner = CloudHypervisorRunner()
    state_dir = runner._state_dir(paths)
    state_dir.mkdir(mode=0o700)
    (state_dir / "console.log").write_text("guest console\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n", encoding="utf-8")
    destination = paths.host_runtime_dir / "cloud-hypervisor-console.log"
    destination.symlink_to(victim)

    try:
        runner._copy_console_log(paths)
    finally:
        runner._cleanup_state_dir(paths)

    assert victim.read_text(encoding="utf-8") == "keep me\n"
    assert destination.read_text(encoding="utf-8") == "guest console\n"
    assert not destination.is_symlink()
    assert destination.stat().st_mode & 0o777 == 0o600


def test_cloud_hypervisor_loader_resolves_kernel_rootfs_and_mount_sources(
    tmp_path: Path,
):
    pipeline = load_pipeline_from_text(
        json.dumps(
            {
                "name": "cloud-hypervisor-loader",
                "working_dir": "repo",
                "nodes": [
                    {
                        "id": "vm",
                        "agent": "shell",
                        "prompt": "true",
                        "target": {
                            "kind": "cloud_hypervisor",
                            "kernel": "assets/vmlinux",
                            "rootfs": "assets/rootfs",
                            "mounts": [{"source": "fixtures", "target": "/inputs"}],
                        },
                    }
                ],
            }
        ),
        base_dir=tmp_path,
    )

    target = pipeline.nodes[0].target
    assert isinstance(target, CloudHypervisorTarget)
    assert target.kernel == str((tmp_path / "repo" / "assets" / "vmlinux").resolve())
    assert target.rootfs == str((tmp_path / "repo" / "assets" / "rootfs").resolve())
    assert target.mounts[0].source == str((tmp_path / "repo" / "fixtures").resolve())


def test_cloud_hypervisor_execution_paths_use_guest_mounts(tmp_path: Path):
    target = CloudHypervisorTarget(
        kernel="vmlinux",
        rootfs="rootfs",
        workdir_mount="/repo",
        runtime_mount="/runtime-alt",
    )

    paths = build_execution_paths(
        base_dir=tmp_path / "runs",
        pipeline_workdir=tmp_path / "workspace",
        run_id="run-1",
        node_id="vm",
        node_target=target,
    )

    assert paths.host_workdir == tmp_path / "workspace"
    assert paths.target_workdir == "/repo"
    assert paths.target_runtime_dir == "/runtime-alt"


def test_cloud_hypervisor_runner_is_registered():
    assert isinstance(RunnerRegistry().get("cloud_hypervisor"), CloudHypervisorRunner)


def test_cloud_hypervisor_execute_reports_launch_validation_failure(
    tmp_path: Path, monkeypatch
):
    runner = CloudHypervisorRunner()

    def fail_validation(_target, _paths, _shares) -> None:
        raise FileNotFoundError("missing test kernel")

    monkeypatch.setattr(runner, "_validate_execution_host", fail_validation)
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": str(tmp_path / "vmlinux"),
            "rootfs": str(tmp_path / "rootfs"),
        }
    )
    prepared = PreparedExecution(
        command=["true"], env={}, cwd="/workspace", trace_kind="shell"
    )
    output: list[tuple[str, str]] = []

    async def on_output(stream: str, line: str) -> None:
        output.append((stream, line))

    result = asyncio.run(
        runner.execute(node, prepared, _paths(tmp_path), on_output, lambda: False)
    )

    assert result.exit_code == 1
    assert result.stderr_lines == [
        "Cloud Hypervisor launch validation failed: missing test kernel"
    ]
    assert output == [("stderr", result.stderr_lines[0])]


def test_cloud_hypervisor_runner_handles_hybrid_vsock_acknowledgement(
    tmp_path: Path, monkeypatch
):
    runner = CloudHypervisorRunner()
    paths = _paths(tmp_path)
    state_dir = Path(tempfile.mkdtemp(prefix="agentflow-ch-test-"))
    monkeypatch.setattr(runner, "_state_dir", lambda paths: state_dir)
    target = CloudHypervisorTarget(kernel="vmlinux", rootfs="rootfs")
    received: list[bytes] = []

    class FakeVmm:
        returncode = None

    async def scenario() -> None:
        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            received.append(await reader.readline())
            writer.write(b"OK 1073741824\n")
            writer.write(b'{"event":"hello","protocol":1}\n')
            await writer.drain()
            await reader.read()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=state_dir / "vsock.sock")
        try:
            loop = asyncio.get_running_loop()
            _, writer = await runner._connect_guest(
                target,
                paths,
                FakeVmm(),
                lambda: False,
                loop.time() + 5,
            )
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0)
        finally:
            server.close()
            await server.wait_closed()

    try:
        asyncio.run(scenario())
    finally:
        shutil.rmtree(state_dir)

    assert received == [b"CONNECT 4050\n"]


def test_cloud_hypervisor_runner_reassembles_fragmented_guest_lines(
    tmp_path: Path,
):
    runner = CloudHypervisorRunner()
    guest_events = (
        {"event": "started", "pid": 42},
        {
            "event": "stream",
            "stream": "stdout",
            "text": "first ",
            "line_end": False,
        },
        {
            "event": "stream",
            "stream": "stdout",
            "text": "line",
            "line_end": True,
        },
        {"event": "result", "exit_code": 0},
    )

    class FakeWriter:
        def __init__(self) -> None:
            self.data = b""

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            return None

    writer = FakeWriter()
    streamed: list[tuple[str, str]] = []

    async def scenario():
        reader = asyncio.StreamReader()
        for event in guest_events:
            reader.feed_data(json.dumps(event).encode("utf-8") + b"\n")
        reader.feed_eof()

        async def on_output(stream: str, line: str) -> None:
            streamed.append((stream, line))

        return await runner._consume_guest(
            reader,
            writer,
            {"protocol": 1},
            on_output,
            lambda: False,
            asyncio.get_running_loop().time() + 5,
        )

    result = asyncio.run(scenario())

    assert result.exit_code == 0
    assert result.stdout_lines == ["first line"]
    assert streamed == [("stdout", "first line")]
    assert writer.data == b'{"protocol":1}\n'


@pytest.mark.parametrize(
    "event",
    [
        {"event": "stream", "stream": "invalid", "text": "value"},
        {"event": "started", "pid": True},
        {"event": "started", "pid": 0},
        {"event": "result", "exit_code": True},
    ],
)
def test_cloud_hypervisor_runner_rejects_malformed_guest_events(
    event: dict[str, object],
):
    runner = CloudHypervisorRunner()

    class FakeWriter:
        def write(self, data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    streamed: list[tuple[str, str]] = []

    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data(json.dumps(event).encode("utf-8") + b"\n")
        reader.feed_eof()

        async def on_output(stream: str, line: str) -> None:
            streamed.append((stream, line))

        return await runner._consume_guest(
            reader,
            FakeWriter(),
            {"protocol": 1},
            on_output,
            lambda: False,
            asyncio.get_running_loop().time() + 5,
        )

    result = asyncio.run(scenario())

    assert result.exit_code == 1
    assert len(result.stderr_lines) == 1
    assert streamed == [("stderr", result.stderr_lines[0])]


def test_codex_cloud_hypervisor_credentials_are_opt_in(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text("model = 'gpt-5'\n", encoding="utf-8")
    (codex_home / "auth.json").write_text('{"token":"secret"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    paths = _paths(tmp_path)

    isolated = _node(
        {"kind": "cloud_hypervisor", "kernel": "vmlinux", "rootfs": "rootfs"},
        agent="codex",
    )
    inherited = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": "vmlinux",
            "rootfs": "rootfs",
            "inherit_credentials": True,
        },
        agent="codex",
    )

    isolated_prepared = CodexAdapter().prepare(isolated, "hi", paths)
    inherited_prepared = CodexAdapter().prepare(inherited, "hi", paths)

    assert isolated_prepared.runtime_symlinks == {}
    assert "CODEX_HOME" not in isolated_prepared.env
    assert inherited_prepared.runtime_symlinks == {
        "codex_home/config.toml": str(codex_home / "config.toml"),
        "codex_home/auth.json": str(codex_home / "auth.json"),
    }
    assert inherited_prepared.env["CODEX_HOME"] == "/agentflow-runtime/codex_home"


def test_cloud_hypervisor_inspection_redacts_env_and_warns_for_tap_and_credentials(
    tmp_path: Path,
):
    pipeline = PipelineSpec.model_validate(
        {
            "name": "inspect-cloud-hypervisor",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "vm",
                    "agent": "shell",
                    "prompt": "echo hi",
                    "env": {"DATABASE_URL": "postgres://secret"},
                    "target": {
                        "kind": "cloud_hypervisor",
                        "kernel": str(tmp_path / "vmlinux"),
                        "rootfs": str(tmp_path / "rootfs"),
                        "workdir_read_only": True,
                        "inherit_credentials": True,
                        "network_policy": "tap",
                        "seccomp": "false",
                    },
                }
            ],
        }
    )

    report = build_launch_inspection(pipeline, runs_dir=str(tmp_path / "runs"))
    node = report["nodes"][0]

    assert node["launch"]["kind"] == "cloud_hypervisor"
    assert node["prepared"]["env"] == {"DATABASE_URL": "<redacted>"}
    assert "postgres://secret" not in json.dumps(node)
    assert any("credential inheritance" in warning for warning in node["warnings"])
    assert any("TAP networking" in warning for warning in node["warnings"])
    assert any(
        "seccomp filtering is disabled" in warning for warning in node["warnings"]
    )


def test_guest_request_validation_rejects_system_and_overlapping_mounts():
    base = {
        "protocol": 1,
        "command": ["true"],
        "env": {},
        "cwd": "/workspace",
        "stdin": None,
        "uid": 0,
        "gid": 0,
        "network": {"mode": "none"},
    }
    with pytest.raises(GuestRequestError, match="reserved guest path"):
        _validate_request(
            {
                **base,
                "mounts": [
                    {"tag": "agentflow-bad", "target": "/proc/data", "read_only": True}
                ],
            }
        )
    with pytest.raises(GuestRequestError, match="overlaps another"):
        _validate_mounts(
            [
                {"tag": "agentflow-one", "target": "/inputs", "read_only": True},
                {"tag": "agentflow-two", "target": "/inputs/nested", "read_only": True},
            ]
        )


def test_guest_dhcp_installs_dns_through_writable_runtime_files(
    tmp_path: Path, monkeypatch
):
    resolv_conf = tmp_path / "agentflow-resolv.conf"
    dhcp_config = tmp_path / "agentflow-udhcpc.conf"
    commands: list[tuple[list[str], str]] = []

    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._RUNTIME_RESOLV_CONF", resolv_conf
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._RUNTIME_DHCP_CONFIG", dhcp_config
    )

    def run_checked(command: list[str], *, description: str) -> None:
        commands.append((command, description))
        if command[0] == "udhcpc":
            resolv_conf.write_text("nameserver 192.0.2.1\n", encoding="utf-8")

    monkeypatch.setattr("agentflow.cloud_hypervisor_guest._run_checked", run_checked)

    assert _configure_dhcp() is True
    assert dhcp_config.read_text(encoding="utf-8") == (f'RESOLV_CONF="{resolv_conf}"\n')
    assert commands == [
        (
            [
                "mount",
                "--bind",
                str(dhcp_config),
                "/etc/udhcpc/udhcpc.conf",
            ],
            "installing read-only-rootfs DHCP config",
        ),
        (["udhcpc", "-q", "-n", "-i", "eth0"], "guest DHCP configuration"),
        (
            ["mount", "--bind", str(resolv_conf), "/etc/resolv.conf"],
            "installing guest DNS config",
        ),
    ]


def test_guest_explicit_dns_replaces_dhcp_nameservers_without_rebinding(
    tmp_path: Path, monkeypatch
):
    resolv_conf = tmp_path / "agentflow-resolv.conf"
    commands: list[list[str]] = []

    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._RUNTIME_RESOLV_CONF", resolv_conf
    )

    def configure_dhcp() -> bool:
        resolv_conf.write_text("nameserver 192.0.2.1\n", encoding="utf-8")
        return True

    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._configure_dhcp", configure_dhcp
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._run_checked",
        lambda command, *, description: commands.append(command),
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest.subprocess.run",
        lambda *args, **kwargs: None,
    )

    _configure_network({"mode": "tap", "dhcp": True, "dns": ["1.1.1.1", "9.9.9.9"]})

    assert resolv_conf.read_text(encoding="utf-8") == (
        "nameserver 1.1.1.1\nnameserver 9.9.9.9\n"
    )
    assert commands == [["ip", "link", "set", "eth0", "up"]]


def test_guest_privilege_drop_uses_absolute_launcher_with_empty_environment():
    command, launcher_env = _command_for_identity(
        ["tool=with-equals", "argument"],
        {
            "PATH": "/workspace/untrusted",
            "LD_PRELOAD": "/workspace/untrusted.so",
            "TOKEN": "secret",
        },
        1000,
        1000,
    )

    assert command[:6] == [
        "/sbin/su-exec",
        "1000:1000",
        "/usr/bin/env",
        "-i",
        "--",
        "PATH=/workspace/untrusted",
    ]
    assert "LD_PRELOAD=/workspace/untrusted.so" in command
    assert command[-6:] == [
        "/bin/sh",
        "-c",
        'exec -- "$@"',
        "agentflow-command",
        "tool=with-equals",
        "argument",
    ]
    assert launcher_env == {}


def test_guest_agent_executes_and_multiplexes_stdout_stderr(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._mount_shares", lambda mounts: None
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._configure_network", lambda network: None
    )
    home = tmp_path / "home"
    home.mkdir()
    request = {
        "command": [
            sys.executable,
            "-c",
            "import sys; data=sys.stdin.read(); print('out:'+data.strip()); print('err-line', file=sys.stderr)",
        ],
        "env": {"HOME": str(home)},
        "cwd": str(tmp_path),
        "stdin": "payload\n",
        "uid": 0,
        "gid": 0,
        "mounts": [],
        "network": {"mode": "none"},
    }
    stream = io.BytesIO()

    exit_code = _execute_request(request, stream)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert exit_code == 0
    assert events[0]["event"] == "started"
    assert {
        event.get("text") for event in events if event.get("event") == "stream"
    } == {
        "out:payload",
        "err-line",
    }


def test_guest_agent_chunks_large_unbroken_output_below_protocol_reader_limit(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._mount_shares", lambda mounts: None
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._configure_network", lambda network: None
    )
    request = {
        "command": [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'\\0' * 20000)",
        ],
        "env": {"HOME": str(tmp_path)},
        "cwd": str(tmp_path),
        "stdin": None,
        "uid": 0,
        "gid": 0,
        "mounts": [],
        "network": {"mode": "none"},
    }
    stream = io.BytesIO()

    assert _execute_request(request, stream) == 0

    encoded_events = stream.getvalue().splitlines()
    assert max(map(len, encoded_events)) < 64 * 1024
    events = [json.loads(line) for line in encoded_events]
    stream_events = [event for event in events if event.get("event") == "stream"]
    assert any(event["line_end"] is False for event in stream_events)
    assert stream_events[-1]["line_end"] is True
    stdout = "".join(
        event["text"]
        for event in events
        if event.get("event") == "stream" and event.get("stream") == "stdout"
    )
    assert stdout == "\0" * 20000


def test_guest_agent_preserves_utf8_across_large_output_chunks(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._mount_shares", lambda mounts: None
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._configure_network", lambda network: None
    )
    expected = "a" * 8191 + "é" + "tail"
    request = {
        "command": [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.buffer.write({expected.encode()!r})",
        ],
        "env": {"HOME": str(tmp_path)},
        "cwd": str(tmp_path),
        "stdin": None,
        "uid": 0,
        "gid": 0,
        "mounts": [],
        "network": {"mode": "none"},
    }
    stream = io.BytesIO()

    assert _execute_request(request, stream) == 0

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    stdout = "".join(
        event["text"]
        for event in events
        if event.get("event") == "stream" and event.get("stream") == "stdout"
    )
    assert stdout == expected


def test_guest_agent_does_not_wait_for_child_inheriting_output_pipes(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._mount_shares", lambda mounts: None
    )
    monkeypatch.setattr(
        "agentflow.cloud_hypervisor_guest._configure_network", lambda network: None
    )
    monkeypatch.setattr("agentflow.cloud_hypervisor_guest._STREAM_DRAIN_SECONDS", 0.05)
    request = {
        "command": [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "print(child.pid, flush=True)"
            ),
        ],
        "env": {"HOME": str(tmp_path)},
        "cwd": str(tmp_path),
        "stdin": None,
        "uid": 0,
        "gid": 0,
        "mounts": [],
        "network": {"mode": "none"},
    }
    stream = io.BytesIO()
    started = time.monotonic()
    assert _execute_request(request, stream) == 0
    elapsed = time.monotonic() - started

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    child_pid = int(
        next(
            event["text"]
            for event in events
            if event.get("event") == "stream" and event.get("stream") == "stdout"
        )
    )
    try:
        os.kill(child_pid, 9)
    except ProcessLookupError:
        pass

    assert elapsed < 1.0


def test_cloud_hypervisor_live_kvm_smoke(tmp_path: Path):
    if os.environ.get("AGENTFLOW_CH_LIVE_TEST") != "1":
        pytest.skip("set AGENTFLOW_CH_LIVE_TEST=1 with kernel and rootfs paths")

    kernel = os.environ["AGENTFLOW_CH_KERNEL"]
    rootfs = os.environ["AGENTFLOW_CH_ROOTFS"]
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    app = tmp_path / "app"
    for directory in (workspace, runtime, inputs, outputs, app):
        directory.mkdir()
    (inputs / "value.txt").write_text("mounted input\n", encoding="utf-8")
    fake_launcher = workspace / "su-exec"
    fake_launcher.write_text(
        '#!/bin/sh\ntouch /workspace/fake-su-exec-used\nexec /sbin/su-exec "$@"\n',
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)
    paths = ExecutionPaths(
        host_workdir=workspace,
        host_runtime_dir=runtime,
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=app,
    )
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": kernel,
            "rootfs": rootfs,
            "binary": os.environ.get("AGENTFLOW_CH_BINARY", "cloud-hypervisor"),
            "virtiofsd": os.environ.get("AGENTFLOW_CH_VIRTIOFSD", "virtiofsd"),
            "cpus": 1,
            "memory_mib": 512,
            "mounts": [
                {"source": str(inputs), "target": "/inputs", "read_only": True},
                {"source": str(outputs), "target": "/outputs", "read_only": False},
            ],
        },
        env={
            "PROBE": "forwarded",
            "PATH": "/workspace:/opt/agentflow-venv/bin:/usr/local/bin:/usr/bin:/bin",
        },
    )
    prepared = PreparedExecution(
        command=[
            "python3",
            "-c",
            (
                "import os, pathlib, pwd, sys; "
                "data=sys.stdin.read().strip(); "
                "pathlib.Path('vm-owned.txt').write_text(data); "
                "pathlib.Path('/outputs/result.txt').write_text("
                "pathlib.Path('/inputs/value.txt').read_text()); "
                "print(f'{pwd.getpwuid(os.getuid()).pw_name}:{os.getuid()}:{os.getgid()}:{os.environ[\"PROBE\"]}:{data}'); "
                "print('a' * 8191 + 'é' + 'tail', end=''); "
                "print('guest-stderr', file=sys.stderr)"
            ),
        ],
        env={
            "PROBE": "forwarded",
            "PATH": "/workspace:/opt/agentflow-venv/bin:/usr/local/bin:/usr/bin:/bin",
        },
        cwd="/workspace",
        trace_kind="shell",
        stdin="stdin payload\n",
    )
    streamed: dict[str, list[str]] = {"stdout": [], "stderr": []}

    async def on_output(stream: str, line: str) -> None:
        streamed[stream].append(line)

    result = asyncio.run(
        CloudHypervisorRunner().execute(node, prepared, paths, on_output, lambda: False)
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.cancelled is False
    assert any(
        line.endswith(":forwarded:stdin payload") for line in result.stdout_lines
    )
    assert "a" * 8191 + "é" + "tail" in result.stdout_lines
    assert "guest-stderr" in result.stderr_lines
    assert streamed["stdout"] == result.stdout_lines
    assert streamed["stderr"] == result.stderr_lines
    assert (workspace / "vm-owned.txt").read_text(encoding="utf-8") == "stdin payload"
    assert (outputs / "result.txt").read_text(encoding="utf-8") == "mounted input\n"
    assert not (workspace / "fake-su-exec-used").exists()
    if hasattr(os, "getuid"):
        assert (workspace / "vm-owned.txt").stat().st_uid == os.getuid()
        assert (outputs / "result.txt").stat().st_uid == os.getuid()


def test_cloud_hypervisor_live_cancellation_cleans_processes_and_state(
    tmp_path: Path,
):
    if os.environ.get("AGENTFLOW_CH_LIVE_TEST") != "1":
        pytest.skip("set AGENTFLOW_CH_LIVE_TEST=1 with kernel and rootfs paths")

    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    app = tmp_path / "app"
    for directory in (workspace, runtime, app):
        directory.mkdir()
    paths = ExecutionPaths(
        host_workdir=workspace,
        host_runtime_dir=runtime,
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=app,
    )
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": os.environ["AGENTFLOW_CH_KERNEL"],
            "rootfs": os.environ["AGENTFLOW_CH_ROOTFS"],
            "binary": os.environ.get("AGENTFLOW_CH_BINARY", "cloud-hypervisor"),
            "virtiofsd": os.environ.get("AGENTFLOW_CH_VIRTIOFSD", "virtiofsd"),
            "cpus": 1,
            "memory_mib": 512,
        }
    )
    prepared = PreparedExecution(
        command=["sh", "-c", "printf 'command-started\\n'; sleep 30"],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )
    cancel_requested = False

    async def on_output(stream: str, line: str) -> None:
        nonlocal cancel_requested
        if stream == "stdout" and line == "command-started":
            cancel_requested = True

    runner = CloudHypervisorRunner()
    result = asyncio.run(
        runner.execute(node, prepared, paths, on_output, lambda: cancel_requested)
    )

    assert result.exit_code == 130
    assert result.cancelled is True
    assert not runner._state_dir(paths).exists()


def test_cloud_hypervisor_live_precreated_tap_connectivity(tmp_path: Path):
    tap = os.environ.get("AGENTFLOW_CH_LIVE_TAP")
    if os.environ.get("AGENTFLOW_CH_LIVE_TEST") != "1" or not tap:
        pytest.skip("set AGENTFLOW_CH_LIVE_TEST=1 and AGENTFLOW_CH_LIVE_TAP")

    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    app = tmp_path / "app"
    for directory in (workspace, runtime, app):
        directory.mkdir()
    paths = ExecutionPaths(
        host_workdir=workspace,
        host_runtime_dir=runtime,
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=app,
    )
    host_address = os.environ.get("AGENTFLOW_CH_LIVE_TAP_HOST", "192.0.2.1")
    guest_address = os.environ.get("AGENTFLOW_CH_LIVE_TAP_GUEST", "192.0.2.2/30")
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": os.environ["AGENTFLOW_CH_KERNEL"],
            "rootfs": os.environ["AGENTFLOW_CH_ROOTFS"],
            "binary": os.environ.get("AGENTFLOW_CH_BINARY", "cloud-hypervisor"),
            "virtiofsd": os.environ.get("AGENTFLOW_CH_VIRTIOFSD", "virtiofsd"),
            "cpus": 1,
            "memory_mib": 512,
            "user": "root",
            "network_policy": {
                "mode": "tap",
                "tap": tap,
                "guest_address": guest_address,
                "num_queues": 2,
            },
        }
    )
    prepared = PreparedExecution(
        command=["ping", "-c", "1", "-W", "2", host_address],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    async def on_output(stream: str, line: str) -> None:
        return None

    result = asyncio.run(
        CloudHypervisorRunner().execute(node, prepared, paths, on_output, lambda: False)
    )

    assert result.exit_code == 0, result.stderr_lines
    assert any(host_address in line for line in result.stdout_lines)


def test_cloud_hypervisor_live_precreated_tap_dhcp_and_dns(tmp_path: Path):
    tap = os.environ.get("AGENTFLOW_CH_LIVE_DHCP_TAP")
    if os.environ.get("AGENTFLOW_CH_LIVE_TEST") != "1" or not tap:
        pytest.skip("set AGENTFLOW_CH_LIVE_TEST=1 and AGENTFLOW_CH_LIVE_DHCP_TAP")

    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    app = tmp_path / "app"
    for directory in (workspace, runtime, app):
        directory.mkdir()
    paths = ExecutionPaths(
        host_workdir=workspace,
        host_runtime_dir=runtime,
        target_workdir="/workspace",
        target_runtime_dir="/agentflow-runtime",
        app_root=app,
    )
    host_address = os.environ.get("AGENTFLOW_CH_LIVE_DHCP_HOST", "192.0.2.1")
    guest_prefix = os.environ.get("AGENTFLOW_CH_LIVE_DHCP_GUEST_PREFIX", "192.0.2.")
    dns_address = os.environ.get("AGENTFLOW_CH_LIVE_DHCP_DNS", host_address)
    node = _node(
        {
            "kind": "cloud_hypervisor",
            "kernel": os.environ["AGENTFLOW_CH_KERNEL"],
            "rootfs": os.environ["AGENTFLOW_CH_ROOTFS"],
            "binary": os.environ.get("AGENTFLOW_CH_BINARY", "cloud-hypervisor"),
            "virtiofsd": os.environ.get("AGENTFLOW_CH_VIRTIOFSD", "virtiofsd"),
            "cpus": 1,
            "memory_mib": 512,
            "user": "root",
            "network_policy": tap,
        }
    )
    prepared = PreparedExecution(
        command=[
            "sh",
            "-c",
            'ip -4 -o address show dev eth0; cat /etc/resolv.conf; ping -c 1 -W 2 "$1"',
            "agentflow-dhcp-probe",
            host_address,
        ],
        env={},
        cwd="/workspace",
        trace_kind="shell",
    )

    async def on_output(stream: str, line: str) -> None:
        return None

    result = asyncio.run(
        CloudHypervisorRunner().execute(node, prepared, paths, on_output, lambda: False)
    )

    assert result.exit_code == 0, result.stderr_lines
    assert any(guest_prefix in line for line in result.stdout_lines)
    assert f"nameserver {dns_address}" in result.stdout_lines
    assert any(host_address in line for line in result.stdout_lines)
