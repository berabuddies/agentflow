"""Cloud Hypervisor execution through virtio-fs and a small vsock guest agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.runners.base import (
    CancelCallback,
    LaunchPlan,
    RawExecutionResult,
    Runner,
    StreamCallback,
)
from agentflow.specs import CloudHypervisorTarget, NodeSpec

_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_GUEST_EVENT_BYTES = 256 * 1024
_MAX_LOGICAL_STREAM_LINE_CHARS = 16 * 1024 * 1024
_MAX_STREAM_FRAGMENTS_PER_LINE = 4096
_ROOTFS_TAG = "/dev/root"
_MAX_UNIX_SOCKET_PATH_BYTES = 100
_NSS_DIRECTORY = ".agentflow-nss"


@dataclass(frozen=True, slots=True)
class _VirtioFsShare:
    tag: str
    source: Path
    target: str | None
    read_only: bool
    socket: Path


class CloudHypervisorRunner(Runner):
    """Boot one ephemeral VM per node and execute through a vsock protocol."""

    def _target(self, node: NodeSpec) -> CloudHypervisorTarget:
        target = node.target
        if not isinstance(target, CloudHypervisorTarget):
            raise TypeError("CloudHypervisorRunner requires a CloudHypervisorTarget")
        return target

    def _resolve_host_path(self, value: str | Path, paths: ExecutionPaths) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = paths.host_workdir / candidate
        return candidate.resolve()

    def _host_paths_overlap(self, left: Path, right: Path) -> bool:
        left = left.resolve()
        right = right.resolve()
        try:
            left.relative_to(right)
            return True
        except ValueError:
            pass
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False

    def _state_dir(self, paths: ExecutionPaths) -> Path:
        identity = str(paths.host_runtime_dir.resolve()).encode("utf-8")
        suffix = hashlib.sha256(identity).hexdigest()[:20]
        return Path(tempfile.gettempdir()) / f"agentflow-ch-{suffix}"

    def _socket_path(self, state_dir: Path, name: str) -> Path:
        path = state_dir / name
        if "," in str(path):
            raise ValueError(
                f"Cloud Hypervisor Unix socket paths must not contain commas: {path}"
            )
        if len(os.fsencode(path)) > _MAX_UNIX_SOCKET_PATH_BYTES:
            raise ValueError(
                "Cloud Hypervisor runtime path is too long for Unix sockets; configure a shorter "
                f"AgentFlow base directory (socket would be `{path}`)"
            )
        return path

    def _effective_guest_ids(self, target: CloudHypervisorTarget) -> tuple[int, int]:
        if target.user == "host":
            if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
                raise ValueError("`target.user: host` requires a POSIX host")
            return os.getuid(), os.getgid()
        if target.user in {None, "root", "0", "0:0"}:
            return 0, 0
        assert target.user is not None
        uid_text, separator, gid_text = target.user.partition(":")
        uid = int(uid_text)
        return uid, int(gid_text) if separator else uid

    def _vsock_cid(self, target: CloudHypervisorTarget, paths: ExecutionPaths) -> int:
        if target.vsock_cid is not None:
            return target.vsock_cid
        identity = str(paths.host_runtime_dir.resolve()).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
        return 3 + (value % (4_294_967_295 - 3))

    def _shares(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
    ) -> list[_VirtioFsShare]:
        state_dir = self._state_dir(paths)
        rootfs_source = self._resolve_host_path(target.rootfs, paths)
        workspace_source = paths.host_workdir.resolve()
        runtime_source = paths.host_runtime_dir.resolve()
        if not target.workdir_read_only and self._host_paths_overlap(
            workspace_source, rootfs_source
        ):
            raise ValueError(
                "the writable managed workspace overlaps the immutable guest rootfs; "
                "move the rootfs outside the workspace or set `target.workdir_read_only: true`"
            )
        if self._host_paths_overlap(runtime_source, rootfs_source):
            raise ValueError(
                "the writable managed runtime overlaps the immutable guest rootfs; "
                "move the rootfs or AgentFlow runs directory"
            )
        if (
            target.app_mount is not None
            and not target.workdir_read_only
            and self._host_paths_overlap(workspace_source, paths.app_root)
        ):
            raise ValueError(
                "the writable managed workspace overlaps the read-only managed AgentFlow app; "
                "omit `target.app_mount` or use a read-only/separate workspace"
            )
        raw_shares: list[tuple[str, Path, str | None, bool]] = [
            (
                _ROOTFS_TAG,
                rootfs_source,
                None,
                True,
            ),
            (
                "agentflow-workspace",
                workspace_source,
                target.workdir_mount,
                target.workdir_read_only,
            ),
            (
                "agentflow-runtime",
                runtime_source,
                target.runtime_mount,
                False,
            ),
        ]
        if target.app_mount is not None:
            raw_shares.append(
                (
                    "agentflow-app",
                    paths.app_root.resolve(),
                    target.app_mount,
                    True,
                )
            )
        explicit_sources: list[tuple[Path, bool]] = []
        for index, mount in enumerate(target.mounts):
            source = self._resolve_host_path(mount.source, paths)
            if not mount.read_only and self._host_paths_overlap(source, rootfs_source):
                raise ValueError(
                    "a read-write Cloud Hypervisor mount source overlaps the immutable guest rootfs: "
                    f"{source}"
                )
            if (
                not mount.read_only
                and target.workdir_read_only
                and self._host_paths_overlap(source, paths.host_workdir)
            ):
                raise ValueError(
                    "a read-write Cloud Hypervisor mount source overlaps the read-only managed workspace: "
                    f"{source}"
                )
            if (
                not mount.read_only
                and target.app_mount is not None
                and self._host_paths_overlap(source, paths.app_root)
            ):
                raise ValueError(
                    "a read-write Cloud Hypervisor mount source overlaps the read-only managed AgentFlow app: "
                    f"{source}"
                )
            if (
                mount.read_only
                and not target.workdir_read_only
                and self._host_paths_overlap(source, paths.host_workdir)
            ):
                raise ValueError(
                    "a read-only Cloud Hypervisor mount source overlaps the writable managed workspace: "
                    f"{source}"
                )
            if mount.read_only and self._host_paths_overlap(
                source, paths.host_runtime_dir
            ):
                raise ValueError(
                    "a read-only Cloud Hypervisor mount source overlaps the writable managed runtime: "
                    f"{source}"
                )
            for previous_source, previous_read_only in explicit_sources:
                if mount.read_only != previous_read_only and self._host_paths_overlap(
                    source, previous_source
                ):
                    raise ValueError(
                        "read-only and read-write Cloud Hypervisor mount sources must not overlap: "
                        f"{previous_source} <> {source}"
                    )
            explicit_sources.append((source, mount.read_only))
            raw_shares.append(
                (
                    f"agentflow-mount-{index:03d}",
                    source,
                    mount.target,
                    mount.read_only,
                )
            )
        return [
            _VirtioFsShare(
                tag=tag,
                source=source,
                target=guest_target,
                read_only=read_only,
                socket=self._socket_path(state_dir, f"fs-{index:03d}.sock"),
            )
            for index, (tag, source, guest_target, read_only) in enumerate(raw_shares)
        ]

    def _kernel_cmdline(self, target: CloudHypervisorTarget) -> str:
        arguments = [
            "console=ttyS0",
            f"root={_ROOTFS_TAG}",
            "rootfstype=virtiofs",
            "ro",
            f"init={target.init_path}",
            f"agentflow.guest_port={target.guest_agent_port}",
            "panic=1",
            "reboot=k",
            *target.kernel_args,
        ]
        return " ".join(arguments)

    def _network_argument(self, target: CloudHypervisorTarget) -> str | None:
        policy = target.network_policy
        if policy.mode == "none":
            return None
        parts = [f"tap={policy.tap or ''}"]
        if policy.mac is not None:
            parts.append(f"mac={policy.mac}")
        if policy.host_ip is not None:
            parts.append(f"ip={policy.host_ip}")
        if policy.host_mask is not None:
            parts.append(f"mask={policy.host_mask}")
        parts.append(f"num_queues={policy.num_queues}")
        return ",".join(parts)

    def _vmm_command(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
        shares: list[_VirtioFsShare],
    ) -> list[str]:
        state_dir = self._state_dir(paths)
        command = [
            target.binary,
            "--api-socket",
            f"path={self._socket_path(state_dir, 'api.sock')}",
            "--kernel",
            str(self._resolve_host_path(target.kernel, paths)),
            "--cmdline",
            self._kernel_cmdline(target),
            "--cpus",
            f"boot={target.cpus}",
            "--memory",
            f"size={target.memory_mib}M,shared=on",
            "--console",
            "off",
            "--serial",
            f"file={state_dir / 'console.log'}",
            "--seccomp",
            target.seccomp,
            "--vsock",
            f"cid={self._vsock_cid(target, paths)},socket={self._socket_path(state_dir, 'vsock.sock')}",
        ]
        for share in shares:
            command.extend(
                [
                    "--fs",
                    f"tag={share.tag},socket={share.socket},num_queues=1,queue_size=512",
                ]
            )
        network_argument = self._network_argument(target)
        if network_argument is not None:
            command.extend(["--net", network_argument])
        return command

    def _virtiofsd_command(
        self,
        target: CloudHypervisorTarget,
        share: _VirtioFsShare,
        guest_uid: int,
        guest_gid: int,
    ) -> list[str]:
        host_uid = os.getuid() if hasattr(os, "getuid") else guest_uid
        host_gid = os.getgid() if hasattr(os, "getgid") else guest_gid
        command = [
            target.virtiofsd,
            f"--socket-path={share.socket}",
            f"--shared-dir={share.source}",
            "--cache=never",
            "--log-level=warn",
            f"--translate-uid=map:{guest_uid}:{host_uid}:1",
            f"--translate-gid=map:{guest_gid}:{host_gid}:1",
        ]
        if share.read_only:
            command.append("--readonly")
        return command

    def _prepare_private_runtime_target(
        self, base_dir: Path, relative_path: str
    ) -> Path:
        relative = Path(relative_path)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "\x00" in relative_path
        ):
            raise ValueError(
                f"runtime file path must stay below the runtime directory: {relative_path}"
            )
        root = base_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        parent = root
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent.is_symlink():
                raise ValueError(f"runtime file parent must not be a symlink: {parent}")
            parent.mkdir(exist_ok=True)
            parent.chmod(0o700)
        return parent / relative.parts[-1]

    def materialize_runtime_files(
        self, base_dir: Path, runtime_files: dict[str, str]
    ) -> None:
        for relative_path, content in runtime_files.items():
            target = self._prepare_private_runtime_target(base_dir, relative_path)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                target.unlink()
            if target.exists():
                raise ValueError(
                    f"runtime file target must not be a directory: {target}"
                )
            target.write_text(content, encoding="utf-8")
            target.chmod(0o600)

    def _materialize_inherited_credentials(
        self,
        target: CloudHypervisorTarget,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> None:
        if prepared.runtime_symlinks and not target.inherit_credentials:
            raise ValueError(
                "Cloud Hypervisor targets do not expose adapter-requested host credentials by default; "
                "set `target.inherit_credentials: true` to copy selected files into the private VM runtime"
            )
        for relative_path, source in prepared.runtime_symlinks.items():
            source_path = self._resolve_host_path(source, paths)
            if not source_path.is_file():
                raise ValueError(
                    f"inherited Cloud Hypervisor credential must be a regular file: {source_path}"
                )
            destination = self._prepare_private_runtime_target(
                paths.host_runtime_dir, relative_path
            )
            if destination.is_symlink() or (
                destination.exists() and not destination.is_dir()
            ):
                destination.unlink()
            if destination.exists():
                raise ValueError(
                    f"credential runtime target must not be a directory: {destination}"
                )
            destination.write_bytes(source_path.read_bytes())
            destination.chmod(0o600)

    def _validate_credentials_policy(
        self,
        target: CloudHypervisorTarget,
        prepared: PreparedExecution,
    ) -> None:
        if prepared.runtime_symlinks and not target.inherit_credentials:
            raise ValueError(
                "Cloud Hypervisor targets do not expose adapter-requested host credentials by default; "
                "set `target.inherit_credentials: true` to copy selected files into the private VM runtime"
            )

    def _nss_runtime_files(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
    ) -> tuple[dict[str, str], dict[str, str]]:
        guest_uid, guest_gid = self._effective_guest_ids(target)
        runtime_files: dict[str, str] = {}
        env: dict[str, str] = {}
        if guest_uid == 0 or target.nss_wrapper_path is None:
            return runtime_files, env

        passwd_relative = f"{_NSS_DIRECTORY}/passwd"
        group_relative = f"{_NSS_DIRECTORY}/group"
        runtime_files[passwd_relative] = (
            "root:x:0:0:root:/root:/bin/sh\n"
            f"agentflow:x:{guest_uid}:{guest_gid}:AgentFlow VM:{target.runtime_mount}/home:/bin/sh\n"
        )
        runtime_files[group_relative] = f"root:x:0:\nagentflow:x:{guest_gid}:\n"
        env.update(
            {
                "NSS_WRAPPER_PASSWD": str(
                    PurePosixPath(target.runtime_mount) / passwd_relative
                ),
                "NSS_WRAPPER_GROUP": str(
                    PurePosixPath(target.runtime_mount) / group_relative
                ),
                "USER": "agentflow",
                "LOGNAME": "agentflow",
            }
        )
        current_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (
            f"{target.nss_wrapper_path}:{current_preload}"
            if current_preload
            else target.nss_wrapper_path
        )
        return runtime_files, env

    def _guest_environment(
        self,
        target: CloudHypervisorTarget,
        prepared: PreparedExecution,
        nss_env: dict[str, str],
    ) -> dict[str, str]:
        env = dict(prepared.env)
        if target.app_mount is not None:
            inherited = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = (
                f"{target.app_mount}:{inherited}" if inherited else target.app_mount
            )
        env.setdefault("HOME", f"{target.runtime_mount.rstrip('/')}/home")
        guest_uid, _ = self._effective_guest_ids(target)
        if guest_uid != 0 and nss_env:
            original_preload = env.get("LD_PRELOAD", "")
            env.update(nss_env)
            if original_preload:
                env["LD_PRELOAD"] = f"{nss_env['LD_PRELOAD']}:{original_preload}"
        return env

    def _guest_request(
        self,
        target: CloudHypervisorTarget,
        prepared: PreparedExecution,
        shares: list[_VirtioFsShare],
        nss_env: dict[str, str],
    ) -> dict[str, Any]:
        guest_uid, guest_gid = self._effective_guest_ids(target)
        policy = target.network_policy
        return {
            "protocol": _PROTOCOL_VERSION,
            "command": list(prepared.command),
            "env": self._guest_environment(target, prepared, nss_env),
            "cwd": prepared.cwd,
            "stdin": prepared.stdin,
            "uid": guest_uid,
            "gid": guest_gid,
            "mounts": [
                {
                    "tag": share.tag,
                    "target": share.target,
                    "read_only": share.read_only,
                }
                for share in shares
                if share.target is not None
            ],
            "network": {
                "mode": policy.mode,
                "dhcp": policy.dhcp,
                "guest_address": policy.guest_address,
                "gateway": policy.gateway,
                "dns": list(policy.dns),
            },
        }

    def _validate_execution_host(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
        shares: list[_VirtioFsShare],
    ) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Cloud Hypervisor targets require a Linux host with KVM")
        kvm = Path("/dev/kvm")
        if not kvm.exists() or not os.access(kvm, os.R_OK | os.W_OK):
            raise PermissionError(
                "Cloud Hypervisor target requires read/write access to `/dev/kvm`"
            )
        kernel = self._resolve_host_path(target.kernel, paths)
        if not kernel.is_file():
            raise FileNotFoundError(
                f"Cloud Hypervisor kernel does not exist or is not a file: {kernel}"
            )
        for share in shares:
            if not share.source.is_dir():
                raise FileNotFoundError(
                    f"Cloud Hypervisor virtio-fs source does not exist or is not a directory: {share.source}"
                )
        for executable, field_name in (
            (target.binary, "binary"),
            (target.virtiofsd, "virtiofsd"),
        ):
            if os.path.sep in executable:
                executable_path = Path(executable).expanduser()
                if not executable_path.is_file() or not os.access(
                    executable_path, os.X_OK
                ):
                    raise FileNotFoundError(
                        f"`target.{field_name}` is not executable: {executable_path}"
                    )
            elif shutil.which(executable) is None:
                raise FileNotFoundError(
                    f"`target.{field_name}` executable was not found on PATH: {executable}"
                )

        self._validate_executable_features(
            target.binary,
            "cloud-hypervisor",
            ("--api-socket", "--fs", "--vsock", "--seccomp"),
        )
        self._validate_executable_features(
            target.virtiofsd,
            "virtiofsd",
            ("--readonly", "--translate-uid", "--translate-gid"),
        )

    def _validate_executable_features(
        self,
        executable: str,
        label: str,
        required_options: tuple[str, ...],
    ) -> None:
        try:
            result = subprocess.run(
                [executable, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"could not inspect {label} capabilities: {exc}"
            ) from exc
        help_text = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"{label} --help failed with status {result.returncode}: {detail}"
            )
        missing = [option for option in required_options if option not in help_text]
        if missing:
            raise RuntimeError(
                f"{label} is missing required options {missing}; install a current compatible release"
            )

    def _prepare_runtime(
        self,
        target: CloudHypervisorTarget,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> dict[str, str]:
        paths.host_runtime_dir.mkdir(parents=True, exist_ok=True)
        paths.host_runtime_dir.chmod(0o700)
        nss_files, nss_env = self._nss_runtime_files(target, paths)
        collisions = sorted(set(prepared.runtime_files).intersection(nss_files))
        if collisions:
            raise ValueError(
                f"adapter runtime files collide with Cloud Hypervisor NSS files: {collisions}"
            )
        self.materialize_runtime_files(
            paths.host_runtime_dir, {**prepared.runtime_files, **nss_files}
        )
        self._materialize_inherited_credentials(target, prepared, paths)
        home = paths.host_runtime_dir / "home"
        home.mkdir(parents=True, exist_ok=True)
        home.chmod(0o700)
        return nss_env

    def _open_private_log(self, path: Path) -> BinaryIO:
        if path.is_symlink():
            path.unlink()
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(
                    f"Cloud Hypervisor log path is not a regular file: {path}"
                )
            os.fchmod(descriptor, 0o600)
            if hasattr(os, "set_blocking"):
                os.set_blocking(descriptor, True)
            return os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise

    async def _start_virtiofsd(
        self,
        target: CloudHypervisorTarget,
        share: _VirtioFsShare,
        guest_uid: int,
        guest_gid: int,
        log_path: Path,
    ) -> asyncio.subprocess.Process:
        log_handle = self._open_private_log(log_path)
        try:
            return await asyncio.create_subprocess_exec(
                *self._virtiofsd_command(target, share, guest_uid, guest_gid),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
            )
        finally:
            log_handle.close()

    async def _wait_for_virtiofs_sockets(
        self,
        processes: list[asyncio.subprocess.Process],
        shares: list[_VirtioFsShare],
        timeout: float,
        should_cancel: CancelCallback,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if should_cancel():
                raise asyncio.CancelledError
            failed = next(
                (process for process in processes if process.returncode is not None),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"virtiofsd exited before Cloud Hypervisor launch with status {failed.returncode}"
                )
            if all(share.socket.exists() for share in shares):
                return
            if loop.time() >= deadline:
                raise TimeoutError(
                    "virtiofsd sockets were not ready before the boot timeout"
                )
            await asyncio.sleep(0.05)

    async def _start_vmm(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
        shares: list[_VirtioFsShare],
    ) -> asyncio.subprocess.Process:
        log_handle = self._open_private_log(
            paths.host_runtime_dir / "cloud-hypervisor-vmm.log"
        )
        try:
            return await asyncio.create_subprocess_exec(
                *self._vmm_command(target, paths, shares),
                cwd=str(paths.host_workdir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
            )
        finally:
            log_handle.close()

    async def _connect_guest(
        self,
        target: CloudHypervisorTarget,
        paths: ExecutionPaths,
        vmm: asyncio.subprocess.Process,
        should_cancel: CancelCallback,
        deadline: float,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        boot_deadline = min(deadline, loop.time() + target.boot_timeout_seconds)
        vsock_path = self._socket_path(self._state_dir(paths), "vsock.sock")
        last_connection_error: BaseException | None = None
        while loop.time() < boot_deadline:
            if should_cancel():
                raise asyncio.CancelledError
            if vmm.returncode is not None:
                detail = (
                    f"; last vsock connection error: {last_connection_error}"
                    if last_connection_error is not None
                    else ""
                )
                raise RuntimeError(
                    f"cloud-hypervisor exited during guest boot with status {vmm.returncode}{detail}"
                )
            if not vsock_path.exists():
                await asyncio.sleep(0.1)
                continue
            writer: asyncio.StreamWriter | None = None
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(
                        vsock_path, limit=_MAX_GUEST_EVENT_BYTES
                    ),
                    timeout=0.5,
                )
                writer.write(f"CONNECT {target.guest_agent_port}\n".encode("ascii"))
                await writer.drain()
                first_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                if first_line.startswith(b"OK "):
                    allocated_port = first_line[3:].strip()
                    if not allocated_port.isdigit():
                        raise ValueError(
                            f"invalid Cloud Hypervisor vsock acknowledgement: {first_line!r}"
                        )
                    hello_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                else:
                    hello_line = first_line
                hello = json.loads(hello_line)
                if hello != {"event": "hello", "protocol": _PROTOCOL_VERSION}:
                    raise ValueError(f"unexpected guest hello: {hello!r}")
                return reader, writer
            except (
                ConnectionError,
                OSError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                asyncio.TimeoutError,
            ) as exc:
                last_connection_error = exc
                if writer is not None:
                    writer.close()
                    with suppress(Exception):
                        await writer.wait_closed()
                await asyncio.sleep(0.1)
        if loop.time() >= deadline:
            raise TimeoutError("Cloud Hypervisor node timed out while booting")
        raise TimeoutError(
            f"Cloud Hypervisor guest agent did not become ready within {target.boot_timeout_seconds}s"
        )

    async def _consume_guest(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request: dict[str, Any],
        on_output: StreamCallback,
        should_cancel: CancelCallback,
        deadline: float,
    ) -> RawExecutionResult:
        request_bytes = (
            json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        partial_lines: dict[str, list[str]] = {"stdout": [], "stderr": []}
        partial_sizes = {"stdout": 0, "stderr": 0}
        partial_open = {"stdout": False, "stderr": False}
        if len(request_bytes) > _MAX_REQUEST_BYTES:
            message = "Cloud Hypervisor guest request exceeds the 16 MiB protocol limit"
            stderr_lines.append(message)
            await on_output("stderr", message)
            return RawExecutionResult(
                exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
            )
        try:
            writer.write(request_bytes)
            await writer.drain()
        except (ConnectionError, OSError) as exc:
            message = f"could not send Cloud Hypervisor guest request: {exc}"
            stderr_lines.append(message)
            await on_output("stderr", message)
            return RawExecutionResult(
                exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
            )
        loop = asyncio.get_running_loop()
        while True:
            if should_cancel():
                return RawExecutionResult(
                    exit_code=130,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    cancelled=True,
                )
            if loop.time() >= deadline:
                return RawExecutionResult(
                    exit_code=124,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    timed_out=True,
                )
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            except (OSError, ValueError) as exc:
                message = f"invalid Cloud Hypervisor guest protocol stream: {exc}"
                stderr_lines.append(message)
                await on_output("stderr", message)
                return RawExecutionResult(
                    exit_code=1,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                )
            if not line:
                message = "Cloud Hypervisor guest agent disconnected before returning a result"
                stderr_lines.append(message)
                await on_output("stderr", message)
                return RawExecutionResult(
                    exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
                )
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                message = f"invalid Cloud Hypervisor guest protocol event: {exc}"
                stderr_lines.append(message)
                await on_output("stderr", message)
                return RawExecutionResult(
                    exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
                )
            if not isinstance(event, dict):
                message = f"invalid Cloud Hypervisor guest protocol event: {event!r}"
                stderr_lines.append(message)
                await on_output("stderr", message)
                return RawExecutionResult(
                    exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
                )
            event_name = event.get("event")
            if event_name == "stream":
                stream_name = event.get("stream")
                text = event.get("text")
                line_end = event.get("line_end", True)
                if (
                    stream_name not in {"stdout", "stderr"}
                    or not isinstance(text, str)
                    or not isinstance(line_end, bool)
                ):
                    message = f"invalid Cloud Hypervisor stream event: {event!r}"
                    stderr_lines.append(message)
                    await on_output("stderr", message)
                    return RawExecutionResult(
                        exit_code=1,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                    )
                fragments = partial_lines[stream_name]
                fragments.append(text)
                partial_sizes[stream_name] += len(text)
                partial_open[stream_name] = not line_end
                if (
                    partial_sizes[stream_name] > _MAX_LOGICAL_STREAM_LINE_CHARS
                    or len(fragments) > _MAX_STREAM_FRAGMENTS_PER_LINE
                ):
                    message = f"Cloud Hypervisor {stream_name} line exceeds the guest protocol limit"
                    stderr_lines.append(message)
                    await on_output("stderr", message)
                    return RawExecutionResult(
                        exit_code=1,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                    )
                if not line_end:
                    continue
                complete_line = "".join(fragments)
                fragments.clear()
                partial_sizes[stream_name] = 0
                buffer = stdout_lines if stream_name == "stdout" else stderr_lines
                buffer.append(complete_line)
                await on_output(stream_name, complete_line)
                continue
            if event_name == "started":
                if (
                    not isinstance(event.get("pid"), int)
                    or isinstance(event.get("pid"), bool)
                    or event["pid"] <= 0
                ):
                    message = f"invalid Cloud Hypervisor started event: {event!r}"
                    stderr_lines.append(message)
                    await on_output("stderr", message)
                    return RawExecutionResult(
                        exit_code=1,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                    )
                continue
            if event_name == "result":
                if any(partial_open.values()):
                    message = (
                        "Cloud Hypervisor guest ended with an incomplete stream line"
                    )
                    stderr_lines.append(message)
                    await on_output("stderr", message)
                    return RawExecutionResult(
                        exit_code=1,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                    )
                exit_code = event.get("exit_code")
                if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                    message = f"invalid Cloud Hypervisor result event: {event!r}"
                    stderr_lines.append(message)
                    await on_output("stderr", message)
                    return RawExecutionResult(
                        exit_code=1,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                    )
                return RawExecutionResult(
                    exit_code=exit_code,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                )
            if event_name == "error":
                message = str(
                    event.get("message") or "Cloud Hypervisor guest agent failed"
                )
                stderr_lines.append(message)
                await on_output("stderr", message)
                return RawExecutionResult(
                    exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
                )
            message = f"invalid Cloud Hypervisor guest protocol event: {event!r}"
            stderr_lines.append(message)
            await on_output("stderr", message)
            return RawExecutionResult(
                exit_code=1, stdout_lines=stdout_lines, stderr_lines=stderr_lines
            )

    async def _request_vmm_shutdown(self, api_socket: Path) -> None:
        if not api_socket.exists():
            return
        writer: asyncio.StreamWriter | None = None
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(api_socket), timeout=0.5
            )
            writer.write(
                b"PUT /api/v1/vmm.shutdown HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            if writer is not None:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

    async def _stop_process(
        self, process: asyncio.subprocess.Process, timeout: float
    ) -> None:
        if process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return
        except asyncio.TimeoutError:
            pass
        with suppress(ProcessLookupError):
            process.kill()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)

    def _copy_console_log(self, paths: ExecutionPaths) -> None:
        source = self._state_dir(paths) / "console.log"
        destination = paths.host_runtime_dir / "cloud-hypervisor-console.log"
        if source.is_file():
            with (
                source.open("rb") as source_handle,
                self._open_private_log(destination) as destination_handle,
            ):
                shutil.copyfileobj(source_handle, destination_handle)

    def _cleanup_state_dir(self, paths: ExecutionPaths) -> None:
        state_dir = self._state_dir(paths)
        if state_dir.is_symlink():
            raise ValueError(
                f"Cloud Hypervisor state directory must not be a symlink: {state_dir}"
            )
        if state_dir.exists():
            if hasattr(os, "getuid") and state_dir.stat().st_uid != os.getuid():
                raise PermissionError(
                    f"refusing to remove Cloud Hypervisor state directory owned by another user: {state_dir}"
                )
            shutil.rmtree(state_dir)

    def plan_execution(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> LaunchPlan:
        target = self._target(node)
        self._validate_credentials_policy(target, prepared)
        shares = self._shares(target, paths)
        guest_uid, guest_gid = self._effective_guest_ids(target)
        nss_files, nss_env = self._nss_runtime_files(target, paths)
        request = self._guest_request(target, prepared, shares, nss_env)
        return LaunchPlan(
            kind="cloud_hypervisor",
            command=self._vmm_command(target, paths, shares),
            env={},
            cwd=str(paths.host_workdir),
            stdin=None,
            runtime_files=sorted(
                set(prepared.runtime_files)
                | set(prepared.runtime_symlinks)
                | set(nss_files)
            ),
            payload={
                "kernel": str(self._resolve_host_path(target.kernel, paths)),
                "rootfs": str(self._resolve_host_path(target.rootfs, paths)),
                "binary": target.binary,
                "virtiofsd": target.virtiofsd,
                "cpus": target.cpus,
                "memory_mib": target.memory_mib,
                "guest_user": f"{guest_uid}:{guest_gid}",
                "vsock_cid": self._vsock_cid(target, paths),
                "guest_agent_port": target.guest_agent_port,
                "network_policy": target.network_policy.model_dump(mode="json"),
                "inherit_credentials": target.inherit_credentials,
                "env_keys": sorted(request["env"]),
                "mounts": [
                    {
                        "tag": share.tag,
                        "source": str(share.source),
                        "target": share.target or "/",
                        "read_only": share.read_only,
                    }
                    for share in shares
                ],
                "virtiofsd_commands": [
                    self._virtiofsd_command(target, share, guest_uid, guest_gid)
                    for share in shares
                ],
            },
        )

    async def execute(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
        on_output: StreamCallback,
        should_cancel: CancelCallback,
    ) -> RawExecutionResult:
        try:
            target = self._target(node)
            self._validate_credentials_policy(target, prepared)
            shares = self._shares(target, paths)
            self._validate_execution_host(target, paths, shares)
            nss_env = self._prepare_runtime(target, prepared, paths)
            self._cleanup_state_dir(paths)
            state_dir = self._state_dir(paths)
            state_dir.mkdir(mode=0o700)
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"Cloud Hypervisor launch validation failed: {exc}"
            await on_output("stderr", message)
            return RawExecutionResult(exit_code=1, stderr_lines=[message])

        guest_uid, guest_gid = self._effective_guest_ids(target)
        virtiofsd_processes: list[asyncio.subprocess.Process] = []
        vmm: asyncio.subprocess.Process | None = None
        writer: asyncio.StreamWriter | None = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + node.timeout_seconds
        try:
            for index, share in enumerate(shares):
                process = await self._start_virtiofsd(
                    target,
                    share,
                    guest_uid,
                    guest_gid,
                    paths.host_runtime_dir / f"virtiofsd-{index:03d}.log",
                )
                virtiofsd_processes.append(process)
            await self._wait_for_virtiofs_sockets(
                virtiofsd_processes,
                shares,
                min(
                    float(target.boot_timeout_seconds), max(0.1, deadline - loop.time())
                ),
                should_cancel,
            )
            vmm = await self._start_vmm(target, paths, shares)
            reader, writer = await self._connect_guest(
                target, paths, vmm, should_cancel, deadline
            )
            request = self._guest_request(target, prepared, shares, nss_env)
            return await self._consume_guest(
                reader, writer, request, on_output, should_cancel, deadline
            )
        except asyncio.CancelledError:
            return RawExecutionResult(exit_code=130, cancelled=True)
        except TimeoutError as exc:
            message = str(exc)
            await on_output("stderr", message)
            return RawExecutionResult(
                exit_code=124, stderr_lines=[message], timed_out=True
            )
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"Cloud Hypervisor launch failed: {exc}"
            await on_output("stderr", message)
            return RawExecutionResult(exit_code=1, stderr_lines=[message])
        finally:
            if writer is not None:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
            if vmm is not None:
                await self._request_vmm_shutdown(
                    self._socket_path(state_dir, "api.sock")
                )
                try:
                    await asyncio.wait_for(
                        vmm.wait(), timeout=target.shutdown_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    await self._stop_process(vmm, target.shutdown_timeout_seconds)
            if virtiofsd_processes:
                await asyncio.gather(
                    *(
                        self._stop_process(process, target.shutdown_timeout_seconds)
                        for process in reversed(virtiofsd_processes)
                    ),
                    return_exceptions=True,
                )
            try:
                with suppress(OSError, ValueError):
                    self._copy_console_log(paths)
            finally:
                self._cleanup_state_dir(paths)
