"""Local Docker execution with explicit mount and isolation controls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import re
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path, PurePosixPath

from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.runners.base import LaunchPlan
from agentflow.runners.local import LocalRunner
from agentflow.specs import DockerTarget, NodeSpec

_CONTAINER_DOCKER_SOCKET = "/var/run/docker.sock"
_DOCKER_ENV_RELATIVE_PATH = ".agentflow/docker-env.list"
_PORTABLE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DockerRunner(LocalRunner):
    """Run one prepared AgentFlow node in a local Docker container."""

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

        runtime_root = base_dir.resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_root.chmod(0o700)
        parent = runtime_root
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
        """Materialize Docker-only adapter files with credential-safe modes."""

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

    def _target(self, node: NodeSpec) -> DockerTarget:
        target = node.target
        if not isinstance(target, DockerTarget):
            raise TypeError("DockerRunner requires a DockerTarget")
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

    def _docker_daemon_socket(
        self, target: DockerTarget, paths: ExecutionPaths
    ) -> Path:
        if target.docker_daemon_socket is not None:
            return self._resolve_host_path(target.docker_daemon_socket, paths)

        docker_host = self._docker_cli_host()
        if docker_host:
            if not docker_host.startswith("unix://"):
                raise ValueError(
                    "`target.mount_docker_daemon` can mount only a local Unix Docker endpoint; "
                    f"the active endpoint is `{docker_host}`. Use `target.docker_daemon_socket` with an "
                    "absolute Unix-socket path, or disable the daemon mount."
                )
            socket_path = docker_host.removeprefix("unix://")
            if socket_path:
                return self._resolve_host_path(socket_path, paths)
        return Path(_CONTAINER_DOCKER_SOCKET)

    def _docker_cli_host(self) -> str | None:
        return (
            os.environ.get("DOCKER_HOST", "").strip()
            or self._active_docker_context_host()
        )

    def _validate_local_docker_endpoint(self) -> None:
        docker_host = self._docker_cli_host()
        if docker_host and not docker_host.startswith("unix://"):
            raise ValueError(
                "Docker targets use local host bind mounts and require a local Unix Docker endpoint; "
                f"the active endpoint is `{docker_host}`"
            )

    def _active_docker_context_host(self) -> str | None:
        """Resolve a named Docker context without invoking the Docker CLI."""

        docker_config = Path(os.environ.get("DOCKER_CONFIG", "~/.docker")).expanduser()
        context_name = os.environ.get("DOCKER_CONTEXT", "").strip()
        if not context_name:
            try:
                config = json.loads(
                    (docker_config / "config.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                config = {}
            configured_context = (
                config.get("currentContext") if isinstance(config, dict) else None
            )
            if isinstance(configured_context, str):
                context_name = configured_context.strip()

        if not context_name or context_name == "default":
            return None

        metadata_root = docker_config / "contexts" / "meta"
        context_hash = hashlib.sha256(context_name.encode("utf-8")).hexdigest()
        candidates = [metadata_root / context_hash / "meta.json"]
        try:
            candidates.extend(metadata_root.glob("*/meta.json"))
        except OSError:
            pass

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(metadata, dict) or metadata.get("Name") != context_name:
                continue
            endpoints = metadata.get("Endpoints")
            docker_endpoint = (
                endpoints.get("docker") if isinstance(endpoints, dict) else None
            )
            host = (
                docker_endpoint.get("Host")
                if isinstance(docker_endpoint, dict)
                else None
            )
            if isinstance(host, str) and host.strip():
                return host.strip()
            break

        raise ValueError(
            f"cannot resolve the active Docker context `{context_name}` to a local Unix endpoint; "
            "set `target.docker_daemon_socket` explicitly"
        )

    def _validate_docker_daemon_socket(
        self, target: DockerTarget, paths: ExecutionPaths
    ) -> None:
        socket_path = self._docker_daemon_socket(target, paths)
        if not socket_path.exists():
            raise FileNotFoundError(
                f"Docker daemon socket does not exist: {socket_path}"
            )
        if not socket_path.is_socket():
            raise ValueError(f"Docker daemon path is not a Unix socket: {socket_path}")

    def _bind_mount(
        self,
        *,
        source: str | Path,
        target: str,
        read_only: bool,
        paths: ExecutionPaths,
        managed: bool = False,
    ) -> dict[str, object]:
        source_path = self._resolve_host_path(source, paths)
        source_text = str(source_path)
        if "," in source_text:
            label = "AgentFlow-managed mount" if managed else "Docker mount source"
            raise ValueError(f"{label} must not contain commas: {source_text}")
        return {
            "type": "bind",
            "source": source_text,
            "target": target,
            "read_only": read_only,
            "managed": managed,
        }

    def _mounts(
        self,
        target: DockerTarget,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> list[dict[str, object]]:
        if prepared.runtime_symlinks and not target.inherit_credentials:
            raise ValueError(
                "Docker targets do not expose adapter-requested host files by default; set "
                "`target.inherit_credentials: true` to allow read-only runtime credential/config mounts"
            )
        mounts = [
            self._bind_mount(
                source=paths.host_workdir,
                target=target.workdir_mount,
                read_only=target.workdir_read_only,
                paths=paths,
                managed=True,
            ),
            self._bind_mount(
                source=paths.host_runtime_dir,
                target=target.runtime_mount,
                read_only=False,
                paths=paths,
                managed=True,
            ),
        ]
        if target.app_mount is not None:
            mounts.append(
                self._bind_mount(
                    source=paths.app_root,
                    target=target.app_mount,
                    read_only=True,
                    paths=paths,
                    managed=True,
                )
            )

        for mount in target.mounts:
            source_path = self._resolve_host_path(mount.source, paths)
            if not target.mount_docker_daemon:
                active_socket = self._docker_daemon_socket(target, paths)
                if source_path.is_socket() or self._host_paths_overlap(
                    source_path, active_socket
                ):
                    raise ValueError(
                        "an explicit Docker mount exposes the active daemon socket or one of its parent "
                        "directories; use `target.mount_docker_daemon: true` for this authority"
                    )
            if (
                not mount.read_only
                and target.workdir_read_only
                and self._host_paths_overlap(
                    source_path,
                    paths.host_workdir,
                )
            ):
                raise ValueError(
                    "a read-write `target.mounts` source overlaps the read-only managed workspace: "
                    f"{source_path}"
                )
            if (
                not mount.read_only
                and target.app_mount is not None
                and self._host_paths_overlap(source_path, paths.app_root)
            ):
                raise ValueError(
                    "a read-write `target.mounts` source overlaps the read-only managed AgentFlow app: "
                    f"{source_path}"
                )
            mounts.append(
                self._bind_mount(
                    source=source_path,
                    target=mount.target,
                    read_only=mount.read_only,
                    paths=paths,
                )
            )

        # Adapters use runtime symlinks for host auth/config files. A symlink
        # inside the runtime bind mount points at an inaccessible host path, so
        # expose each source as a nested read-only bind mount instead.
        for relative_path, source in prepared.runtime_symlinks.items():
            relative = PurePosixPath(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"runtime bind mount path must stay below the runtime directory: {relative_path}"
                )
            container_path = str(PurePosixPath(target.runtime_mount) / relative)
            mounts.append(
                self._bind_mount(
                    source=source,
                    target=container_path,
                    read_only=True,
                    paths=paths,
                    managed=True,
                )
            )

        if target.mount_docker_daemon:
            mounts.append(
                self._bind_mount(
                    source=self._docker_daemon_socket(target, paths),
                    target=_CONTAINER_DOCKER_SOCKET,
                    read_only=False,
                    paths=paths,
                    managed=True,
                )
            )

        normalized_targets = [
            posixpath.normpath(str(mount["target"])) for mount in mounts
        ]
        duplicate_targets = sorted(
            mount_target
            for mount_target in set(normalized_targets)
            if normalized_targets.count(mount_target) > 1
        )
        if duplicate_targets:
            raise ValueError(
                f"Docker mounts resolve to duplicate container targets: {duplicate_targets}"
            )
        return mounts

    def _materialize_runtime_bind_mountpoints(
        self,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> None:
        for relative_path, source in prepared.runtime_symlinks.items():
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"runtime bind mount path must stay below the runtime directory: {relative_path}"
                )
            destination = self._prepare_private_runtime_target(
                paths.host_runtime_dir, relative_path
            )
            source_path = self._resolve_host_path(source, paths)
            if source_path.is_dir():
                if destination.exists() and not destination.is_dir():
                    raise ValueError(
                        f"runtime bind mount target must be a directory: {destination}"
                    )
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o700)
                continue
            if destination.exists() and destination.is_dir():
                raise ValueError(
                    f"runtime bind mount target must be a file: {destination}"
                )
            destination.touch(exist_ok=True)
            destination.chmod(0o600)

    def _container_env(
        self,
        target: DockerTarget,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> dict[str, str]:
        env = dict(prepared.env)
        if target.app_mount is not None:
            inherited_pythonpath = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = (
                f"{target.app_mount}:{inherited_pythonpath}"
                if inherited_pythonpath
                else target.app_mount
            )
        if target.mount_docker_daemon:
            env["DOCKER_HOST"] = f"unix://{_CONTAINER_DOCKER_SOCKET}"
            # A sibling container's bind sources are interpreted by the host
            # daemon, not relative to this agent container. Make the exact host
            # paths available so nested Docker commands can mount intentionally.
            env["AGENTFLOW_HOST_WORKDIR"] = str(paths.host_workdir.resolve())
            env["AGENTFLOW_HOST_RUNTIME_DIR"] = str(paths.host_runtime_dir.resolve())
            env["AGENTFLOW_DOCKER_CONTAINER_NAME"] = self._container_name(paths)
        if target.dind:
            env["AGENTFLOW_DIND"] = "1"
            env["DOCKER_HOST"] = f"unix://{_CONTAINER_DOCKER_SOCKET}"
            env["DOCKER_TLS_CERTDIR"] = ""
            entrypoint_user = self._entrypoint_run_user(target)
            if entrypoint_user is not None and not self._is_root_user(entrypoint_user):
                run_uid, run_gid = entrypoint_user.split(":", 1)
                env["AGENTFLOW_RUN_UID"] = run_uid
                env["AGENTFLOW_RUN_GID"] = run_gid
        runtime_user = self._entrypoint_run_user(target)
        if (
            target.user == "host"
            and runtime_user is not None
            and not self._is_root_user(runtime_user)
        ):
            env.setdefault("HOME", f"{target.runtime_mount.rstrip('/')}/home")
        return env

    def _effective_user(self, target: DockerTarget) -> str | None:
        if target.dind:
            return None
        if target.user != "host":
            return target.user
        return self._host_user()

    def _host_user(self) -> str | None:
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            return f"{os.getuid()}:{os.getgid()}"
        return None

    def _entrypoint_run_user(self, target: DockerTarget) -> str | None:
        if target.dind and target.user == "host":
            return self._host_user()
        return self._effective_user(target)

    def _is_root_user(self, user: str) -> bool:
        return user in {"0", "0:0", "root", "root:root"} or user.startswith("0:")

    def _container_name(self, paths: ExecutionPaths) -> str:
        identity = str(paths.host_runtime_dir.resolve()).encode("utf-8")
        return f"agentflow-{hashlib.sha256(identity).hexdigest()[:20]}"

    def _socket_group(
        self, target: DockerTarget, paths: ExecutionPaths, effective_user: str | None
    ) -> int | None:
        if (
            not target.mount_docker_daemon
            or effective_user is None
            or self._is_root_user(effective_user)
        ):
            return None
        with suppress(OSError):
            return self._docker_daemon_socket(target, paths).stat().st_gid
        return None

    def _append_mount(self, command: list[str], mount: dict[str, object]) -> None:
        spec = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount["read_only"]:
            spec += ",readonly"
        command.extend(["--mount", spec])

    def _docker_env_file(self, container_env: dict[str, str]) -> str:
        lines: list[str] = []
        for key, value in container_env.items():
            if not _PORTABLE_ENV_NAME.fullmatch(key):
                raise ValueError(
                    f"Docker environment variable name must match `[A-Za-z_][A-Za-z0-9_]*`: {key!r}"
                )
            if any(character in value for character in ("\x00", "\r", "\n")):
                raise ValueError(
                    f"Docker environment variable `{key}` must be a single-line value; "
                    "mount multiline credentials/configuration as a file instead"
                )
            lines.append(f"{key}={value}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _docker_prepared(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> tuple[PreparedExecution, list[dict[str, object]], dict[str, str]]:
        target = self._target(node)
        self._validate_local_docker_endpoint()
        if (
            _DOCKER_ENV_RELATIVE_PATH in prepared.runtime_files
            or _DOCKER_ENV_RELATIVE_PATH in prepared.runtime_symlinks
        ):
            raise ValueError(
                f"adapter runtime path `{_DOCKER_ENV_RELATIVE_PATH}` is reserved by the Docker runner"
            )
        mounts = self._mounts(target, prepared, paths)
        container_env = self._container_env(target, prepared, paths)
        policy = target.network_policy
        effective_user = self._effective_user(target)
        container_name = self._container_name(paths)

        # Keep the stopped container inspectable until ``execute``'s finalizer.
        # The daemon state is an authoritative fallback when the foreground
        # Docker client remains attached after the container has exited.
        command = [target.engine, "run", "--name", container_name]
        if prepared.stdin is not None:
            command.append("--interactive")
        if target.privileged:
            command.append("--privileged")
        if effective_user is not None:
            command.extend(["--user", effective_user])
        socket_group = self._socket_group(target, paths, effective_user)
        if socket_group is not None:
            command.extend(["--group-add", str(socket_group)])
        command.extend(["--network", policy.docker_network])
        for alias in policy.aliases:
            command.extend(["--network-alias", alias])
        for dns_server in policy.dns:
            command.extend(["--dns", dns_server])
        for host, address in policy.add_hosts.items():
            command.extend(["--add-host", f"{host}={address}"])
        for mount in mounts:
            self._append_mount(command, mount)
        command.extend(["--workdir", prepared.cwd])
        env_file_path = paths.host_runtime_dir / _DOCKER_ENV_RELATIVE_PATH
        if container_env:
            command.extend(["--env-file", str(env_file_path)])
        command.extend(target.extra_args)
        if target.entrypoint is not None:
            command.extend(["--entrypoint", target.entrypoint])
        command.append(target.image)
        command.extend(prepared.command)

        return (
            PreparedExecution(
                command=command,
                env={},
                cwd=str(paths.host_workdir),
                trace_kind=prepared.trace_kind,
                runtime_files=(
                    {_DOCKER_ENV_RELATIVE_PATH: self._docker_env_file(container_env)}
                    if container_env
                    else {}
                ),
                stdin=prepared.stdin,
            ),
            mounts,
            container_env,
        )

    def plan_execution(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> LaunchPlan:
        target = self._target(node)
        docker_prepared, mounts, container_env = self._docker_prepared(
            node, prepared, paths
        )
        return LaunchPlan(
            kind="docker",
            command=list(docker_prepared.command),
            env=dict(docker_prepared.env),
            cwd=docker_prepared.cwd,
            stdin=docker_prepared.stdin,
            runtime_files=sorted(
                set(prepared.runtime_files)
                | set(prepared.runtime_symlinks)
                | ({_DOCKER_ENV_RELATIVE_PATH} if container_env else set())
            ),
            payload={
                "image": target.image,
                "engine": target.engine,
                "workdir": prepared.cwd,
                "env_keys": sorted(container_env),
                "mounts": [dict(mount) for mount in mounts],
                "network_policy": target.network_policy.model_dump(mode="json"),
                "privileged": target.privileged,
                "user": target.user,
                "effective_user": self._entrypoint_run_user(target) or "image-default",
                "docker_launch_user": self._effective_user(target) or "image-default",
                "container_name": self._container_name(paths),
                "mount_docker_daemon": target.mount_docker_daemon,
                "inherit_credentials": target.inherit_credentials,
                "dind": target.dind,
            },
        )

    async def _force_remove_container(self, engine: str, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                engine,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1)

    async def _inspect_container_state(
        self, engine: str, container_name: str
    ) -> dict[str, object] | None:
        try:
            process = await asyncio.create_subprocess_exec(
                engine,
                "inspect",
                "--format",
                "{{json .State}}",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1)
            raise
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1)
            return None
        if process.returncode != 0:
            return None
        try:
            state = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return state if isinstance(state, dict) else None

    async def _wait_for_container_exit(
        self, engine: str, container_name: str
    ) -> int:
        while True:
            state = await self._inspect_container_state(engine, container_name)
            if state is not None and state.get("Status") in {"exited", "dead"}:
                exit_code = state.get("ExitCode")
                if (
                    isinstance(exit_code, int)
                    and not isinstance(exit_code, bool)
                    and 0 <= exit_code <= 255
                ):
                    return exit_code
                return 125
            await asyncio.sleep(0.25)

    def _external_completion(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> Awaitable[int]:
        target = self._target(node)
        return self._wait_for_container_exit(
            target.engine, self._container_name(paths)
        )

    async def execute(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
        on_output,
        should_cancel,
    ):
        target = self._target(node)
        container_name = self._container_name(paths)
        if target.mount_docker_daemon:
            self._validate_docker_daemon_socket(target, paths)
        await self._force_remove_container(target.engine, container_name)
        docker_prepared, _, _ = self._docker_prepared(node, prepared, paths)
        paths.host_runtime_dir.mkdir(parents=True, exist_ok=True)
        paths.host_runtime_dir.chmod(0o700)
        self.materialize_runtime_files(paths.host_runtime_dir, prepared.runtime_files)
        self._materialize_runtime_bind_mountpoints(prepared, paths)
        runtime_user = self._entrypoint_run_user(target)
        if (
            target.user == "host"
            and runtime_user is not None
            and not self._is_root_user(runtime_user)
        ):
            runtime_home = paths.host_runtime_dir / "home"
            runtime_home.mkdir(parents=True, exist_ok=True)
            runtime_home.chmod(0o700)
        try:
            return await super().execute(
                node, docker_prepared, paths, on_output, should_cancel
            )
        finally:
            await self._force_remove_container(target.engine, container_name)
            with suppress(FileNotFoundError):
                (paths.host_runtime_dir / _DOCKER_ENV_RELATIVE_PATH).unlink()
