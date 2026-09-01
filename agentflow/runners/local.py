from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path

from agentflow.local_shell import render_shell_init, shell_wrapper_requires_command_placeholder, target_uses_interactive_bash
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.runners.base import LaunchPlan, RawExecutionResult, Runner, StreamCallback
from agentflow.specs import LocalTarget, NodeSpec
from agentflow.utils import ensure_dir


class LocalRunner(Runner):
    _KNOWN_SHELL_EXECUTABLES = {
        "ash",
        "bash",
        "dash",
        "fish",
        "ksh",
        "mksh",
        "pwsh",
        "sh",
        "zsh",
    }
    _SHELL_BUILTIN_PREFIX_TOKENS = {"exec"}
    _INTERACTIVE_SHELL_STDERR_NOISE = (
        "bash: cannot set terminal process group (",
        "bash: initialize_job_control: no job control in background:",
        "bash: no job control in this shell",
    )
    _TERMINATE_GRACE_SECONDS = 1.0
    _EXTERNAL_COMPLETION_GRACE_SECONDS = 1.0
    _SHELL_COMMAND_PLACEHOLDER_MESSAGE = (
        "`target.shell` already includes a shell command payload. Add `{command}` where AgentFlow should inject "
        "the prepared agent command."
    )

    def _shell_executable_index(self, shell_parts: list[str]) -> int | None:
        for index, part in enumerate(shell_parts):
            if os.path.basename(part) in self._KNOWN_SHELL_EXECUTABLES:
                return index
        if not shell_parts:
            return None
        return 0

    def _looks_like_env_assignment(self, token: str) -> bool:
        if "=" not in token or token.startswith("="):
            return False
        name, _ = token.split("=", 1)
        if not name:
            return False
        return name.replace("_", "a").isalnum() and not name[0].isdigit()

    def _env_wrapper_shell_index(self, command: list[str]) -> int | None:
        if not command or os.path.basename(command[0]) != "env":
            return None

        position = 1
        ignore_environment = False
        while position < len(command):
            token = command[position]
            if token == "--":
                position += 1
                break
            if token in {"-i", "--ignore-environment"}:
                ignore_environment = True
                position += 1
                continue
            if token == "-u":
                position += 2
                continue
            if token.startswith("--unset=") or (token.startswith("-u") and len(token) > 2):
                position += 1
                continue
            if token.startswith("-"):
                position += 1
                continue
            if self._looks_like_env_assignment(token):
                position += 1
                continue
            break

        if not ignore_environment or position >= len(command):
            return None
        return position

    def _env_wrapper_reserved_names(self, command: list[str], shell_index: int) -> set[str]:
        reserved: set[str] = set()
        position = 1
        while position < shell_index:
            token = command[position]
            if token == "--":
                break
            if token == "-u" and position + 1 < shell_index:
                reserved.add(command[position + 1])
                position += 2
                continue
            if token.startswith("--unset="):
                reserved.add(token.split("=", 1)[1])
                position += 1
                continue
            if token.startswith("-u") and len(token) > 2:
                reserved.add(token[2:])
                position += 1
                continue
            if self._looks_like_env_assignment(token):
                reserved.add(token.split("=", 1)[0])
            position += 1
        return reserved

    def _inline_env_wrapper_assignments(self, command: list[str], env: dict[str, str]) -> list[str]:
        shell_index = self._env_wrapper_shell_index(command)
        if shell_index is None or not env:
            return command

        reserved_names = self._env_wrapper_reserved_names(command, shell_index)
        assignments = [f"{key}={value}" for key, value in env.items() if key not in reserved_names]
        if not assignments:
            return command
        return [*command[:shell_index], *assignments, *command[shell_index:]]

    def _has_flag(self, shell_parts: list[str], short_flag: str, long_flag: str | None = None) -> bool:
        shell_index = self._shell_executable_index(shell_parts)
        if shell_index is None:
            return False
        return any(
            part == long_flag or (part.startswith("-") and not part.startswith("--") and short_flag in part[1:])
            for part in shell_parts[shell_index + 1 :]
        )

    def _command_flag_index(self, shell_parts: list[str]) -> int | None:
        shell_index = self._shell_executable_index(shell_parts)
        if shell_index is None:
            return None
        for index, part in enumerate(shell_parts[shell_index + 1 :], start=shell_index + 1):
            if part == "--command" or (part.startswith("-") and not part.startswith("--") and "c" in part[1:]):
                return index
        return None

    def _apply_shell_options(self, shell_parts: list[str], target: LocalTarget) -> list[str]:
        updated = list(shell_parts)
        command_index = self._command_flag_index(updated)
        insert_at = command_index if command_index is not None else len(updated)
        if target.shell_login and not self._has_flag(updated, "l", "--login"):
            updated.insert(insert_at, "-l")
            insert_at += 1
        if target.shell_interactive and not self._has_flag(updated, "i"):
            updated.insert(insert_at, "-i")
        return updated

    def _replace_shell_template_command(self, shell_parts: list[str], placeholder: str, shell_command: str) -> list[str]:
        return [part.replace(placeholder, shell_command) for part in shell_parts]

    def _normalize_shell_command(self, shell_parts: list[str]) -> list[str]:
        normalized = list(shell_parts)
        while normalized and normalized[0] in self._SHELL_BUILTIN_PREFIX_TOKENS:
            normalized.pop(0)
        return normalized

    def _augment_local_env(self, prepared: PreparedExecution, paths: ExecutionPaths) -> dict[str, str]:
        return dict(prepared.env)

    def _command_for_target(self, node: NodeSpec, prepared: PreparedExecution) -> tuple[list[str], dict[str, str]]:
        target = node.target
        if not isinstance(target, LocalTarget) or not target.shell:
            return prepared.command, {}
        if shell_wrapper_requires_command_placeholder(target.shell):
            raise ValueError(self._SHELL_COMMAND_PLACEHOLDER_MESSAGE)

        command_text = shlex.join(prepared.command)
        shell_command = 'eval "$AGENTFLOW_TARGET_COMMAND"'
        shell_init = render_shell_init(target.shell_init)
        if shell_init:
            shell_command = f"{shell_init} && {shell_command}"

        if "{command}" in target.shell:
            placeholder = "__AGENTFLOW_COMMAND_PLACEHOLDER__"
            shell_parts = self._normalize_shell_command(shlex.split(target.shell.replace("{command}", placeholder)))
            if not shell_parts:
                return prepared.command, {}
            shell_parts = self._apply_shell_options(shell_parts, target)
            command_index = self._command_flag_index(shell_parts)
            if command_index is None:
                placeholder_index = next(
                    (index for index, part in enumerate(shell_parts) if placeholder in part),
                    None,
                )
                if placeholder_index is not None:
                    shell_parts.insert(placeholder_index, "-c")
            shell_parts = self._replace_shell_template_command(shell_parts, placeholder, shell_command)
            return shell_parts, {"AGENTFLOW_TARGET_COMMAND": command_text}

        shell_parts = self._normalize_shell_command(shlex.split(target.shell))
        shell_parts = self._apply_shell_options(shell_parts, target)
        if not shell_parts:
            return prepared.command, {}

        command_index = self._command_flag_index(shell_parts)
        if command_index is None:
            shell_parts.append("-c")

        if shell_init:
            shell_parts.append(shell_command)
            return shell_parts, {"AGENTFLOW_TARGET_COMMAND": command_text}

        return [*shell_parts, command_text], {}

    def plan_execution(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> LaunchPlan:
        command, target_env = self._command_for_target(node, prepared)
        plan_env = self._augment_local_env(prepared, paths)
        plan_env.update(target_env)
        command = self._inline_env_wrapper_assignments(command, plan_env)
        return LaunchPlan(
            command=command,
            env=plan_env,
            cwd=prepared.cwd,
            stdin=prepared.stdin,
            runtime_files=sorted(prepared.runtime_files),
        )

    def _should_suppress_stderr(self, node: NodeSpec, text: str) -> bool:
        if not target_uses_interactive_bash(node.target):
            return False
        return any(text.startswith(prefix) for prefix in self._INTERACTIVE_SHELL_STDERR_NOISE)

    async def _wait_for_exit(self, wait_task: asyncio.Task[int], timeout: float) -> bool:
        if wait_task.done():
            return True
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def _terminate_with_fallback(self, process, wait_task: asyncio.Task[int]) -> None:
        with suppress(ProcessLookupError):
            process.terminate()
        if not await self._wait_for_exit(wait_task, self._TERMINATE_GRACE_SECONDS):
            with suppress(ProcessLookupError):
                process.kill()
            await self._wait_for_exit(wait_task, self._TERMINATE_GRACE_SECONDS)

        # asyncio exposes no public Process.close(). If descendants inherited a
        # pipe, the transport otherwise survives after the direct child exits,
        # delaying stream drain and warning when the event loop is later closed.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
            await asyncio.sleep(0)

    async def _consume_stream(self, node: NodeSpec, stream, stream_name: str, buffer: list[str], on_output: StreamCallback) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if stream_name == "stderr" and self._should_suppress_stderr(node, text):
                continue
            buffer.append(text)
            await on_output(stream_name, text)

    def _external_completion(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
    ) -> Awaitable[int] | None:
        """Return an optional authoritative completion signal for the process.

        Most local commands are complete when their subprocess exits. Runners
        backed by another runtime can override this hook when that runtime has a
        more authoritative lifecycle signal than its foreground client.
        """

        return None

    async def execute(
        self,
        node: NodeSpec,
        prepared: PreparedExecution,
        paths: ExecutionPaths,
        on_output: StreamCallback,
        should_cancel,
    ) -> RawExecutionResult:
        self.materialize_runtime_files(paths.host_runtime_dir, prepared.runtime_files)
        self.materialize_runtime_symlinks(paths.host_runtime_dir, prepared.runtime_symlinks)
        ensure_dir(Path(prepared.cwd))
        launch_env = self._augment_local_env(prepared, paths)
        command, target_env = self._command_for_target(node, prepared)
        launch_env.update(target_env)
        env = os.environ.copy()
        env.update(launch_env)
        command = self._inline_env_wrapper_assignments(command, launch_env)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=prepared.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if prepared.stdin is not None else asyncio.subprocess.DEVNULL,
        )
        if prepared.stdin is not None and process.stdin is not None:
            process.stdin.write(prepared.stdin.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
        elif process.stdin is not None:
            process.stdin.close()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_task = asyncio.create_task(self._consume_stream(node, process.stdout, "stdout", stdout_lines, on_output))
        stderr_task = asyncio.create_task(self._consume_stream(node, process.stderr, "stderr", stderr_lines, on_output))
        wait_task = asyncio.create_task(process.wait())
        external_completion = self._external_completion(node, prepared, paths)
        external_task = (
            asyncio.ensure_future(external_completion)
            if external_completion is not None
            else None
        )
        external_exit_code: int | None = None
        timed_out = False
        cancelled = False

        timeout = node.timeout_seconds if node.timeout_seconds and node.timeout_seconds > 0 else None
        deadline = asyncio.get_running_loop().time() + timeout if timeout else None

        # Monitor process exit, streams, timeout, and cancellation concurrently.
        # Key insight: claude spawns child processes (MCP servers, plugins) that
        # inherit stdout/stderr pipes. When claude exits, those children keep the
        # pipes open — so we CANNOT rely on stream EOF to detect completion.
        # Instead, we treat process exit (wait_task) as the primary signal.
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time() if deadline else None
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                if should_cancel():
                    cancelled = True
                    break
                check_timeout = min(remaining or 1.0, 1.0)
                monitored_tasks = {stdout_task, stderr_task, wait_task}
                if external_task is not None:
                    monitored_tasks.add(external_task)
                done, _ = await asyncio.wait(
                    monitored_tasks,
                    timeout=check_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_task in done:
                    # Process exited — this is our primary completion signal.
                    # Don't wait for streams; child processes may hold pipes open.
                    break
                if external_task is not None and external_task in done:
                    external_exit_code = external_task.result()
                    break
                if stdout_task in done and stderr_task in done:
                    # Both streams EOF'd — process should follow shortly
                    if not wait_task.done():
                        completion_tasks = {wait_task}
                        if external_task is not None:
                            completion_tasks.add(external_task)
                        completed, _ = await asyncio.wait(
                            completion_tasks,
                            timeout=5,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if external_task is not None and external_task in completed:
                            external_exit_code = external_task.result()
                        elif wait_task not in completed:
                            timed_out = True
                    break
        except Exception:
            timed_out = True

        # Drain streams with a hard 3s timeout — child processes may hold pipes
        async def _drain_streams():
            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
                    timeout=3,
                )
            except asyncio.TimeoutError:
                # Cancel stuck stream tasks
                for task in (stdout_task, stderr_task):
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task

        if timed_out:
            await self._terminate_with_fallback(process, wait_task)
            await _drain_streams()
            stderr_lines.append(f"Timed out after {node.timeout_seconds}s")
            await on_output("stderr", stderr_lines[-1])
        elif cancelled:
            await self._terminate_with_fallback(process, wait_task)
            await _drain_streams()
            stderr_lines.append("Cancelled by user")
            await on_output("stderr", stderr_lines[-1])
        elif external_exit_code is not None:
            if not await self._wait_for_exit(
                wait_task, self._EXTERNAL_COMPLETION_GRACE_SECONDS
            ):
                await self._terminate_with_fallback(process, wait_task)
            await _drain_streams()
        else:
            await _drain_streams()
            if not wait_task.done():
                await wait_task

        if external_task is not None:
            if not external_task.done():
                external_task.cancel()
                with suppress(asyncio.CancelledError):
                    await external_task
            elif external_exit_code is None:
                with suppress(asyncio.CancelledError, Exception):
                    external_task.result()

        if timed_out:
            exit_code = 124
        elif cancelled:
            exit_code = 130
        elif external_exit_code is not None:
            exit_code = external_exit_code
        else:
            exit_code = process.returncode if process.returncode is not None else 0
        return RawExecutionResult(
            exit_code=exit_code,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            timed_out=timed_out,
            cancelled=cancelled,
        )
