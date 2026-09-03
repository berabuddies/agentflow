from __future__ import annotations

import json
from pathlib import Path

from agentflow.agents.base import AgentAdapter
from agentflow.env import merge_env_layers
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.specs import NodeSpec, ProviderConfig, RepoInstructionsMode, ToolAccess


_CODEX_GOAL_BOOTSTRAP_SCRIPT = r'''
import json
import os
import select
import subprocess
import sys
import time


def fail(message):
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(1)


payload = json.load(sys.stdin)
executable = payload["executable"]
server = subprocess.Popen(
    [executable, "app-server", "--listen", "stdio://", "--enable", "goals"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ.copy(),
)
next_id = 0


def send(method, params):
    global next_id
    next_id += 1
    request = {"jsonrpc": "2.0", "id": next_id, "method": method, "params": params}
    assert server.stdin is not None
    server.stdin.write(json.dumps(request) + "\n")
    server.stdin.flush()
    return next_id


def read_response(request_id, timeout=30):
    assert server.stdout is not None
    assert server.stderr is not None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            fail(f"codex app-server exited before response {request_id}")
        ready, _, _ = select.select([server.stdout, server.stderr], [], [], 0.2)
        for stream in ready:
            line = stream.readline()
            if not line:
                continue
            if stream is server.stderr:
                print(line.rstrip("\n"), file=sys.stderr, flush=True)
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                fail(f"codex app-server error for {request_id}: {message['error']}")
            return message.get("result")
    fail(f"timed out waiting for codex app-server response {request_id}")


def stop_server():
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=3)


try:
    read_response(send("initialize", payload["initialize"]))
    thread_result = read_response(send("thread/start", payload["thread_start"]))
    thread_id = thread_result["thread"]["id"]
    read_response(send("thread/goal/set", {
        "threadId": thread_id,
        "objective": payload["objective"],
        "status": "active",
    }))
finally:
    stop_server()

resume_args = list(payload["resume_args"]) + [thread_id, payload["prompt"]]
os.execvp(resume_args[0], resume_args)
'''


class CodexAdapter(AgentAdapter):
    _SUPPORTED_SANDBOX_MODES = {"read-only", "workspace-write", "danger-full-access"}

    def _format_toml_value(self, value: object) -> str:
        import json

        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(self._format_toml_value(item) for item in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(f"{key} = {self._format_toml_value(inner)}" for key, inner in value.items())
            return "{" + items + "}"
        return json.dumps(str(value), ensure_ascii=False)

    def _render_config(self, node: NodeSpec, provider: ProviderConfig | None, sandbox_mode: str) -> str:
        lines: list[str] = []
        if node.model:
            lines.append(f"model = {self._format_toml_value(node.model)}")
        lines.append(f"approval_policy = {self._format_toml_value('never')}")
        lines.append(f"sandbox_mode = {self._format_toml_value(sandbox_mode)}")
        if provider and (provider.base_url or provider.api_key_env or provider.wire_api):
            lines.append("")
            lines.append(f"[model_providers.{provider.name}]")
            lines.append(f"name = {self._format_toml_value(provider.name)}")
            if provider.base_url:
                lines.append(f"base_url = {self._format_toml_value(provider.base_url)}")
            if provider.api_key_env:
                lines.append(f"env_key = {self._format_toml_value(provider.api_key_env)}")
            if provider.wire_api:
                lines.append(f"wire_api = {self._format_toml_value(provider.wire_api)}")
        if provider:
            lines.append("")
            lines.append("[profiles.agentflow]")
            if node.model:
                lines.append(f"model = {self._format_toml_value(node.model)}")
            lines.append(f"model_provider = {self._format_toml_value(provider.name)}")
        if node.mcps:
            for mcp in node.mcps:
                lines.append("")
                lines.append(f"[mcp_servers.{mcp.name}]")
                if mcp.transport == "stdio":
                    if mcp.command:
                        lines.append(f"command = {self._format_toml_value(mcp.command)}")
                    if mcp.args:
                        lines.append(f"args = {self._format_toml_value(mcp.args)}")
                    if mcp.env:
                        lines.append(f"env = {self._format_toml_value(mcp.env)}")
                else:
                    if mcp.url:
                        lines.append(f"url = {self._format_toml_value(mcp.url)}")
                    if mcp.headers:
                        lines.append(f"http_headers = {self._format_toml_value(mcp.headers)}")
        return "\n".join(lines) + "\n"

    def _resolve_sandbox_mode(self, node: NodeSpec, env: dict[str, str]) -> str:
        override = (env.pop("AGENTFLOW_CODEX_SANDBOX_MODE", "") or "").strip()
        if not override:
            return "read-only" if node.tools == ToolAccess.READ_ONLY else "workspace-write"
        if override not in self._SUPPORTED_SANDBOX_MODES:
            raise ValueError(
                "AGENTFLOW_CODEX_SANDBOX_MODE must be one of: "
                + ", ".join(sorted(self._SUPPORTED_SANDBOX_MODES))
            )
        return override

    _WRAPPER_FILENAME = "agentflow_wrapper.md"
    _WRAPPER_SEPARATOR = "\n\n---\n\n"

    def _maybe_prepend_wrapper(self, node: NodeSpec, prompt: str) -> str:
        """Prepend an agentflow-side wrapper to the user prompt if one exists.

        For tuned codex builds the executable lives at
        ``<version>/repo/codex-rs/target/debug/codex``; we look for
        ``<version>/repo/codex-rs/agentflow_wrapper.md`` next to it. This is
        the most reliable evolution surface because the wrapper text becomes
        part of the user message — gateways that override server-side system
        prompts cannot strip it.
        """
        executable = node.executable
        if not executable:
            return prompt
        exec_path = Path(executable).expanduser()
        if not exec_path.is_absolute():
            return prompt
        # codex_tuned binary path: .../codex-rs/target/debug/codex
        # → walk up three parents to reach codex-rs/
        if len(exec_path.parents) < 3:
            return prompt
        codex_rs_root = exec_path.parents[2]
        wrapper_path = codex_rs_root / self._WRAPPER_FILENAME
        if not wrapper_path.is_file():
            return prompt
        try:
            wrapper_text = wrapper_path.read_text(encoding="utf-8").strip()
        except OSError:
            return prompt
        if not wrapper_text:
            return prompt
        return wrapper_text + self._WRAPPER_SEPARATOR + prompt

    def _goal_payload(
        self,
        node: NodeSpec,
        prompt: str,
        *,
        executable: str,
        provider: ProviderConfig | None,
        repo_instructions_ignored: bool,
        sandbox: str,
        target_workdir: str,
        workspace_root: str | None,
    ) -> dict[str, object] | None:
        goal = getattr(node, "goal", False)
        if not goal:
            return None

        if isinstance(goal, str):
            objective = goal.strip()
        else:
            objective = prompt.strip()

        resume_args = [
            executable,
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-c",
            'approval_policy="never"',
            "-c",
            "suppress_unstable_features_warning=true",
            "--enable",
            "goals",
            "-c",
            f"sandbox_mode={self._format_toml_value(sandbox)}",
        ]
        if node.model:
            resume_args.extend(["--model", node.model])
        if provider:
            resume_args.extend(["-c", f"model_provider={self._format_toml_value(provider.name)}"])
        if repo_instructions_ignored:
            resume_args.extend(["--disable", "plugins"])
            if workspace_root:
                resume_args.extend(
                    [
                        "-c",
                        "sandbox_workspace_write.writable_roots="
                        + self._format_toml_value([workspace_root]),
                    ]
                )
        resume_args.extend(node.extra_args)
        features = {"goals": True}
        if repo_instructions_ignored:
            features["plugins"] = False
        config: dict[str, object] = {
            "features": features,
            "suppress_unstable_features_warning": True,
            "sandbox_mode": sandbox,
        }
        if provider:
            config["model_provider"] = provider.name
        if repo_instructions_ignored and workspace_root:
            config["sandbox_workspace_write"] = {"writable_roots": [workspace_root]}

        thread_start: dict[str, object] = {
            "cwd": target_workdir,
            "approvalPolicy": "never",
            "sandbox": sandbox,
            "config": config,
            "threadSource": "user",
            "sessionStartSource": "startup",
        }
        if node.model:
            thread_start["model"] = node.model
        if provider:
            thread_start["modelProvider"] = provider.name

        return {
            "executable": executable,
            "initialize": {
                "clientInfo": {"name": "agentflow", "version": "0"},
                "capabilities": None,
            },
            "thread_start": thread_start,
            "objective": objective,
            "prompt": prompt.strip() or objective,
            "resume_args": resume_args,
        }

    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        provider = self.provider_config(node.provider, node.agent)
        executable = node.executable or "codex"
        env = merge_env_layers(getattr(provider, "env", None), node.env)
        sandbox = self._resolve_sandbox_mode(node, env)
        repo_instructions_ignored = node.repo_instructions_mode == RepoInstructionsMode.IGNORE
        command = [
            executable,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-c",
            'approval_policy="never"',
            "-c",
            "suppress_unstable_features_warning=true",
            "--sandbox",
            sandbox,
        ]
        if node.goal:
            command.extend(["--enable", "goals"])
        if node.model and not provider:
            command.extend(["--model", node.model])
        if provider:
            command.extend(["--profile", "agentflow"])
        if repo_instructions_ignored:
            command.extend(["--disable", "plugins"])
            command.extend(["--add-dir", paths.target_workdir])
        command.extend(node.extra_args)
        prompt = self._maybe_prepend_wrapper(node, prompt)
        cwd = paths.target_workdir
        if repo_instructions_ignored:
            cwd = str(Path(paths.target_runtime_dir))
        goal_payload = self._goal_payload(
            node,
            prompt,
            executable=executable,
            provider=provider,
            repo_instructions_ignored=repo_instructions_ignored,
            sandbox=sandbox,
            target_workdir=cwd,
            workspace_root=paths.target_workdir if repo_instructions_ignored else None,
        )
        stdin = None
        if goal_payload is not None:
            command = ["python3", "-c", _CODEX_GOAL_BOOTSTRAP_SCRIPT]
            stdin = json.dumps(goal_payload)
        else:
            command.append(prompt)

        runtime_files: dict[str, str] = {}
        runtime_symlinks: dict[str, str] = {}
        is_isolated_target = getattr(node.target, "kind", None) in {"docker", "cloud_hypervisor"}
        inherit_host_credentials = not is_isolated_target or bool(
            getattr(node.target, "inherit_credentials", False)
        )
        needs_scoped_home = bool(
            provider
            or node.mcps
            or repo_instructions_ignored
            or (is_isolated_target and inherit_host_credentials)
        )
        if needs_scoped_home:
            codex_home = str(Path(paths.target_runtime_dir) / "codex_home")
            host_config = Path.home() / ".codex" / "config.toml"
            inherit_host_config = (
                inherit_host_credentials
                and provider is None
                and not node.mcps
                and host_config.is_file()
            )
            if inherit_host_config:
                runtime_symlinks[self.relative_runtime_file("codex_home", "config.toml")] = str(host_config)
            else:
                runtime_files[self.relative_runtime_file("codex_home", "config.toml")] = self._render_config(
                    node,
                    provider,
                    sandbox,
                )
            host_auth = Path.home() / ".codex" / "auth.json"
            if inherit_host_credentials and host_auth.is_file():
                runtime_symlinks[self.relative_runtime_file("codex_home", "auth.json")] = str(host_auth)
            env["CODEX_HOME"] = codex_home
            env["HOME"] = codex_home
        return PreparedExecution(
            command=command,
            env=env,
            cwd=cwd,
            trace_kind="codex",
            runtime_files=runtime_files,
            runtime_symlinks=runtime_symlinks,
            stdin=stdin,
        )
