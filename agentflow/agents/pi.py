from __future__ import annotations

import json
from pathlib import Path

from agentflow.agents.base import AgentAdapter
from agentflow.env import merge_env_layers
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.specs import NodeSpec, ProviderConfig, RepoInstructionsMode, ToolAccess


_PI_READ_ONLY_TOOLS = "read,grep,find,ls"
_PI_READ_WRITE_TOOLS = "read,bash,edit,write,grep,find,ls"
_MINIMAX_MODELS = [
    {
        "id": "MiniMax-M3",
        "name": "MiniMax M3",
        "reasoning": True,
        # Pi's models.json schema currently supports text and image inputs, not video.
        "input": ["text", "image"],
        # Pi requires numeric rates; MiniMax does not publish a cache-write rate for M3.
        "cost": {"input": 0.6, "output": 2.4, "cacheRead": 0.12, "cacheWrite": 0.0},
        "contextWindow": 1_000_000,
    },
    {
        "id": "MiniMax-M2.7",
        "name": "MiniMax M2.7",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.06, "cacheWrite": 0.375},
        "contextWindow": 204_800,
    },
]


class PiAdapter(AgentAdapter):
    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        if node.mcps:
            raise ValueError(
                "pi adapter does not support `mcps`. Pi uses extensions, not MCP servers; "
                "pass `--extension <path>` via `extra_args` instead."
            )

        provider = self.provider_config(node.provider, node.agent)
        executable = node.executable or "pi"
        env = merge_env_layers(getattr(provider, "env", None), node.env)
        repo_instructions_ignored = node.repo_instructions_mode == RepoInstructionsMode.IGNORE

        command: list[str] = [
            executable,
            "--print",
            "--mode",
            "json",
            "--no-session",
        ]

        tools = _PI_READ_ONLY_TOOLS if node.tools == ToolAccess.READ_ONLY else _PI_READ_WRITE_TOOLS
        command.extend(["--tools", tools])

        runtime_files: dict[str, str] = {}
        scoped_home_needed = bool(provider and (provider.base_url or provider.headers))

        if scoped_home_needed:
            pi_home_relative = Path("pi-home") / "agent"
            models_rel = self.relative_runtime_file(str(pi_home_relative), "models.json")
            settings_rel = self.relative_runtime_file(str(pi_home_relative), "settings.json")
            runtime_files[models_rel] = self._render_models_json(provider, node.model)
            runtime_files[settings_rel] = "{}\n"
            env["PI_CODING_AGENT_DIR"] = str(Path(paths.target_runtime_dir) / pi_home_relative)
        elif provider and provider.name and "/" not in (node.model or ""):
            command.extend(["--provider", provider.name])

        if provider and provider.api_key_env and provider.api_key_env not in env:
            # Surface the key into the subprocess env so Pi can read it by name.
            import os

            resolved = os.getenv(provider.api_key_env)
            if resolved is not None:
                env.setdefault(provider.api_key_env, resolved)

        if node.model:
            command.extend(["--model", node.model])

        if repo_instructions_ignored:
            command.extend(["--no-skills", "--no-extensions", "--no-prompt-templates"])

        command.extend(node.extra_args)

        cwd = paths.target_workdir
        if repo_instructions_ignored:
            cwd = str(Path(paths.target_runtime_dir))

        # Pass the prompt via stdin so it is never parsed as a flag or `@file`
        # reference by Pi's positional-message argument handling.
        return PreparedExecution(
            command=command,
            env=env,
            cwd=cwd,
            trace_kind="pi",
            runtime_files=runtime_files,
            stdin=prompt,
        )

    def _render_models_json(self, provider: ProviderConfig, model: str | None) -> str:
        """Render a scoped ``models.json`` containing only the declared provider.

        Pi resolves custom providers from its agent directory's ``models.json``. When the
        caller supplies a full ``ProviderConfig`` (with ``base_url``), we materialize a
        minimal ``models.json`` under a scoped ``PI_CODING_AGENT_DIR`` so the run does
        not depend on the user's ``~/.pi/agent/models.json``.
        """
        entry: dict[str, object] = {
            "baseUrl": provider.base_url,
            "api": provider.wire_api or "openai-completions",
        }
        if provider.api_key_env:
            entry["apiKey"] = provider.api_key_env
        if provider.headers:
            entry["headers"] = dict(provider.headers)
            entry["authHeader"] = True
        model_id = self._extract_model_id(model, provider.name)
        models = [dict(model_entry) for model_entry in _MINIMAX_MODELS] if provider.name == "minimax" else []
        if model_id and not any(model_entry["id"] == model_id for model_entry in models):
            models.append({"id": model_id})
        entry["models"] = models

        payload = {"providers": {provider.name: entry}}
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @staticmethod
    def _extract_model_id(model: str | None, provider_name: str) -> str | None:
        if not model:
            return None
        ident = model
        if "/" in ident:
            prefix, _, rest = ident.partition("/")
            if prefix == provider_name:
                ident = rest
        if ":" in ident:
            ident = ident.split(":", 1)[0]
        return ident or None
