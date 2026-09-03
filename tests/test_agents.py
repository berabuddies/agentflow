from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from agentflow.agents.claude import ClaudeAdapter
from agentflow.agents.codex import CodexAdapter
from agentflow.agents.kimi import KimiAdapter
from agentflow.agents.pi import PiAdapter
from agentflow.prepared import ExecutionPaths
from agentflow.specs import NodeSpec, RepoInstructionsMode

import pytest


def _paths(tmp_path: Path) -> ExecutionPaths:
    return ExecutionPaths(
        host_workdir=tmp_path,
        host_runtime_dir=tmp_path / ".runtime",
        target_workdir=str(tmp_path),
        target_runtime_dir=str(tmp_path / ".runtime"),
        app_root=tmp_path,
    )


def test_claude_adapter_uses_provider_api_key_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "test-secret")
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "provider": {
                "name": "kimi-proxy",
                "base_url": "https://example.test/anthropic",
                "api_key_env": "TEST_CLAUDE_API_KEY",
                "headers": {"x-provider": "kimi"},
            },
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["ANTHROPIC_BASE_URL"] == "https://example.test/anthropic"
    assert prepared.env["ANTHROPIC_API_KEY"] == "test-secret"
    assert json.loads(prepared.env["ANTHROPIC_CUSTOM_HEADERS"]) == {"x-provider": "kimi"}
    assert "ANTHROPIC_API_KEY_ENV" not in prepared.env


def test_codex_adapter_uses_current_exec_flags(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert prepared.command[:4] == ["codex", "exec", "--json", "--skip-git-repo-check"]
    assert "--ask-for-approval" not in prepared.command
    assert prepared.command[4:10] == [
        "-c",
        'approval_policy="never"',
        "-c",
        "suppress_unstable_features_warning=true",
        "--sandbox",
        "read-only",
    ]


def test_codex_adapter_suppresses_unstable_feature_warning(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert prepared.command.count("-c") == 2
    assert 'suppress_unstable_features_warning=true' in prepared.command


def test_codex_adapter_runs_prompt_through_native_goal_bootstrap(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "implement",
            "agent": "codex",
            "prompt": "Finish the migration and keep tests green.",
            "goal": True,
        }
    )

    prepared = CodexAdapter().prepare(node, "Finish the migration and keep tests green.", _paths(tmp_path))

    assert prepared.command[:2] == ["python3", "-c"]
    assert prepared.stdin is not None
    payload = json.loads(prepared.stdin)
    assert payload["objective"] == "Finish the migration and keep tests green."
    assert payload["prompt"] == "Finish the migration and keep tests green."
    assert payload["resume_args"][:5] == ["codex", "exec", "resume", "--json", "--skip-git-repo-check"]
    assert "--enable" in payload["resume_args"]
    assert payload["resume_args"][payload["resume_args"].index("--enable") + 1] == "goals"
    assert "-c" in payload["resume_args"]
    assert 'sandbox_mode="read-only"' in payload["resume_args"]


def test_codex_adapter_combines_explicit_goal_with_prompt_context(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "repair",
            "agent": "codex",
            "prompt": "Use pytest after each change.",
            "goal": "Repair the failing API tests.",
        }
    )

    prepared = CodexAdapter().prepare(node, "Use pytest after each change.", _paths(tmp_path))

    assert prepared.stdin is not None
    payload = json.loads(prepared.stdin)
    assert payload["objective"] == "Repair the failing API tests."
    assert payload["prompt"] == "Use pytest after each change."


def test_codex_goal_resume_preserves_custom_provider_and_model(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "repair",
            "agent": "codex",
            "prompt": "Use the custom provider.",
            "goal": True,
            "model": "gpt-custom",
            "provider": {
                "name": "openai-pinned",
                "base_url": "http://example.test/v1",
                "api_key_env": "OPENAI_API_KEY",
                "wire_api": "responses",
            },
        }
    )

    prepared = CodexAdapter().prepare(node, "Use the custom provider.", _paths(tmp_path))

    assert prepared.stdin is not None
    payload = json.loads(prepared.stdin)
    assert payload["thread_start"]["model"] == "gpt-custom"
    assert payload["thread_start"]["modelProvider"] == "openai-pinned"
    assert payload["thread_start"]["config"]["model_provider"] == "openai-pinned"
    assert "--model" in payload["resume_args"]
    assert payload["resume_args"][payload["resume_args"].index("--model") + 1] == "gpt-custom"
    assert 'model_provider="openai-pinned"' in payload["resume_args"]


def test_codex_goal_resume_preserves_ignored_instruction_workspace_root(tmp_path):
    paths = _paths(tmp_path)
    node = NodeSpec.model_validate(
        {
            "id": "repair",
            "agent": "codex",
            "prompt": "Work on the repo without loading repo instructions.",
            "goal": True,
            "repo_instructions_mode": RepoInstructionsMode.IGNORE,
            "tools": "read_write",
        }
    )

    prepared = CodexAdapter().prepare(node, "Work on the repo without loading repo instructions.", paths)

    assert prepared.cwd == str(tmp_path / ".runtime")
    assert prepared.stdin is not None
    payload = json.loads(prepared.stdin)
    expected_roots = [str(tmp_path)]
    assert payload["thread_start"]["cwd"] == str(tmp_path / ".runtime")
    assert payload["thread_start"]["config"]["features"]["plugins"] is False
    assert payload["thread_start"]["config"]["sandbox_workspace_write"]["writable_roots"] == expected_roots
    assert "--disable" in payload["resume_args"]
    assert payload["resume_args"][payload["resume_args"].index("--disable") + 1] == "plugins"
    assert "sandbox_workspace_write.writable_roots=" + json.dumps(expected_roots) in payload["resume_args"]


def test_codex_goal_bootstrap_sets_native_thread_goal_then_resumes(tmp_path):
    log_path = tmp_path / "codex-log.jsonl"
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import sys

log_path = os.environ["FAKE_CODEX_LOG"]

def log(payload):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

args = sys.argv[1:]
if args[:2] == ["app-server", "--listen"]:
    for line in sys.stdin:
        request = json.loads(line)
        log({"kind": "app-server", "method": request["method"], "params": request.get("params")})
        if request["method"] == "initialize":
            response = {"id": request["id"], "result": {"userAgent": "fake", "codexHome": "/tmp/fake", "platformFamily": "unix", "platformOs": "linux"}}
        elif request["method"] == "thread/start":
            response = {"id": request["id"], "result": {"thread": {"id": "thread-123"}}}
        elif request["method"] == "thread/goal/set":
            response = {"id": request["id"], "result": {"goal": {"threadId": "thread-123", "objective": request["params"]["objective"], "status": "active"}}}
        else:
            response = {"id": request["id"], "error": {"message": "unexpected"}}
        print(json.dumps(response), flush=True)
    raise SystemExit(0)

if args[:2] == ["exec", "resume"]:
    log({"kind": "resume", "args": args})
    prompt = args[-1]
    print(json.dumps({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": prompt}]}}))
    raise SystemExit(0)

raise SystemExit(2)
''',
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    node = NodeSpec.model_validate(
        {
            "id": "repair",
            "agent": "codex",
            "prompt": "Use pytest after each change.",
            "goal": "Repair the failing API tests.",
            "executable": str(fake_codex),
        }
    )
    prepared = CodexAdapter().prepare(node, "Use pytest after each change.", _paths(tmp_path))

    result = subprocess.run(
        prepared.command,
        input=prepared.stdin,
        text=True,
        capture_output=True,
        env={**os.environ, **prepared.env, "FAKE_CODEX_LOG": str(log_path)},
        cwd=prepared.cwd,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["method"] for entry in log_entries if entry["kind"] == "app-server"] == [
        "initialize",
        "thread/start",
        "thread/goal/set",
    ]
    goal_set = [entry for entry in log_entries if entry.get("method") == "thread/goal/set"][0]
    assert goal_set["params"]["objective"] == "Repair the failing API tests."
    resume = [entry for entry in log_entries if entry["kind"] == "resume"][0]
    assert resume["args"][-2:] == ["thread-123", "Use pytest after each change."]


def test_codex_adapter_allows_env_override_for_sandbox_mode(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "env": {"AGENTFLOW_CODEX_SANDBOX_MODE": "danger-full-access"},
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert "--sandbox" in prepared.command
    sandbox_index = prepared.command.index("--sandbox")
    assert prepared.command[sandbox_index + 1] == "danger-full-access"
    assert "AGENTFLOW_CODEX_SANDBOX_MODE" not in prepared.env


def test_codex_adapter_does_not_force_runtime_codex_home_for_model_only_nodes(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "model": "gpt-5-codex",
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert "CODEX_HOME" not in prepared.env
    assert prepared.runtime_files == {}


def test_codex_adapter_uses_runtime_codex_home_for_mcp_config(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "mcps": [
                {
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
                }
            ],
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert prepared.env["CODEX_HOME"] == str(tmp_path / ".runtime" / "codex_home")
    assert prepared.runtime_files.keys() == {"codex_home/config.toml"}
    assert "[mcp_servers.filesystem]" in prepared.runtime_files["codex_home/config.toml"]
    assert 'command = "npx"' in prepared.runtime_files["codex_home/config.toml"]


def test_codex_adapter_isolates_home_when_runtime_codex_home_is_used(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "provider": {
                "name": "openai-pinned",
                "base_url": "http://example.test/v1",
                "api_key_env": "OPENAI_API_KEY",
                "wire_api": "responses",
            },
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    expected_home = str(tmp_path / ".runtime" / "codex_home")
    assert prepared.env["CODEX_HOME"] == expected_home
    assert prepared.env["HOME"] == expected_home
    assert prepared.runtime_files.keys() == {"codex_home/config.toml"}


def test_codex_adapter_can_ignore_repo_instructions_with_isolated_runtime_cwd(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "repo_instructions_mode": "ignore",
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    expected_home = str(tmp_path / ".runtime" / "codex_home")
    assert prepared.env["CODEX_HOME"] == expected_home
    assert prepared.env["HOME"] == expected_home
    assert prepared.cwd == str(tmp_path / ".runtime")
    assert "--disable" in prepared.command
    disable_index = prepared.command.index("--disable")
    assert prepared.command[disable_index + 1] == "plugins"
    assert "--add-dir" in prepared.command
    add_dir_index = prepared.command.index("--add-dir")
    assert prepared.command[add_dir_index + 1] == str(tmp_path)


def test_claude_adapter_uses_tools_flag_for_read_only_access(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert "--allowedTools" not in prepared.command
    index = prepared.command.index("--tools")
    assert prepared.command[index + 1] == "Read,Glob,Grep,LS,NotebookRead,Task,TaskOutput,TodoRead,WebFetch,WebSearch"


def test_claude_adapter_uses_tools_flag_for_read_write_access(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "implement",
            "agent": "claude",
            "prompt": "Implement",
            "tools": "read_write",
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Implement", _paths(tmp_path))

    index = prepared.command.index("--tools")
    assert "Bash" in prepared.command[index + 1].split(",")
    assert "Write" in prepared.command[index + 1].split(",")


def test_claude_adapter_can_ignore_repo_instructions_with_bare_runtime_cwd(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "repo_instructions_mode": "ignore",
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert "--bare" in prepared.command
    assert "--add-dir" in prepared.command
    add_dir_index = prepared.command.index("--add-dir")
    assert prepared.command[add_dir_index + 1] == str(tmp_path)
    assert prepared.cwd == str(tmp_path / ".runtime")


def test_claude_adapter_supports_kimi_provider_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-kimi-secret")
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "provider": "kimi",
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["ANTHROPIC_BASE_URL"] == "https://api.kimi.com/coding/"
    assert prepared.env["ANTHROPIC_API_KEY"] == "test-kimi-secret"


def test_kimi_adapter_uses_kimi_cli_directly(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "kimi",
            "prompt": "Review",
        }
    )

    prepared = KimiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.command[0] == "kimi"
    assert "--print" in prepared.command
    assert "--output-format" in prepared.command
    assert "stream-json" in prepared.command
    assert "--yolo" in prepared.command
    assert "-p" in prepared.command
    assert "Review" in prepared.command


def test_kimi_adapter_passes_model_flag(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "kimi",
            "prompt": "Review",
            "model": "kimi-k2-turbo-preview",
        }
    )

    prepared = KimiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert "--model" in prepared.command
    model_index = prepared.command.index("--model")
    assert prepared.command[model_index + 1] == "kimi-k2-turbo-preview"


def test_kimi_adapter_respects_custom_executable(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "kimi",
            "prompt": "Review",
            "executable": "/usr/local/bin/kimi",
        }
    )

    prepared = KimiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.command[0] == "/usr/local/bin/kimi"


def test_kimi_adapter_can_ignore_repo_instructions_with_isolated_runtime_cwd(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "kimi",
            "prompt": "Review",
            "repo_instructions_mode": "ignore",
        }
    )

    prepared = KimiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert "--add-dir" in prepared.command
    add_dir_index = prepared.command.index("--add-dir")
    assert prepared.command[add_dir_index + 1] == str(tmp_path)
    assert "--skills-dir" in prepared.command
    skills_dir_index = prepared.command.index("--skills-dir")
    assert prepared.command[skills_dir_index + 1] == str(tmp_path / ".runtime" / "empty-skills")
    assert prepared.cwd == str(tmp_path / ".runtime")
    assert prepared.runtime_files.keys() == {"empty-skills/.gitkeep"}


def test_claude_adapter_prefers_node_env_over_provider_env(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "env": {"SHARED_FLAG": "node", "ANTHROPIC_API_KEY": "node-secret"},
            "provider": {
                "name": "kimi-proxy",
                "base_url": "https://example.test/anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "env": {"SHARED_FLAG": "provider", "ANTHROPIC_API_KEY": "provider-secret"},
            },
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["SHARED_FLAG"] == "node"
    assert prepared.env["ANTHROPIC_API_KEY"] == "node-secret"


def test_claude_adapter_respects_node_env_clear_for_custom_provider_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "ambient-secret")
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "env": {"TEST_CLAUDE_API_KEY": ""},
            "provider": {
                "name": "kimi-proxy",
                "base_url": "https://example.test/anthropic",
                "api_key_env": "TEST_CLAUDE_API_KEY",
            },
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["TEST_CLAUDE_API_KEY"] == ""
    assert prepared.env["ANTHROPIC_API_KEY"] == ""


def test_claude_adapter_respects_provider_env_clear_for_custom_provider_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "ambient-secret")
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "claude",
            "prompt": "Review",
            "provider": {
                "name": "kimi-proxy",
                "base_url": "https://example.test/anthropic",
                "api_key_env": "TEST_CLAUDE_API_KEY",
                "env": {"TEST_CLAUDE_API_KEY": ""},
            },
        }
    )

    prepared = ClaudeAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["TEST_CLAUDE_API_KEY"] == ""
    assert prepared.env["ANTHROPIC_API_KEY"] == ""


def test_codex_adapter_prefers_node_env_over_provider_env(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "env": {"SHARED_FLAG": "node", "OPENAI_API_KEY": "node-secret"},
            "provider": {
                "name": "openai-proxy",
                "base_url": "https://example.test/openai",
                "api_key_env": "OPENAI_API_KEY",
                "wire_api": "responses",
                "env": {"SHARED_FLAG": "provider", "OPENAI_API_KEY": "provider-secret"},
            },
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert prepared.env["SHARED_FLAG"] == "node"
    assert prepared.env["OPENAI_API_KEY"] == "node-secret"


def test_codex_adapter_preserves_empty_openai_base_url_override(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "plan",
            "agent": "codex",
            "prompt": "Plan",
            "env": {"OPENAI_BASE_URL": ""},
        }
    )

    prepared = CodexAdapter().prepare(node, "Plan", _paths(tmp_path))

    assert "OPENAI_BASE_URL" in prepared.env
    assert prepared.env["OPENAI_BASE_URL"] == ""


def test_kimi_adapter_prefers_node_env_over_provider_env(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "kimi",
            "prompt": "Review",
            "env": {"SHARED_FLAG": "node", "KIMI_API_KEY": "node-secret"},
            "provider": {
                "name": "moonshot-proxy",
                "base_url": "https://example.test/moonshot",
                "api_key_env": "KIMI_API_KEY",
                "env": {"SHARED_FLAG": "provider", "KIMI_API_KEY": "provider-secret"},
            },
        }
    )

    prepared = KimiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.env["SHARED_FLAG"] == "node"
    assert prepared.env["KIMI_API_KEY"] == "node-secret"


def test_pi_adapter_uses_pi_cli_directly(tmp_path):
    node = NodeSpec.model_validate({"id": "review", "agent": "pi", "prompt": "Review"})
    prepared = PiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert prepared.command[0] == "pi"
    # Always-on flags for non-interactive pipeline execution.
    assert prepared.command[1:5] == ["--print", "--mode", "json", "--no-session"]
    # Prompt is piped via stdin so Pi's positional-message parser cannot
    # interpret it as a flag or @file reference.
    assert prepared.stdin == "Review"
    assert "Review" not in prepared.command


def test_pi_adapter_read_only_tool_mapping(tmp_path):
    node = NodeSpec.model_validate(
        {"id": "scan", "agent": "pi", "prompt": "Scan", "tools": "read_only"}
    )
    prepared = PiAdapter().prepare(node, "Scan", _paths(tmp_path))

    tools_idx = prepared.command.index("--tools")
    assert prepared.command[tools_idx + 1] == "read,grep,find,ls"


def test_pi_adapter_read_write_tool_mapping(tmp_path):
    node = NodeSpec.model_validate(
        {"id": "impl", "agent": "pi", "prompt": "Implement", "tools": "read_write"}
    )
    prepared = PiAdapter().prepare(node, "Implement", _paths(tmp_path))

    tools_idx = prepared.command.index("--tools")
    assert prepared.command[tools_idx + 1] == "read,bash,edit,write,grep,find,ls"


def test_pi_adapter_passes_model_flag(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "pi",
            "prompt": "Review",
            "model": "lmstudio/mythos-26b-a4b-prism-pro-dq-mlx",
        }
    )
    prepared = PiAdapter().prepare(node, "Review", _paths(tmp_path))

    model_idx = prepared.command.index("--model")
    assert prepared.command[model_idx + 1] == "lmstudio/mythos-26b-a4b-prism-pro-dq-mlx"
    # Model already has a provider prefix, so `--provider` should not be added.
    assert "--provider" not in prepared.command


def test_pi_adapter_passes_provider_name_when_model_bare(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "review",
            "agent": "pi",
            "prompt": "Review",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6:high",
        }
    )
    prepared = PiAdapter().prepare(node, "Review", _paths(tmp_path))

    assert "--provider" in prepared.command
    provider_idx = prepared.command.index("--provider")
    assert prepared.command[provider_idx + 1] == "anthropic"


def test_pi_adapter_materializes_scoped_models_json(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "scan",
            "agent": "pi",
            "prompt": "Scan",
            "provider": {
                "name": "lmstudio-remote",
                "base_url": "http://192.168.1.42:1234/v1",
                "wire_api": "openai-completions",
                "api_key_env": "LMSTUDIO_API_KEY",
            },
            "model": "lmstudio-remote/qwen3.6-27b",
        }
    )
    prepared = PiAdapter().prepare(node, "Scan", _paths(tmp_path))

    # When a ProviderConfig with base_url is supplied, the adapter writes a
    # scoped models.json and points Pi at it via PI_CODING_AGENT_DIR rather
    # than passing `--provider` on the command line.
    assert "--provider" not in prepared.command
    assert "PI_CODING_AGENT_DIR" in prepared.env
    scoped_dir = prepared.env["PI_CODING_AGENT_DIR"]
    assert scoped_dir.endswith("/pi-home/agent")

    models_rel = str(Path("pi-home") / "agent" / "models.json")
    assert models_rel in prepared.runtime_files
    parsed = json.loads(prepared.runtime_files[models_rel])
    entry = parsed["providers"]["lmstudio-remote"]
    assert entry["baseUrl"] == "http://192.168.1.42:1234/v1"
    assert entry["api"] == "openai-completions"
    assert entry["apiKey"] == "LMSTUDIO_API_KEY"
    # Provider prefix is stripped from model id in the scoped entry.
    assert entry["models"] == [{"id": "qwen3.6-27b"}]

    settings_rel = str(Path("pi-home") / "agent" / "settings.json")
    assert settings_rel in prepared.runtime_files


def test_pi_adapter_honors_repo_instructions_ignore(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "scan",
            "agent": "pi",
            "prompt": "Scan",
            "repo_instructions_mode": "ignore",
        }
    )
    prepared = PiAdapter().prepare(node, "Scan", _paths(tmp_path))

    assert "--no-skills" in prepared.command
    assert "--no-extensions" in prepared.command
    assert "--no-prompt-templates" in prepared.command
    # cwd moves out of the project workdir to avoid picking up AGENTS.md.
    assert prepared.cwd == str(tmp_path / ".runtime")


def test_pi_adapter_rejects_mcp_servers(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "scan",
            "agent": "pi",
            "prompt": "Scan",
            "mcps": [{"name": "demo", "transport": "stdio", "command": "echo"}],
        }
    )
    with pytest.raises(ValueError, match="pi adapter does not support `mcps`"):
        PiAdapter().prepare(node, "Scan", _paths(tmp_path))


def test_pi_adapter_forwards_api_key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_API_KEY", "lm-studio-secret")
    node = NodeSpec.model_validate(
        {
            "id": "scan",
            "agent": "pi",
            "prompt": "Scan",
            "provider": {
                "name": "lmstudio-remote",
                "base_url": "http://remote:1234/v1",
                "api_key_env": "LMSTUDIO_API_KEY",
            },
            "model": "mythos-26b",
        }
    )
    prepared = PiAdapter().prepare(node, "Scan", _paths(tmp_path))

    assert prepared.env.get("LMSTUDIO_API_KEY") == "lm-studio-secret"


def test_pi_adapter_preserves_extra_args(tmp_path):
    node = NodeSpec.model_validate(
        {
            "id": "scan",
            "agent": "pi",
            "prompt": "Scan",
            "extra_args": ["--thinking", "high"],
        }
    )
    prepared = PiAdapter().prepare(node, "Scan", _paths(tmp_path))

    assert prepared.command[-2:] == ["--thinking", "high"]
