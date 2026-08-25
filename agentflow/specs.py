from __future__ import annotations

import ipaddress
import os
import posixpath
import re
import shlex
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentflow.local_shell import (
    invalid_bash_long_option_error,
    shell_init_commands,
    shell_init_uses_kimi_helper,
    shell_wrapper_requires_command_placeholder,
    target_disables_bash_login_startup,
    target_disables_bash_rc_startup,
    target_uses_bash,
    target_uses_interactive_bash,
    target_uses_login_bash,
)


class AgentKind(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"
    KIMI = "kimi"
    PI = "pi"
    PYTHON = "python"
    SHELL = "shell"
    SYNC = "sync"


class ToolAccess(StrEnum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class CaptureMode(StrEnum):
    FINAL = "final"
    TRACE = "trace"


class RepoInstructionsMode(StrEnum):
    INHERIT = "inherit"
    IGNORE = "ignore"


class PeriodicActuationMode(StrEnum):
    NONE = "none"
    OUTPUT_JSON = "output_json"


class NodeStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    READY = "ready"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


_INTERACTIVE_AGENT_KINDS = {AgentKind.CODEX, AgentKind.CLAUDE, AgentKind.KIMI, AgentKind.PI}


def normalize_agent_name(value: str | AgentKind) -> str:
    if isinstance(value, AgentKind):
        return value.value
    return str(value).strip()


def builtin_agent_kind(value: str | AgentKind | None) -> AgentKind | None:
    if isinstance(value, AgentKind):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return AgentKind(normalized)
    except ValueError:
        return None


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    base_url: str | None = None
    api_key_env: str | None = None
    wire_api: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class InferenceSetupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: str
    model: str
    engine: Literal["vllm", "sglang"] = "vllm"
    use_spot: bool = True
    max_hourly_cost: float | None = Field(default=None, gt=0)
    image_id: str | None = None
    name: str | None = None
    cluster_name: str | None = None
    api_key: str | None = None
    port: int = Field(default=8000, ge=1, le=65535)
    idle_minutes_to_autostop: int = Field(default=60, ge=0)
    retry_until_up: bool = False
    endpoint_timeout_seconds: int = Field(default=600, ge=1)

    @field_validator("gpu", "model")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"`inference.{info.field_name}` must not be empty")
        return normalized

    @field_validator("image_id", "name", "cluster_name", "api_key")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


_KIMI_ANTHROPIC_BASE_URL = "https://api.kimi.com/coding/"
_LOCAL_KIMI_BOOTSTRAP_SHELL_INIT = ("command -v kimi >/dev/null 2>&1", "kimi")
_LOCAL_BOOTSTRAP_TARGET_KEYS = ("shell", "shell_login", "shell_interactive", "shell_init")
_FANOUT_ALIAS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FANOUT_RESERVED_CONTEXT_NAMES = {"fanout", "fanouts", "nodes", "pipeline"}
_FANOUT_MEMBER_RESERVED_NAMES = {"index", "number", "count", "suffix", "value", "template_id", "node_id"}
_FANOUT_TEMPLATE_PATTERN = re.compile(r"{{\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*}}")
_FANOUT_EXPANSION_MODE_KEYS = ("count", "values", "matrix", "group_by", "batches")
_NODE_DEFAULT_FORBIDDEN_FIELDS = {
    "id",
    "prompt",
    "depends_on",
    "fanout",
    "fanout_group",
    "fanout_member",
    "fanout_dependencies",
}
_NODE_DEFAULT_LIST_MERGE_FIELDS = {"extra_args", "skills", "mcps"}
_NODE_DEFAULT_DICT_MERGE_FIELDS = {"env", "provider"}


def _normalize_local_bootstrap(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _local_bootstrap_defaults(bootstrap: str) -> dict[str, Any]:
    if bootstrap == "kimi":
        return {
            "shell": "bash",
            "shell_login": True,
            "shell_interactive": True,
            "shell_init": list(_LOCAL_KIMI_BOOTSTRAP_SHELL_INIT),
        }
    return {}


def _merge_bootstrap_shell_init(bootstrap: str, shell_init: Any) -> str | list[str] | None:
    defaults = _local_bootstrap_defaults(bootstrap)
    default_shell_init = defaults.get("shell_init")
    if default_shell_init is None:
        return shell_init
    if shell_init is None:
        return default_shell_init
    if bootstrap == "kimi" and shell_init_uses_kimi_helper(shell_init):
        return shell_init

    extra_commands = list(shell_init_commands(shell_init))
    if not extra_commands:
        return default_shell_init

    return [*extra_commands, *shell_init_commands(default_shell_init)]


def _normalized_provider_base_url(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped.rstrip("/")


def _shell_program(shell: str | None) -> str | None:
    if not isinstance(shell, str) or not shell.strip():
        return None
    try:
        parts = shlex.split(shell)
    except ValueError:
        return None
    if not parts:
        return None
    return os.path.basename(parts[0]) or None


def _normalized_provider_env_text(provider: ProviderConfig, key: str) -> str | None:
    raw_value = provider.env.get(key)
    if raw_value is None:
        return None
    stripped = str(raw_value).strip()
    if not stripped:
        return None
    return stripped


def _normalized_provider_env_base_url(provider: ProviderConfig, key: str) -> str | None:
    return _normalized_provider_base_url(_normalized_provider_env_text(provider, key))


def _coerce_base_dir(value: object) -> Path | None:
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser().resolve()
    return None


def provider_uses_kimi_anthropic_auth(provider: ProviderConfig | None) -> bool:
    if provider is None:
        return False

    effective_base_url = _normalized_provider_env_base_url(provider, "ANTHROPIC_BASE_URL")
    if effective_base_url is None:
        effective_base_url = _normalized_provider_base_url(provider.base_url)
    if effective_base_url is not None:
        return effective_base_url == _KIMI_ANTHROPIC_BASE_URL.rstrip("/")

    return (provider.name or "").strip().lower() in {"kimi", "moonshot", "moonshot-ai"}


def resolve_provider(value: str | ProviderConfig | None, agent: str | AgentKind) -> ProviderConfig | None:
    if value is None:
        return None
    if isinstance(value, ProviderConfig):
        return value

    resolved_agent = builtin_agent_kind(agent)
    if resolved_agent is None:
        return ProviderConfig(name=value)

    alias = value.strip().lower()
    if alias == "openai" and resolved_agent == AgentKind.CODEX:
        return ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            wire_api="responses",
        )
    if alias == "anthropic" and resolved_agent == AgentKind.CLAUDE:
        return ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
        )
    if alias in {"kimi", "moonshot", "moonshot-ai"}:
        if resolved_agent == AgentKind.CLAUDE:
            return ProviderConfig(
                name="kimi",
                base_url="https://api.kimi.com/coding/",
                api_key_env="ANTHROPIC_API_KEY",
            )
        if resolved_agent == AgentKind.KIMI:
            return ProviderConfig(
                name="moonshot",
                base_url="https://api.moonshot.ai/v1",
                api_key_env="KIMI_API_KEY",
            )
        raise ValueError(
            "provider 'kimi' is not supported for codex nodes because Codex requires an "
            "OpenAI Responses API backend and Kimi's public endpoints do not expose /responses"
        )
    if alias in {"minimax", "minimax-cn"}:
        # MiniMax exposes an Anthropic-compatible endpoint (for Claude nodes) and an
        # OpenAI-compatible chat-completions endpoint (for Pi nodes, which materialize a
        # scoped models.json). Its OpenAI surface is chat completions, not /responses, so
        # it cannot back Codex nodes.
        china_region = alias == "minimax-cn"
        if resolved_agent == AgentKind.CLAUDE:
            return ProviderConfig(
                name="minimax",
                base_url=(
                    "https://api.minimaxi.com/anthropic"
                    if china_region
                    else "https://api.minimax.io/anthropic"
                ),
                api_key_env="MINIMAX_API_KEY",
            )
        if resolved_agent == AgentKind.PI:
            return ProviderConfig(
                name="minimax",
                base_url="https://api.minimaxi.com/v1" if china_region else "https://api.minimax.io/v1",
                api_key_env="MINIMAX_API_KEY",
            )
        raise ValueError(
            f"provider '{alias}' is only supported for `pi` and `claude` nodes; MiniMax's "
            "OpenAI-compatible endpoints expose chat completions, not the Responses API Codex requires"
        )
    return ProviderConfig(name=value)


def resolve_execution_provider(value: str | ProviderConfig | None, agent: str | AgentKind) -> ProviderConfig | None:
    provider = resolve_provider(value, agent)
    if provider is not None:
        return provider
    if builtin_agent_kind(agent) == AgentKind.KIMI:
        return ProviderConfig(
            name="moonshot",
            base_url="https://api.moonshot.ai/v1",
            api_key_env="KIMI_API_KEY",
        )
    return None


class MCPServerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerSpec":
        if self.transport == "stdio":
            if not self.command or not self.command.strip():
                raise ValueError("stdio MCP servers require `command`")
            unsupported_fields = []
            if self.url and self.url.strip():
                unsupported_fields.append("url")
            if self.headers:
                unsupported_fields.append("headers")
        else:
            if not self.url or not self.url.strip():
                raise ValueError("streamable_http MCP servers require `url`")
            unsupported_fields = []
            if self.command and self.command.strip():
                unsupported_fields.append("command")
            if self.args:
                unsupported_fields.append("args")
            if self.env:
                unsupported_fields.append("env")

        if unsupported_fields:
            joined = ", ".join(f"`{field}`" for field in unsupported_fields)
            raise ValueError(f"{self.transport} MCP servers do not support {joined}")
        return self


class LocalTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _SHELL_COMMAND_PLACEHOLDER_MESSAGE = (
        "`target.shell` already includes a shell command payload. Add `{command}` where AgentFlow should "
        "inject the prepared agent command."
    )

    kind: Literal["local"] = "local"
    cwd: str | None = None
    bootstrap: str | None = None
    shell: str | None = None
    shell_login: bool = False
    shell_interactive: bool = False
    shell_init: str | list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_bootstrap_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        bootstrap = _normalize_local_bootstrap(data.get("bootstrap"))
        if bootstrap is None:
            return data

        updated = dict(data)
        for key, value in _local_bootstrap_defaults(bootstrap).items():
            if key == "shell_init":
                updated[key] = _merge_bootstrap_shell_init(bootstrap, updated.get(key))
                continue
            if key not in updated or updated[key] is None:
                updated[key] = value
        return updated

    @field_validator("bootstrap")
    @classmethod
    def validate_bootstrap(cls, value: str | None) -> str | None:
        normalized = _normalize_local_bootstrap(value)
        if normalized is None:
            return None
        if normalized != "kimi":
            raise ValueError("`target.bootstrap` must be `kimi`")
        return normalized

    @field_validator("shell_init")
    @classmethod
    def validate_shell_init(cls, value: str | list[str] | None) -> str | list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("`target.shell_init` must not be empty")
            return normalized

        normalized_commands = [command.strip() for command in value if command.strip()]
        if not normalized_commands:
            raise ValueError("`target.shell_init` must contain at least one non-empty command")
        if len(normalized_commands) != len(value):
            raise ValueError("`target.shell_init` list entries must be non-empty strings")
        return normalized_commands

    @model_validator(mode="after")
    def validate_shell_bootstrap(self) -> "LocalTarget":
        if self.shell and self.shell.strip():
            invalid_option_error = invalid_bash_long_option_error(self.shell)
            if invalid_option_error is not None:
                raise ValueError(f"`target.shell` uses an unsupported bash long option. {invalid_option_error}")
            if shell_wrapper_requires_command_placeholder(self.shell):
                raise ValueError(self._SHELL_COMMAND_PLACEHOLDER_MESSAGE)
        else:
            missing_shell_fields: list[str] = []
            if self.shell_login:
                missing_shell_fields.append("shell_login")
            if self.shell_interactive:
                missing_shell_fields.append("shell_interactive")
            if self.shell_init:
                missing_shell_fields.append("shell_init")
            if missing_shell_fields:
                joined = ", ".join(f"`target.{field}`" for field in missing_shell_fields)
                raise ValueError(f"{joined} require `target.shell` on local targets")

        if self.bootstrap == "kimi":
            target_shell = _shell_program(self.shell) or "this shell"
            if not target_uses_bash(self):
                raise ValueError(
                    f"`target.bootstrap: kimi` requires bash-style shell bootstrap, but `target.shell` resolves "
                    f"to `{target_shell}`. Use `shell: bash` with `target.shell_interactive: true`, use `bash -lic`, "
                    "or drop `target.bootstrap` and configure the bootstrap explicitly."
                )
            if not target_uses_interactive_bash(self):
                raise ValueError(
                    "`target.bootstrap: kimi` requires interactive bash startup so helpers from `~/.bashrc` are "
                    "available. Set `target.shell_interactive: true`, use `bash -lic`, or drop `target.bootstrap` "
                    "and configure the bootstrap explicitly."
                )
            if target_uses_login_bash(self) and target_disables_bash_login_startup(self):
                raise ValueError(
                    "`target.bootstrap: kimi` cannot use bash with `--noprofile` because login startup files will "
                    "not load the `kimi` helper. Remove `--noprofile` or drop `target.bootstrap` and configure the "
                    "bootstrap explicitly."
                )
            if not target_uses_login_bash(self) and target_disables_bash_rc_startup(self):
                raise ValueError(
                    "`target.bootstrap: kimi` cannot use bash with `--norc` because interactive startup will not "
                    "load `~/.bashrc` and the `kimi` helper will usually be unavailable. Remove `--norc` or drop "
                    "`target.bootstrap` and configure the bootstrap explicitly."
                )
        return self


class ContainerTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["container"] = "container"
    image: str
    engine: str = "docker"
    workdir_mount: str = "/workspace"
    runtime_mount: str = "/agentflow-runtime"
    app_mount: str = "/agentflow-app"
    extra_args: list[str] = Field(default_factory=list)
    entrypoint: str | None = None


class DockerMount(BaseModel):
    """An explicit host bind mount made available to a Docker target."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    read_only: bool = False

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`target.mounts[].source` must not be empty")
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.mounts[].source` must not contain NUL bytes or commas")
        return normalized

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`target.mounts[].target` must not be empty")
        if not normalized.startswith("/"):
            raise ValueError("`target.mounts[].target` must be an absolute container path")
        if normalized.startswith("//"):
            raise ValueError("`target.mounts[].target` must use a single leading slash")
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.mounts[].target` must not contain NUL bytes or commas")
        return posixpath.normpath(normalized)


class DockerNetworkPolicy(BaseModel):
    """Docker's native network attachment policy for an agent container.

    ``custom`` attaches to a pre-created, user-managed Docker network. That is
    the extension point for an egress proxy or firewall when callers need a
    narrower policy than Docker's built-in bridge/host/none modes.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "bridge", "host", "custom"] = "bridge"
    name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    dns: list[str] = Field(default_factory=list)
    add_hosts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            normalized = data.strip()
            if not normalized:
                raise ValueError("`target.network_policy` must not be empty")
            if normalized in {"none", "bridge", "host"}:
                return {"mode": normalized}
            return {"mode": "custom", "name": normalized}
        if isinstance(data, dict) and "network" in data and "name" not in data:
            updated = dict(data)
            updated["name"] = updated.pop("network")
            return updated
        return data

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.network_policy.name` must not contain NUL bytes or commas")
        return normalized or None

    @field_validator("aliases", "dns")
    @classmethod
    def validate_non_empty_list_entries(cls, value: list[str], info) -> list[str]:
        normalized = [entry.strip() for entry in value]
        if any(not entry for entry in normalized):
            raise ValueError(f"`target.network_policy.{info.field_name}` entries must not be empty")
        if any("\x00" in entry for entry in normalized):
            raise ValueError(f"`target.network_policy.{info.field_name}` entries must not contain NUL bytes")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"`target.network_policy.{info.field_name}` entries must be unique")
        return normalized

    @field_validator("add_hosts")
    @classmethod
    def validate_add_hosts(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_host, raw_address in value.items():
            host = raw_host.strip()
            address = raw_address.strip()
            if not host or not address:
                raise ValueError("`target.network_policy.add_hosts` names and addresses must not be empty")
            if any(character in host for character in ("\x00", ",", ":", "=")):
                raise ValueError(
                    "`target.network_policy.add_hosts` host names must not contain NUL, comma, colon, or equals"
                )
            if "\x00" in address or "," in address or "=" in address:
                raise ValueError(
                    "`target.network_policy.add_hosts` addresses must not contain NUL, comma, or equals"
                )
            if address != "host-gateway":
                try:
                    ipaddress.ip_address(address)
                except ValueError as exc:
                    raise ValueError(
                        "`target.network_policy.add_hosts` addresses must be IP addresses or `host-gateway`"
                    ) from exc
            normalized[host] = address
        return normalized

    @model_validator(mode="after")
    def validate_mode(self) -> "DockerNetworkPolicy":
        if self.mode == "custom":
            if self.name is None:
                raise ValueError("`target.network_policy.name` is required when mode is `custom`")
            if self.name in {"none", "bridge", "host"} or self.name.startswith("container:"):
                raise ValueError(
                    "`target.network_policy.name` must identify a user-managed Docker network, "
                    "not a built-in or container network mode"
                )
        elif self.name is not None:
            raise ValueError("`target.network_policy.name` is only valid when mode is `custom`")
        if self.aliases and self.mode != "custom":
            raise ValueError("`target.network_policy.aliases` requires mode `custom`")
        return self

    @property
    def docker_network(self) -> str:
        return self.name if self.mode == "custom" and self.name is not None else self.mode


_DOCKER_EXTRA_ARG_VALUE_OPTIONS = {
    "--cpu-shares",
    "--cpus",
    "--cpuset-cpus",
    "--hostname",
    "--label",
    "--log-driver",
    "--log-opt",
    "--memory",
    "--memory-reservation",
    "--memory-swap",
    "--pids-limit",
    "--platform",
    "--pull",
    "--shm-size",
    "--stop-signal",
    "--stop-timeout",
    "--ulimit",
}
_DOCKER_EXTRA_ARG_BOOLEAN_OPTIONS = {
    "--init",
    "--oom-kill-disable",
    "--read-only",
}


def _docker_container_paths_overlap(left: str, right: str) -> bool:
    """Return whether either absolute container path contains the other."""

    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


class DockerTarget(BaseModel):
    """Run an agent in a local Docker container with explicit isolation controls."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["docker"] = "docker"
    image: str = "agentflow-agents:latest"
    engine: str = "docker"
    workdir_mount: str = "/workspace"
    runtime_mount: str = "/agentflow-runtime"
    app_mount: str | None = None
    workdir_read_only: bool = False
    user: str | None = "host"
    inherit_credentials: bool = False
    mounts: list[DockerMount] = Field(default_factory=list)
    network_policy: DockerNetworkPolicy = Field(default_factory=DockerNetworkPolicy)
    privileged: bool = False
    mount_docker_daemon: bool = False
    docker_daemon_socket: str | None = None
    dind: bool = False
    extra_args: list[str] = Field(default_factory=list)
    entrypoint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_compatibility_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        updated = dict(data)
        if "mount_docker_socket" in updated and "mount_docker_daemon" not in updated:
            updated["mount_docker_daemon"] = updated.pop("mount_docker_socket")
        if "network" in updated and "network_policy" not in updated:
            updated["network_policy"] = updated.pop("network")
        return updated

    @field_validator("image", "engine")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"`target.{info.field_name}` must not be empty")
        if "\x00" in normalized:
            raise ValueError(f"`target.{info.field_name}` must not contain NUL bytes")
        if info.field_name == "image" and (
            normalized.startswith("-") or any(character.isspace() for character in normalized)
        ):
            raise ValueError("`target.image` must be a Docker image reference, not a command-line option")
        return normalized

    @field_validator("workdir_mount", "runtime_mount")
    @classmethod
    def validate_required_container_path(cls, value: str, info) -> str:
        normalized = value.strip()
        if not normalized.startswith("/"):
            raise ValueError(f"`target.{info.field_name}` must be an absolute container path")
        if normalized.startswith("//"):
            raise ValueError(f"`target.{info.field_name}` must use a single leading slash")
        if "\x00" in normalized or "," in normalized:
            raise ValueError(f"`target.{info.field_name}` must not contain NUL bytes or commas")
        return posixpath.normpath(normalized)

    @field_validator("app_mount")
    @classmethod
    def validate_optional_container_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized.startswith("/"):
            raise ValueError("`target.app_mount` must be an absolute container path or null")
        if normalized.startswith("//"):
            raise ValueError("`target.app_mount` must use a single leading slash")
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.app_mount` must not contain NUL bytes or commas")
        return posixpath.normpath(normalized)

    @field_validator("entrypoint")
    @classmethod
    def normalize_entrypoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("`target.entrypoint` must not be empty")
        if "\x00" in normalized:
            raise ValueError("`target.entrypoint` must not contain NUL bytes")
        return normalized

    @field_validator("user")
    @classmethod
    def normalize_user(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("`target.user` must be `host`, a Docker user/group value, or null")
        if "\x00" in normalized or any(character.isspace() for character in normalized):
            raise ValueError("`target.user` must not contain whitespace or NUL bytes")
        return normalized

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, value: list[str]) -> list[str]:
        normalized = [argument.strip() for argument in value]
        if any(not argument for argument in normalized):
            raise ValueError("`target.extra_args` entries must not be empty")
        if any("\x00" in argument for argument in normalized):
            raise ValueError("`target.extra_args` entries must not contain NUL bytes")

        position = 0
        while position < len(normalized):
            argument = normalized[position]
            option, separator, inline_value = argument.partition("=")
            if option in _DOCKER_EXTRA_ARG_BOOLEAN_OPTIONS:
                if separator and inline_value not in {"true", "false"}:
                    raise ValueError(f"`target.extra_args` has an invalid boolean value for `{option}`")
                position += 1
                continue
            if option not in _DOCKER_EXTRA_ARG_VALUE_OPTIONS:
                if not argument.startswith("-"):
                    raise ValueError(
                        "`target.extra_args` cannot contain positional values; the Docker image is configured "
                        "with `target.image`"
                    )
                raise ValueError(
                    f"`target.extra_args` cannot set unsupported or isolation-sensitive option `{option}`"
                )
            if separator:
                if not inline_value:
                    raise ValueError(f"`target.extra_args` requires a value for `{option}`")
                position += 1
                continue
            if position + 1 >= len(normalized):
                raise ValueError(f"`target.extra_args` requires a value after `{option}`")
            if not normalized[position + 1]:
                raise ValueError(f"`target.extra_args` requires a non-empty value after `{option}`")
            position += 2
        return normalized

    @field_validator("docker_daemon_socket")
    @classmethod
    def validate_docker_daemon_socket(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(Path(value.strip()).expanduser())
        if not normalized or not Path(normalized).is_absolute():
            raise ValueError("`target.docker_daemon_socket` must be an absolute host path")
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.docker_daemon_socket` must not contain NUL bytes or commas")
        return normalized

    @model_validator(mode="after")
    def validate_isolation_options(self) -> "DockerTarget":
        if self.dind and not self.privileged:
            raise ValueError("`target.dind: true` requires `target.privileged: true`")
        if self.dind and self.mount_docker_daemon:
            raise ValueError("`target.dind` and `target.mount_docker_daemon` are mutually exclusive")
        if self.dind and self.entrypoint is not None:
            raise ValueError(
                "`target.entrypoint` cannot override the bundled image entrypoint when `target.dind` is enabled"
            )
        if self.dind and self.user not in {None, "host", "0", "0:0", "root", "root:root"}:
            raise ValueError(
                "`target.dind` supports `target.user: host` (drop to the invoking UID:GID after daemon "
                "startup) or an explicit root/image-default user"
            )
        if self.dind and any(
            argument.partition("=")[0] == "--read-only"
            and (not argument.partition("=")[1] or argument.partition("=")[2] == "true")
            for argument in self.extra_args
        ):
            raise ValueError("`target.dind` cannot be combined with `--read-only`")
        if self.docker_daemon_socket is not None and not self.mount_docker_daemon:
            raise ValueError(
                "`target.docker_daemon_socket` requires `target.mount_docker_daemon: true`"
            )

        managed_targets = [self.workdir_mount, self.runtime_mount]
        if self.app_mount is not None:
            managed_targets.append(self.app_mount)

        managed_overlaps = sorted(
            (left, right)
            for index, left in enumerate(managed_targets)
            for right in managed_targets[index + 1 :]
            if _docker_container_paths_overlap(left, right)
        )
        if managed_overlaps:
            rendered = ", ".join(f"{left} <> {right}" for left, right in managed_overlaps)
            raise ValueError(
                "AgentFlow-managed Docker mount targets must not overlap as ancestors or descendants: "
                + rendered
            )
        reserved_targets = {*managed_targets, "/var/run/docker.sock"}
        mount_targets = [mount.target for mount in self.mounts]
        duplicates = sorted(target for target, count in Counter(mount_targets).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate Docker mount targets: {duplicates}")
        mount_overlaps = sorted(
            (left, right)
            for index, left in enumerate(mount_targets)
            for right in mount_targets[index + 1 :]
            if left != right and _docker_container_paths_overlap(left, right)
        )
        if mount_overlaps:
            rendered = ", ".join(f"{left} <> {right}" for left, right in mount_overlaps)
            raise ValueError(
                "`target.mounts` targets must not overlap as ancestors or descendants: " + rendered
            )
        collisions = sorted(
            (mount_target, reserved_target)
            for mount_target in mount_targets
            for reserved_target in reserved_targets
            if _docker_container_paths_overlap(mount_target, reserved_target)
        )
        if collisions:
            rendered = ", ".join(
                mount_target if mount_target == reserved_target else f"{mount_target} <> {reserved_target}"
                for mount_target, reserved_target in collisions
            )
            raise ValueError(
                "`target.mounts` cannot overlap AgentFlow-managed mount targets as ancestors or descendants: "
                + rendered
            )

        return self


class CloudHypervisorMount(BaseModel):
    """A host directory shared with a Cloud Hypervisor guest through virtio-fs."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    read_only: bool = True

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`target.mounts[].source` must not be empty")
        if "\x00" in normalized:
            raise ValueError("`target.mounts[].source` must not contain NUL bytes")
        return normalized

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("/"):
            raise ValueError("`target.mounts[].target` must be an absolute guest path")
        if normalized.startswith("//"):
            raise ValueError("`target.mounts[].target` must use a single leading slash")
        if "\x00" in normalized or "," in normalized:
            raise ValueError(
                "`target.mounts[].target` must not contain NUL bytes or commas"
            )
        return posixpath.normpath(normalized)


class CloudHypervisorNetworkPolicy(BaseModel):
    """Network attachment and guest configuration for a Cloud Hypervisor VM."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "tap"] = "none"
    tap: str | None = None
    mac: str | None = None
    host_ip: str | None = None
    host_mask: str | None = None
    dhcp: bool = False
    guest_address: str | None = None
    gateway: str | None = None
    dns: list[str] = Field(default_factory=list)
    num_queues: int = Field(default=2, ge=2, le=256)

    @model_validator(mode="before")
    @classmethod
    def normalize_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            normalized = data.strip()
            if normalized == "none":
                return {"mode": "none"}
            if normalized == "tap":
                return {
                    "mode": "tap",
                    "host_ip": "192.168.249.1",
                    "host_mask": "255.255.255.0",
                    "guest_address": "192.168.249.2/24",
                    "gateway": "192.168.249.1",
                }
            if normalized:
                return {"mode": "tap", "tap": normalized, "dhcp": True}
            raise ValueError("`target.network_policy` must not be empty")
        return data

    @field_validator("tap")
    @classmethod
    def validate_tap(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", normalized):
            raise ValueError(
                "`target.network_policy.tap` must be a Linux interface name of at most 15 characters"
            )
        return normalized

    @field_validator("mac")
    @classmethod
    def validate_mac(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}", normalized):
            raise ValueError(
                "`target.network_policy.mac` must be a colon-separated MAC address"
            )
        return normalized

    @field_validator("host_ip", "gateway")
    @classmethod
    def validate_ip_address(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        try:
            address = ipaddress.ip_address(normalized)
        except ValueError as exc:
            raise ValueError(
                f"`target.network_policy.{info.field_name}` must be an IP address"
            ) from exc
        if info.field_name == "host_ip" and address.version != 4:
            raise ValueError("`target.network_policy.host_ip` must be an IPv4 address")
        return str(address)

    @field_validator("host_mask")
    @classmethod
    def validate_host_mask(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        try:
            address = ipaddress.IPv4Address(normalized)
            ipaddress.IPv4Network(f"0.0.0.0/{address}")
        except ValueError as exc:
            raise ValueError(
                "`target.network_policy.host_mask` must be a contiguous IPv4 netmask"
            ) from exc
        return str(address)

    @field_validator("guest_address")
    @classmethod
    def validate_guest_address(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        try:
            address = ipaddress.ip_interface(normalized)
        except ValueError as exc:
            raise ValueError(
                "`target.network_policy.guest_address` must be an IP interface in CIDR form"
            ) from exc
        return str(address)

    @field_validator("dns")
    @classmethod
    def validate_dns(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for entry in value:
            try:
                normalized.append(str(ipaddress.ip_address(entry.strip())))
            except ValueError as exc:
                raise ValueError(
                    "`target.network_policy.dns` entries must be IP addresses"
                ) from exc
        if len(set(normalized)) != len(normalized):
            raise ValueError("`target.network_policy.dns` entries must be unique")
        return normalized

    @field_validator("num_queues")
    @classmethod
    def validate_num_queues(cls, value: int) -> int:
        if value % 2:
            raise ValueError("`target.network_policy.num_queues` must be even")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> CloudHypervisorNetworkPolicy:
        configured = any(
            (
                self.tap is not None,
                self.mac is not None,
                self.host_ip is not None,
                self.host_mask is not None,
                self.dhcp,
                self.guest_address is not None,
                self.gateway is not None,
                bool(self.dns),
                self.num_queues != 2,
            )
        )
        if self.mode == "none" and configured:
            raise ValueError(
                "Cloud Hypervisor network settings require `target.network_policy.mode: tap`"
            )
        if (self.host_ip is None) != (self.host_mask is None):
            raise ValueError("`host_ip` and `host_mask` must be configured together")
        if self.dhcp and self.guest_address is not None:
            raise ValueError("`dhcp` and `guest_address` are mutually exclusive")
        if self.gateway is not None and self.guest_address is None:
            raise ValueError("`gateway` requires a static `guest_address`")
        if self.guest_address is not None and self.gateway is not None:
            guest_version = ipaddress.ip_interface(self.guest_address).version
            gateway_version = ipaddress.ip_address(self.gateway).version
            if guest_version != gateway_version:
                raise ValueError(
                    "`guest_address` and `gateway` must use the same IP version"
                )
        if (
            self.host_ip is not None
            and self.guest_address is not None
            and ipaddress.ip_interface(self.guest_address).version != 4
        ):
            raise ValueError(
                "`host_ip` and `guest_address` must use the same IP version"
            )
        return self


_CLOUD_HYPERVISOR_SYSTEM_PATHS = ("/dev", "/proc", "/run", "/sys")


class CloudHypervisorTarget(BaseModel):
    """Boot an ephemeral AgentFlow guest with Cloud Hypervisor and virtio-fs."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cloud_hypervisor"] = "cloud_hypervisor"
    kernel: str
    rootfs: str
    binary: str = "cloud-hypervisor"
    virtiofsd: str = "virtiofsd"
    cpus: int = Field(default=2, ge=1, le=256)
    memory_mib: int = Field(default=4096, ge=256)
    workdir_mount: str = "/workspace"
    runtime_mount: str = "/agentflow-runtime"
    app_mount: str | None = None
    workdir_read_only: bool = False
    user: str | None = "host"
    inherit_credentials: bool = False
    mounts: list[CloudHypervisorMount] = Field(default_factory=list)
    network_policy: CloudHypervisorNetworkPolicy = Field(
        default_factory=CloudHypervisorNetworkPolicy
    )
    guest_agent_port: int = Field(default=4050, ge=1, le=65535)
    vsock_cid: int | None = Field(default=None, ge=3, le=4_294_967_294)
    boot_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    shutdown_timeout_seconds: float = Field(default=5.0, ge=0.1, le=60.0)
    init_path: str = "/usr/local/bin/agentflow-cloud-hypervisor-init"
    nss_wrapper_path: str | None = "/usr/lib/libnss_wrapper.so"
    kernel_args: list[str] = Field(default_factory=list)
    seccomp: Literal["true", "false", "log"] = "true"

    @field_validator("kernel", "rootfs", "binary", "virtiofsd")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"`target.{info.field_name}` must not be empty")
        if "\x00" in normalized:
            raise ValueError(f"`target.{info.field_name}` must not contain NUL bytes")
        if (
            info.field_name in {"binary", "virtiofsd"}
            and "/" in normalized
            and not Path(normalized).is_absolute()
        ):
            raise ValueError(
                f"`target.{info.field_name}` must be a PATH executable name or an absolute host path"
            )
        return normalized

    @field_validator("workdir_mount", "runtime_mount")
    @classmethod
    def validate_required_guest_path(cls, value: str, info) -> str:
        normalized = value.strip()
        if not normalized.startswith("/"):
            raise ValueError(
                f"`target.{info.field_name}` must be an absolute guest path"
            )
        if normalized.startswith("//"):
            raise ValueError(
                f"`target.{info.field_name}` must use a single leading slash"
            )
        if "\x00" in normalized or "," in normalized:
            raise ValueError(
                f"`target.{info.field_name}` must not contain NUL bytes or commas"
            )
        if info.field_name == "runtime_mount" and any(
            character in normalized for character in (":", "\r", "\n")
        ):
            raise ValueError(
                "`target.runtime_mount` must not contain colons or line breaks because it is used in the guest passwd file"
            )
        return posixpath.normpath(normalized)

    @field_validator("app_mount")
    @classmethod
    def validate_optional_guest_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized.startswith("/"):
            raise ValueError(
                "`target.app_mount` must be an absolute guest path or null"
            )
        if normalized.startswith("//"):
            raise ValueError("`target.app_mount` must use a single leading slash")
        if "\x00" in normalized or "," in normalized:
            raise ValueError("`target.app_mount` must not contain NUL bytes or commas")
        return posixpath.normpath(normalized)

    @field_validator("init_path")
    @classmethod
    def validate_init_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("/") or normalized.startswith("//"):
            raise ValueError("`target.init_path` must be an absolute guest path")
        if "\x00" in normalized or any(character.isspace() for character in normalized):
            raise ValueError(
                "`target.init_path` must not contain whitespace or NUL bytes"
            )
        return posixpath.normpath(normalized)

    @field_validator("nss_wrapper_path")
    @classmethod
    def validate_nss_wrapper_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized.startswith("/") or normalized.startswith("//"):
            raise ValueError(
                "`target.nss_wrapper_path` must be an absolute guest path or null"
            )
        if "\x00" in normalized or any(character.isspace() for character in normalized):
            raise ValueError(
                "`target.nss_wrapper_path` must not contain whitespace or NUL bytes"
            )
        return posixpath.normpath(normalized)

    @field_validator("user")
    @classmethod
    def validate_user(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if normalized in {"host", "root", "0", "0:0"}:
            return normalized
        if not re.fullmatch(r"[0-9]+(?::[0-9]+)?", normalized):
            raise ValueError(
                "`target.user` must be `host`, `root`, a numeric UID[:GID], or null"
            )
        if any(int(identifier) > 4_294_967_294 for identifier in normalized.split(":")):
            raise ValueError(
                "`target.user` UID and GID must fit Linux 32-bit identifiers"
            )
        return normalized

    @field_validator("kernel_args")
    @classmethod
    def validate_kernel_args(cls, value: list[str]) -> list[str]:
        normalized = [argument.strip() for argument in value]
        if any(not argument for argument in normalized):
            raise ValueError("`target.kernel_args` entries must not be empty")
        if any(
            "\x00" in argument or any(character.isspace() for character in argument)
            for argument in normalized
        ):
            raise ValueError(
                "`target.kernel_args` entries must be single arguments without whitespace or NUL bytes"
            )
        protected_prefixes = (
            "agentflow.guest_port=",
            "init=",
            "root=",
            "rootflags=",
            "rootfstype=",
        )
        if any(
            argument in {"--", "ro", "rw"} or argument.startswith(protected_prefixes)
            for argument in normalized
        ):
            raise ValueError(
                "`target.kernel_args` cannot override AgentFlow's root filesystem or guest init"
            )
        return normalized

    @model_validator(mode="after")
    def validate_mount_layout(self) -> CloudHypervisorTarget:
        managed_targets = [self.workdir_mount, self.runtime_mount]
        if self.app_mount is not None:
            managed_targets.append(self.app_mount)

        all_static_targets = [
            *managed_targets,
            *(mount.target for mount in self.mounts),
        ]
        for target in all_static_targets:
            for system_path in _CLOUD_HYPERVISOR_SYSTEM_PATHS:
                if _docker_container_paths_overlap(target, system_path):
                    raise ValueError(
                        f"Cloud Hypervisor guest mount target `{target}` overlaps reserved system path `{system_path}`"
                    )

        overlaps = sorted(
            (left, right)
            for index, left in enumerate(all_static_targets)
            for right in all_static_targets[index + 1 :]
            if _docker_container_paths_overlap(left, right)
        )
        if overlaps:
            rendered = ", ".join(f"{left} <> {right}" for left, right in overlaps)
            raise ValueError(
                "Cloud Hypervisor guest mount targets must not overlap as ancestors or descendants: "
                + rendered
            )
        return self


class SSHTarget(BaseModel):
    """Remote execution via SSH."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["ssh"] = "ssh"
    host: str
    port: int = 22
    username: str | None = None
    identity_file: str | None = None
    remote_workdir: str | None = None
    forward_credentials: bool = False


class EC2Target(BaseModel):
    """Run agent on a fresh EC2 instance, SSH in, execute, then terminate."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["ec2"] = "ec2"
    region: str = "us-east-1"
    ami: str | None = None
    instance_type: str = "t3.medium"
    key_name: str | None = None
    identity_file: str | None = None
    security_group_ids: list[str] = Field(default_factory=list)
    subnet_id: str | None = None
    username: str = "ubuntu"
    install_agents: list[str] = Field(default_factory=lambda: ["codex", "claude"])
    user_data: str | None = None
    spot: bool = False
    terminate: bool = True
    snapshot: bool = False
    shared: str | None = None


class ECSTarget(BaseModel):
    """Run agent as an ECS Fargate task."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["ecs"] = "ecs"
    region: str = "us-east-1"
    cluster: str = "agentflow"
    image: str | None = None
    dockerfile: str | None = None
    cpu: str = "1024"
    memory: str = "2048"
    subnets: list[str] = Field(default_factory=list)
    security_groups: list[str] = Field(default_factory=list)
    assign_public_ip: bool = True
    install_agents: list[str] = Field(default_factory=lambda: ["codex", "claude"])
    shared: str | None = None


TargetSpec = Annotated[
    LocalTarget | ContainerTarget | DockerTarget | CloudHypervisorTarget | SSHTarget | EC2Target | ECSTarget,
    Field(discriminator="kind"),
]


class OutputContainsCriterion(BaseModel):
    kind: Literal["output_contains"] = "output_contains"
    value: str
    case_sensitive: bool = False


class OutputRegexCriterion(BaseModel):
    """Regex match against the captured node output.

    ``value`` is a Python ``re`` pattern. ``multiline`` enables ``re.MULTILINE``
    so ``^``/``$`` match line boundaries (useful when checking that a status
    line appears anywhere in the agent's reply). ``case_sensitive`` defaults
    to ``True`` because regex authors typically express case explicitly.
    """

    kind: Literal["output_regex"] = "output_regex"
    value: str
    case_sensitive: bool = True
    multiline: bool = True


class FileExistsCriterion(BaseModel):
    kind: Literal["file_exists"] = "file_exists"
    path: str


class FileContainsCriterion(BaseModel):
    kind: Literal["file_contains"] = "file_contains"
    path: str
    value: str
    case_sensitive: bool = False


class FileNonEmptyCriterion(BaseModel):
    kind: Literal["file_nonempty"] = "file_nonempty"
    path: str


SuccessCriterion = Annotated[
    OutputContainsCriterion
    | OutputRegexCriterion
    | FileExistsCriterion
    | FileContainsCriterion
    | FileNonEmptyCriterion,
    Field(discriminator="kind"),
]


class FanoutGroupBySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    fields: list[str]

    @field_validator("from_")
    @classmethod
    def validate_source_group(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`fanout.group_by.from` must not be empty")
        return normalized

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("`fanout.group_by.fields` must contain at least one field")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_field in value:
            if not isinstance(raw_field, str):
                raise ValueError("`fanout.group_by.fields` entries must be strings")
            field = raw_field.strip()
            if not field:
                raise ValueError("`fanout.group_by.fields` entries must not be empty")
            if not _FANOUT_ALIAS_PATTERN.fullmatch(field):
                raise ValueError("`fanout.group_by.fields` entries must be valid member field names")
            if field in seen:
                raise ValueError(f"`fanout.group_by.fields` contains duplicate field `{field}`")
            seen.add(field)
            normalized.append(field)
        return normalized


class FanoutBatchesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    size: int = Field(gt=0)

    @field_validator("from_")
    @classmethod
    def validate_source_group(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`fanout.batches.from` must not be empty")
        return normalized


class FanoutSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    count: int | None = Field(default=None, ge=1)
    values: list[Any] | None = None
    matrix: dict[str, list[Any]] | None = None
    include: list[dict[str, Any]] | None = None
    exclude: list[dict[str, Any]] | None = None
    derive: dict[str, Any] = Field(default_factory=dict)
    as_: str = Field(default="item", alias="as")

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[Any] | None) -> list[Any] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("`fanout.values` must contain at least one item")
        return value

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, value: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("`fanout.matrix` must contain at least one axis")

        normalized: dict[str, list[Any]] = {}
        for axis_name, axis_values in value.items():
            axis = axis_name.strip()
            if not axis:
                raise ValueError("`fanout.matrix` axis names must not be empty")
            if not _FANOUT_ALIAS_PATTERN.fullmatch(axis):
                raise ValueError("`fanout.matrix` axis names must be valid template variable names")
            if axis in _FANOUT_MEMBER_RESERVED_NAMES:
                raise ValueError(
                    "`fanout.matrix` axis names must not use reserved member fields such as "
                    "`index`, `number`, `count`, `suffix`, `value`, `template_id`, or `node_id`"
                )
            if axis in normalized:
                raise ValueError(f"`fanout.matrix` axis `{axis}` was provided more than once")
            if not axis_values:
                raise ValueError(f"`fanout.matrix.{axis}` must contain at least one item")
            normalized[axis] = axis_values
        return normalized

    @field_validator("include")
    @classmethod
    def validate_include(cls, value: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("`fanout.include` must contain at least one item")
        return [_normalize_fanout_matrix_member(item) for item in value]

    @field_validator("exclude")
    @classmethod
    def validate_exclude(cls, value: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("`fanout.exclude` must contain at least one item")
        return value

    @field_validator("derive")
    @classmethod
    def validate_derive(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field_name, field_value in value.items():
            if not isinstance(field_name, str):
                raise ValueError("`fanout.derive` field names must be strings")
            field = field_name.strip()
            if not field:
                raise ValueError("`fanout.derive` field names must not be empty")
            if not _FANOUT_ALIAS_PATTERN.fullmatch(field):
                raise ValueError("`fanout.derive` field names must be valid template variable names")
            if field in _FANOUT_MEMBER_RESERVED_NAMES:
                raise ValueError(
                    "`fanout.derive` field names must not use reserved member fields such as "
                    "`index`, `number`, `count`, `suffix`, `value`, `template_id`, or `node_id`"
                )
            normalized[field] = field_value
        return normalized

    @field_validator("as_")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`fanout.as` must not be empty")
        if normalized in _FANOUT_RESERVED_CONTEXT_NAMES:
            raise ValueError(
                "`fanout.as` uses a reserved template variable name; choose something other than "
                "`fanout`, `fanouts`, `nodes`, `pipeline`, or `item`"
            )
        if not _FANOUT_ALIAS_PATTERN.fullmatch(normalized):
            raise ValueError("`fanout.as` must be a valid template variable name")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> "FanoutSpec":
        modes = (
            self.count is not None,
            self.values is not None,
            self.matrix is not None,
        )
        selected = sum(modes)
        if selected == 0:
            raise ValueError("fanout requires exactly one of `count`, `values`, or `matrix`")
        if selected > 1:
            raise ValueError("fanout accepts exactly one of `count`, `values`, or `matrix`")
        if (self.include is not None or self.exclude is not None) and self.matrix is None:
            raise ValueError("`fanout.include` and `fanout.exclude` require `fanout.matrix`")
        if self.matrix is not None and not _curate_fanout_matrix_members(
            self.matrix,
            include=self.include,
            exclude=self.exclude,
        ):
            raise ValueError("`fanout.matrix` produced no members after applying `fanout.exclude`")
        return self

    @property
    def member_values(self) -> list[Any]:
        if self.values is not None:
            return self.values
        if self.matrix is not None:
            return _curate_fanout_matrix_members(self.matrix, include=self.include, exclude=self.exclude)
        if self.count is None:
            return []
        return list(range(self.count))

    @property
    def member_count(self) -> int:
        if self.values is not None:
            return len(self.values)
        if self.matrix is not None:
            return len(_curate_fanout_matrix_members(self.matrix, include=self.include, exclude=self.exclude))
        if self.count is None:
            return 0
        return self.count


class PeriodicScheduleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_seconds: int = Field(ge=1)
    until_fanout_settles_from: str
    actuation: PeriodicActuationMode = PeriodicActuationMode.NONE

    @field_validator("until_fanout_settles_from")
    @classmethod
    def validate_until_fanout_settles_from(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("`schedule.until_fanout_settles_from` must not be empty")
        return normalized


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent: AgentKind | str
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    on_failure_restart: list[str] = Field(default_factory=list)
    model: str | None = None
    provider: str | ProviderConfig | None = None
    tools: ToolAccess = ToolAccess.READ_ONLY
    mcps: list[MCPServerSpec] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    target: TargetSpec = Field(default_factory=LocalTarget)
    capture: CaptureMode = CaptureMode.FINAL
    repo_instructions_mode: RepoInstructionsMode = RepoInstructionsMode.INHERIT
    output_key: str | None = None
    timeout_seconds: int = Field(default=1800, gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    executable: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    description: str | None = None
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    retries: int = Field(default=0, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0.0)
    retry_backoff_max_seconds: float = Field(default=300.0, ge=0.0)
    retry_backoff_strategy: Literal["linear", "exponential"] = "exponential"
    schedule: PeriodicScheduleSpec | None = None
    fanout_group: str | None = Field(default=None, exclude=True)
    fanout_member: dict[str, Any] | None = Field(default=None, exclude=True)
    fanout_dependencies: dict[str, list[str]] = Field(default_factory=dict, exclude=True)

    @field_validator("agent")
    @classmethod
    def validate_agent(cls, value: AgentKind | str) -> AgentKind | str:
        if isinstance(value, AgentKind):
            return value
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("`agent` must not be empty")
        return builtin_agent_kind(normalized) or normalized

    @model_validator(mode="after")
    def ensure_unique_dependencies(self) -> "NodeSpec":
        self.depends_on = list(dict.fromkeys(self.depends_on))
        duplicate_mcp_names = sorted(name for name, count in Counter(mcp.name for mcp in self.mcps).items() if count > 1)
        if duplicate_mcp_names:
            raise ValueError(f"duplicate MCP server names on node {self.id!r}: {duplicate_mcp_names}")
        if self.schedule is not None:
            if self.fanout_group is not None:
                raise ValueError("scheduled nodes cannot also use `fanout`")
            if self.target.kind != "local":
                raise ValueError("scheduled nodes currently require a local target")
        resolve_provider(self.provider, self.agent)
        return self


def _fanout_suffix(index: int, count: int) -> str:
    width = max(1, len(str(count)))
    return str(index).zfill(width)


def _lift_fanout_member_mapping(
    member: dict[str, Any],
    mapping: dict[str, Any],
    *,
    strict: bool = False,
    source: str | None = None,
) -> None:
    for key, item in mapping.items():
        if not isinstance(key, str) or not _FANOUT_ALIAS_PATTERN.fullmatch(key):
            continue
        if key in _FANOUT_MEMBER_RESERVED_NAMES:
            if strict:
                axis_label = f" axis `{source}`" if source else ""
                raise ValueError(
                    f"fanout.matrix{axis_label} item uses reserved lifted key `{key}`; "
                    "choose a different key name"
                )
            continue
        if key in member:
            if strict and member[key] != item:
                axis_label = f" axis `{source}`" if source else ""
                raise ValueError(
                    f"fanout.matrix{axis_label} item conflicts on lifted key `{key}`; "
                    "use distinct field names across axes"
                )
            continue
        member[key] = item


def _expand_fanout_matrix(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    axis_names = list(matrix)
    axis_values = [matrix[axis_name] for axis_name in axis_names]
    members: list[dict[str, Any]] = []
    for combination in product(*axis_values):
        member: dict[str, Any] = {}
        for axis_name, axis_value in zip(axis_names, combination):
            if axis_name in member and member[axis_name] != axis_value:
                raise ValueError(
                    f"fanout.matrix axis `{axis_name}` conflicts with another lifted field; "
                    "rename the axis or the conflicting field"
                )
            member[axis_name] = axis_value
            if isinstance(axis_value, dict):
                _lift_fanout_member_mapping(member, axis_value, strict=True, source=axis_name)
        members.append(member)
    return members


def _normalize_fanout_matrix_member(value: dict[str, Any]) -> dict[str, Any]:
    member = dict(value)
    for key, item in value.items():
        if isinstance(item, dict):
            _lift_fanout_member_mapping(member, item, strict=True, source=key)
    return member


def _fanout_member_matches_selector(member: Any, selector: Any) -> bool:
    if isinstance(selector, dict):
        if not isinstance(member, dict):
            return False
        return all(
            key in member and _fanout_member_matches_selector(member[key], expected)
            for key, expected in selector.items()
        )
    return member == selector


def _curate_fanout_matrix_members(
    matrix: dict[str, list[Any]],
    *,
    include: list[dict[str, Any]] | None = None,
    exclude: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    members = _expand_fanout_matrix(matrix)
    if exclude:
        members = [
            member
            for member in members
            if not any(_fanout_member_matches_selector(member, selector) for selector in exclude)
        ]
    if include:
        members.extend(dict(member) for member in include)
    return members


def _freeze_fanout_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_fanout_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_fanout_value(item) for item in value)
    return value


def _resolve_grouped_fanout_members(
    group_by: FanoutGroupBySpec,
    *,
    source_members: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    members = source_members.get(group_by.from_)
    if members is None:
        raise ValueError(
            f"`fanout.group_by.from` references unknown prior fanout group `{group_by.from_}`; "
            "place the source fanout earlier in the pipeline"
        )

    grouped_members: list[dict[str, Any]] = []
    grouped_indexes: dict[Any, int] = {}
    scoped_metadata_fields = {"source_group", "source_count", "size", "member_ids", "members"}
    source_count = len(members)
    for member in members:
        grouped_member: dict[str, Any] = {}
        for field in group_by.fields:
            if field not in member:
                raise ValueError(
                    f"`fanout.group_by.fields` references `{field}`, but fanout group `{group_by.from_}` "
                    "does not expose that field"
                )
            grouped_member[field] = member[field]

        conflicting_fields = sorted(scoped_metadata_fields.intersection(grouped_member))
        if conflicting_fields:
            joined = ", ".join(f"`{field}`" for field in conflicting_fields)
            raise ValueError(
                f"`fanout.group_by.fields` cannot use reserved scoped reducer metadata fields {joined}"
            )

        frozen = _freeze_fanout_value(grouped_member)
        node_id = member.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(
                f"fanout group `{group_by.from_}` does not expose `node_id`, so `fanout.group_by` "
                "cannot derive scoped reducer dependencies"
            )

        grouped_index = grouped_indexes.get(frozen)
        if grouped_index is None:
            grouped_indexes[frozen] = len(grouped_members)
            grouped_members.append(
                {
                    "source_group": group_by.from_,
                    "source_count": source_count,
                    "size": 1,
                    "member_ids": [node_id],
                    "members": [dict(member)],
                    **grouped_member,
                }
            )
            continue

        grouped_members[grouped_index]["size"] += 1
        grouped_members[grouped_index]["member_ids"].append(node_id)
        grouped_members[grouped_index]["members"].append(dict(member))
    return grouped_members


def _resolve_batched_fanout_members(
    batches: FanoutBatchesSpec,
    *,
    source_members: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    members = source_members.get(batches.from_)
    if members is None:
        raise ValueError(
            f"`fanout.batches.from` references unknown prior fanout group `{batches.from_}`; "
            "place the source fanout earlier in the pipeline"
        )

    batched_members: list[dict[str, Any]] = []
    source_count = len(members)
    for offset in range(0, source_count, batches.size):
        batch_members = [dict(member) for member in members[offset : offset + batches.size]]
        if not batch_members:
            continue

        member_ids: list[str] = []
        for member in batch_members:
            node_id = member.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(
                    f"fanout group `{batches.from_}` does not expose `node_id`, so `fanout.batches` "
                    "cannot derive reducer dependencies"
                )
            member_ids.append(node_id)

        first = batch_members[0]
        last = batch_members[-1]
        batched_members.append(
            {
                "source_group": batches.from_,
                "source_count": source_count,
                "size": len(batch_members),
                "member_ids": member_ids,
                "members": batch_members,
                "start_index": first["index"],
                "end_index": last["index"],
                "start_number": first["number"],
                "end_number": last["number"],
                "start_suffix": first["suffix"],
                "end_suffix": last["suffix"],
            }
        )
    return batched_members


def _fanout_dependency_overrides(member: dict[str, Any]) -> dict[str, list[str]]:
    source_group = member.get("source_group")
    member_ids = member.get("member_ids")
    if not isinstance(source_group, str) or not source_group:
        return {}
    if not isinstance(member_ids, list):
        return {}

    scoped_member_ids = [member_id for member_id in member_ids if isinstance(member_id, str) and member_id]
    if not scoped_member_ids:
        return {}
    return {source_group: scoped_member_ids}


def _fanout_iteration_context(template_id: str, fanout: FanoutSpec, index: int, value: Any) -> dict[str, Any]:
    member_count = fanout.member_count
    suffix = _fanout_suffix(index, member_count)
    member = {
        "index": index,
        "number": index + 1,
        "count": member_count,
        "suffix": suffix,
        "value": value,
        "template_id": template_id,
        "node_id": f"{template_id}_{suffix}",
    }
    if isinstance(value, dict):
        _lift_fanout_member_mapping(member, value)
    context = {fanout.as_: member, "fanout": member}
    for key, raw_value in fanout.derive.items():
        if key in member:
            raise ValueError(
                f"fanout.derive field `{key}` conflicts with an existing member field; choose a different name"
            )
        member[key] = _render_fanout_value(raw_value, context)
    return context


def _render_fanout_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_fanout_string(value, context)
    if isinstance(value, list):
        return [_render_fanout_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_fanout_value(item, context) for key, item in value.items()}
    return value


def _resolve_fanout_template_expression(context: dict[str, Any], expression: str) -> Any:
    current: Any = context
    for part in expression.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise KeyError(expression)
    return current


def _render_fanout_string(template_text: str, context: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        expression = match.group("expr")
        root = expression.split(".", 1)[0]
        if root not in context:
            return match.group(0)
        try:
            resolved = _resolve_fanout_template_expression(context, expression)
        except KeyError:
            return match.group(0)
        return str(resolved)

    return _FANOUT_TEMPLATE_PATTERN.sub(_replace, template_text)


def _resolve_fanout_manifest_modes(raw_fanout: Any) -> Any:
    if not isinstance(raw_fanout, dict):
        return raw_fanout

    updated = dict(raw_fanout)
    selected_modes = [key for key in _FANOUT_EXPANSION_MODE_KEYS if updated.get(key) is not None]
    if len(selected_modes) > 1:
        joined = ", ".join(f"`{key}`" for key in _FANOUT_EXPANSION_MODE_KEYS)
        raise ValueError(f"fanout accepts exactly one of {joined}")

    return updated


def _resolve_fanout_source_modes(raw_fanout: Any, *, source_members: dict[str, list[dict[str, Any]]]) -> Any:
    if not isinstance(raw_fanout, dict):
        return raw_fanout

    updated = dict(raw_fanout)
    raw_group_by = updated.pop("group_by", None)
    raw_batches = updated.pop("batches", None)
    if raw_group_by is not None and raw_batches is not None:
        joined = ", ".join(f"`{key}`" for key in _FANOUT_EXPANSION_MODE_KEYS)
        raise ValueError(f"fanout accepts exactly one of {joined}")

    if raw_group_by is not None:
        group_by = FanoutGroupBySpec.model_validate(raw_group_by)
        updated["values"] = _resolve_grouped_fanout_members(group_by, source_members=source_members)

    if raw_batches is not None:
        batches = FanoutBatchesSpec.model_validate(raw_batches)
        updated["values"] = _resolve_batched_fanout_members(batches, source_members=source_members)
    return updated


def _expand_fanout_node(node: dict[str, Any], fanout: FanoutSpec) -> tuple[list[dict[str, Any]], list[str]]:
    template_id = node.get("id")
    if not isinstance(template_id, str) or not template_id.strip():
        raise ValueError("fanout nodes require a non-empty string `id`")
    if any(marker in template_id for marker in ("{{", "{%", "{#")):
        raise ValueError("fanout node `id` must be a literal group name, not a rendered template")

    node_template = dict(node)
    node_template.pop("fanout", None)
    expanded_nodes: list[dict[str, Any]] = []
    member_ids: list[str] = []
    for index, value in enumerate(fanout.member_values):
        iteration_context = _fanout_iteration_context(template_id, fanout, index, value)
        expanded = _render_fanout_value(node_template, iteration_context)
        if not isinstance(expanded, dict):
            raise ValueError(f"fanout node {template_id!r} did not expand into an object")
        member_id = iteration_context["fanout"]["node_id"]
        expanded["id"] = member_id
        expanded["fanout_group"] = template_id
        expanded["fanout_member"] = dict(iteration_context["fanout"])
        fanout_dependencies = _fanout_dependency_overrides(iteration_context["fanout"])
        if fanout_dependencies:
            expanded["fanout_dependencies"] = fanout_dependencies
        expanded_nodes.append(expanded)
        member_ids.append(member_id)
    return expanded_nodes, member_ids


def _expand_fanout_dependencies(nodes: list[Any], fanouts: dict[str, list[str]]) -> list[Any]:
    expanded_nodes: list[Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            expanded_nodes.append(node)
            continue
        depends_on = node.get("depends_on")
        if not isinstance(depends_on, list):
            expanded_nodes.append(node)
            continue
        updated = dict(node)
        dependency_overrides = updated.get("fanout_dependencies")
        rewritten: list[Any] = []
        for dependency in depends_on:
            if isinstance(dependency, str) and dependency in fanouts:
                if isinstance(dependency_overrides, dict):
                    scoped_members = dependency_overrides.get(dependency)
                    if isinstance(scoped_members, list) and scoped_members:
                        rewritten.extend(scoped_members)
                        continue
                rewritten.extend(fanouts[dependency])
                continue
            rewritten.append(dependency)
        updated["depends_on"] = rewritten
        expanded_nodes.append(updated)
    return expanded_nodes


def expand_compact_nodes(payload: dict[str, Any], *, base_dir: str | Path | None = None) -> dict[str, Any]:
    resolved = dict(payload)
    nodes = resolved.get("nodes")
    if not isinstance(nodes, list):
        return resolved
    source_ids = [node.get("id") for node in nodes if isinstance(node, dict) and isinstance(node.get("id"), str)]
    duplicate_source_ids = {node_id for node_id, count in Counter(source_ids).items() if count > 1}
    if duplicate_source_ids:
        raise ValueError(f"duplicate node ids: {sorted(duplicate_source_ids)}")

    fanouts: dict[str, list[str]] = {}
    raw_fanouts = resolved.get("fanouts")
    if isinstance(raw_fanouts, dict):
        fanouts = {
            str(group_id): [str(member_id) for member_id in members]
            for group_id, members in raw_fanouts.items()
            if isinstance(group_id, str) and isinstance(members, list)
        }
    saw_fanout = False
    expanded_nodes: list[Any] = []
    fanout_members: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            expanded_nodes.append(node)
            continue
        raw_fanout = node.get("fanout")
        if raw_fanout is None:
            expanded_nodes.append(dict(node))
            continue
        saw_fanout = True
        resolved_fanout = _resolve_fanout_manifest_modes(raw_fanout)
        resolved_fanout = _resolve_fanout_source_modes(resolved_fanout, source_members=fanout_members)
        fanout = FanoutSpec.model_validate(resolved_fanout)
        rendered_nodes, member_ids = _expand_fanout_node(node, fanout)
        fanouts[str(node.get("id"))] = member_ids
        fanout_members[str(node.get("id"))] = [
            dict(rendered_node["fanout_member"])
            for rendered_node in rendered_nodes
            if isinstance(rendered_node, dict) and isinstance(rendered_node.get("fanout_member"), dict)
        ]
        expanded_nodes.extend(rendered_nodes)

    if not saw_fanout:
        return resolved

    resolved["fanouts"] = fanouts
    resolved["nodes"] = _expand_fanout_dependencies(expanded_nodes, fanouts)
    return resolved


def _local_target_defaults_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, LocalTarget):
        payload = value.model_dump(mode="python")
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        return None
    payload.setdefault("kind", "local")
    return payload


def _node_default_payload(
    value: Any,
    *,
    subject: str,
    allow_agent: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"`{subject}` must be an object")

    allowed = set(NodeSpec.model_fields) - _NODE_DEFAULT_FORBIDDEN_FIELDS
    if not allow_agent:
        allowed.discard("agent")

    unknown = sorted(set(value) - allowed)
    if unknown:
        supported = ", ".join(f"`{field}`" for field in sorted(allowed))
        unknown_display = ", ".join(f"`{field}`" for field in unknown)
        raise ValueError(f"`{subject}` does not support {unknown_display}; supported fields: {supported}")

    return dict(value)


def _merge_default_target_payload(default_value: Any, override_value: Any) -> Any:
    if not isinstance(default_value, dict) or not isinstance(override_value, dict):
        return deepcopy(override_value)

    default_kind = default_value.get("kind")
    override_kind = override_value.get("kind")
    if default_kind and override_kind and default_kind != override_kind:
        return deepcopy(override_value)

    merged = deepcopy(default_value)
    merged.update(deepcopy(override_value))
    return merged


def _merge_node_payloads(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if key == "target":
            merged[key] = _merge_default_target_payload(merged.get(key), value)
            continue
        if (
            key in _NODE_DEFAULT_LIST_MERGE_FIELDS
            and isinstance(merged.get(key), list)
            and isinstance(value, list)
        ):
            merged[key] = [*deepcopy(merged[key]), *deepcopy(value)]
            continue
        if (
            key in _NODE_DEFAULT_DICT_MERGE_FIELDS
            and isinstance(merged.get(key), dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**deepcopy(merged[key]), **deepcopy(value)}
            continue
        merged[key] = deepcopy(value)
    return merged


def apply_node_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    node_defaults = _node_default_payload(
        resolved.get("node_defaults"),
        subject="node_defaults",
        allow_agent=True,
    )
    raw_agent_defaults = resolved.get("agent_defaults")
    if raw_agent_defaults is None:
        agent_defaults: dict[AgentKind, dict[str, Any]] = {}
    else:
        if not isinstance(raw_agent_defaults, dict):
            raise ValueError("`agent_defaults` must be an object keyed by agent name")
        agent_defaults = {}
        for raw_agent, defaults in raw_agent_defaults.items():
            try:
                agent = raw_agent if isinstance(raw_agent, AgentKind) else AgentKind(str(raw_agent).strip())
            except ValueError as exc:
                supported = ", ".join(f"`{agent.value}`" for agent in AgentKind)
                raise ValueError(f"`agent_defaults` has unknown agent `{raw_agent}`; supported keys: {supported}") from exc
            normalized = _node_default_payload(
                defaults,
                subject=f"agent_defaults.{agent.value}",
                allow_agent=False,
            )
            if normalized is not None:
                agent_defaults[agent] = normalized

    if node_defaults is None and not agent_defaults:
        return resolved

    nodes = resolved.get("nodes")
    if not isinstance(nodes, list):
        return resolved

    merged_nodes: list[Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            merged_nodes.append(node)
            continue

        merged_node = deepcopy(node_defaults or {})
        raw_agent = node.get("agent", merged_node.get("agent"))
        if raw_agent is not None:
            agent = builtin_agent_kind(raw_agent)
            if agent is not None:
                merged_node = _merge_node_payloads(merged_node, agent_defaults.get(agent, {}))
        merged_nodes.append(_merge_node_payloads(merged_node, dict(node)))

    resolved["nodes"] = merged_nodes
    if node_defaults is not None:
        resolved["node_defaults"] = node_defaults
    if agent_defaults:
        resolved["agent_defaults"] = {agent.value: defaults for agent, defaults in agent_defaults.items()}
    return resolved


def _target_disables_inherited_bootstrap(target_payload: dict[str, Any]) -> bool:
    if "bootstrap" not in target_payload:
        return False
    return _normalize_local_bootstrap(target_payload.get("bootstrap")) is None


def _drop_inherited_bootstrap_defaults(local_target_defaults: dict[str, Any]) -> dict[str, Any]:
    inherited = dict(local_target_defaults)
    bootstrap = _normalize_local_bootstrap(inherited.get("bootstrap"))
    if bootstrap is None:
        return inherited

    inherited.pop("bootstrap", None)
    for key in _LOCAL_BOOTSTRAP_TARGET_KEYS:
        inherited.pop(key, None)
    return inherited


def apply_local_target_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    local_target_defaults = _local_target_defaults_payload(resolved.get("local_target_defaults"))

    nodes = resolved.get("nodes")
    if not isinstance(nodes, list):
        return resolved

    merged_nodes: list[Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            merged_nodes.append(node)
            continue

        updated_node = dict(node)
        target = updated_node.get("target")
        if target is None:
            if local_target_defaults is None:
                merged_nodes.append(updated_node)
                continue
            updated_node["target"] = dict(local_target_defaults)
            merged_nodes.append(updated_node)
            continue

        target_payload = _local_target_defaults_payload(target)
        if target_payload is None:
            merged_nodes.append(updated_node)
            continue
        if local_target_defaults is None:
            updated_node["target"] = target_payload
            merged_nodes.append(updated_node)
            continue

        if target_payload.get("kind", local_target_defaults.get("kind", "local")) != "local":
            merged_nodes.append(updated_node)
            continue

        merged_target = (
            _drop_inherited_bootstrap_defaults(local_target_defaults)
            if _target_disables_inherited_bootstrap(target_payload)
            else dict(local_target_defaults)
        )
        merged_target.update(target_payload)
        updated_node["target"] = merged_target
        merged_nodes.append(updated_node)

    resolved["nodes"] = merged_nodes
    return resolved


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    working_dir: str = "."
    optimizer: str | None = None
    n_run: int = Field(default=1, ge=1)
    concurrency: int = Field(default=4, ge=1)
    fail_fast: bool = False
    max_iterations: int = Field(default=10, ge=1)
    scratchboard: bool = False
    use_worktree: bool = False
    node_defaults: dict[str, Any] | None = None
    agent_defaults: dict[AgentKind, dict[str, Any]] = Field(default_factory=dict)
    local_target_defaults: LocalTarget | None = None
    inference: InferenceSetupSpec | None = None
    fanouts: dict[str, list[str]] = Field(default_factory=dict)
    nodes: list[NodeSpec]

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        base_dir = payload.pop("base_dir", None)
        expanded = expand_compact_nodes(payload, base_dir=base_dir)
        expanded = apply_node_defaults(expanded)
        return apply_local_target_defaults(expanded)

    @model_validator(mode="after")
    def validate_nodes(self) -> "PipelineSpec":
        if self.optimizer is not None:
            normalized_optimizer = self.optimizer.strip()
            if not normalized_optimizer:
                raise ValueError("`optimizer` must not be empty")
            optimizer_kind = builtin_agent_kind(normalized_optimizer)
            if optimizer_kind is None or optimizer_kind not in _INTERACTIVE_AGENT_KINDS:
                supported = ", ".join(f"`{agent.value}`" for agent in sorted(_INTERACTIVE_AGENT_KINDS, key=lambda agent: agent.value))
                raise ValueError(f"`optimizer` must be one of {supported}")
            self.optimizer = normalized_optimizer
        elif self.n_run > 1:
            raise ValueError("`optimizer` is required when `n_run` is greater than 1")

        if not self.nodes:
            raise ValueError("pipeline must contain at least one node")

        ids = [node.id for node in self.nodes]
        duplicates = {node_id for node_id in ids if ids.count(node_id) > 1}
        if duplicates:
            raise ValueError(f"duplicate node ids: {sorted(duplicates)}")
        missing = {
            dependency
            for node in self.nodes
            for dependency in node.depends_on
            if dependency not in ids
        }
        if missing:
            raise ValueError(f"unknown dependencies: {sorted(missing)}")
        fanout_missing = {
            member_id
            for members in self.fanouts.values()
            for member_id in members
            if member_id not in ids
        }
        if fanout_missing:
            raise ValueError(f"fanout metadata references unknown nodes: {sorted(fanout_missing)}")
        node_indexes = {node.id: index for index, node in enumerate(self.nodes)}
        fanout_indexes = {
            group_id: max(node_indexes[member_id] for member_id in member_ids)
            for group_id, member_ids in self.fanouts.items()
            if member_ids
        }
        for node in self.nodes:
            if node.schedule is None:
                continue
            watched_group = node.schedule.until_fanout_settles_from
            if watched_group not in self.fanouts:
                available = ", ".join(f"`{group_id}`" for group_id in sorted(self.fanouts)) or "(none)"
                raise ValueError(
                    f"scheduled node {node.id!r} watches unknown fanout group `{watched_group}`; available fanouts: {available}"
                )
            if fanout_indexes[watched_group] >= node_indexes[node.id]:
                raise ValueError(
                    f"scheduled node {node.id!r} must appear after the watched fanout group `{watched_group}`"
                )
        return self

    @property
    def node_map(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def working_path(self) -> Path:
        return Path(self.working_dir).expanduser().resolve()

    @property
    def uses_graph_optimizer(self) -> bool:
        return self.optimizer is not None and self.n_run > 1


class NormalizedTraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    node_id: str
    agent: AgentKind
    attempt: int = 1
    source: Literal["stdout", "stderr", "system"] = "stdout"
    kind: str
    title: str
    content: str | None = None
    raw: Any | None = None


class NodeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    status: NodeStatus = NodeStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    final_response: str | None = None
    output: str | None = None
    success: bool | None = None
    success_details: list[str] = Field(default_factory=list)


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    final_response: str | None = None
    output: str | None = None
    stdout_lines: list[str] = Field(default_factory=list)
    stderr_lines: list[str] = Field(default_factory=list)
    trace_events: list[NormalizedTraceEvent] = Field(default_factory=list)
    success: bool | None = None
    success_details: list[str] = Field(default_factory=list)
    current_attempt: int = 0
    attempts: list[NodeAttempt] = Field(default_factory=list)
    tick_count: int = 0
    last_tick_started_at: str | None = None
    next_scheduled_at: str | None = None
    diff: str | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: RunStatus = RunStatus.QUEUED
    pipeline: PipelineSpec
    optimization_parent_run_id: str | None = None
    optimization_round: int | None = Field(default=None, ge=1)
    optimization_session: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    nodes: dict[str, NodeResult] = Field(default_factory=dict)


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str
    type: str
    node_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
