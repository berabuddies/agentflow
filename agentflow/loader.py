from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agentflow.specs import PipelineSpec, expand_compact_nodes


def load_pipeline_from_path(path: str | Path) -> PipelineSpec:
    path = Path(path)
    if path.suffix == ".py":
        return _load_pipeline_from_python(path)
    data = path.read_text(encoding="utf-8")
    return load_pipeline_from_text(data, base_dir=path.parent.resolve(), source_path=path)


def _load_pipeline_from_python(path: Path) -> PipelineSpec:
    resolved = path.resolve()
    result = subprocess.run(
        [sys.executable, str(resolved)],
        capture_output=True,
        text=True,
        cwd=str(resolved.parent),
    )
    if result.returncode != 0:
        raise ValueError(f"pipeline script `{path}` failed:\n{result.stderr.strip()}")
    return load_pipeline_from_text(result.stdout, base_dir=path.parent.resolve())


def _parse_structured_text(data: str, *, source_path: str | Path | None = None) -> Any:
    suffix = str(Path(source_path).suffix).lower() if source_path is not None else ""
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(data)
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return yaml.safe_load(data)


def load_pipeline_from_text(
    data: str,
    *,
    base_dir: str | Path | None = None,
    source_path: str | Path | None = None,
) -> PipelineSpec:
    parsed = _parse_structured_text(data, source_path=source_path)
    return load_pipeline_from_data(parsed, base_dir=base_dir)


def load_pipeline_from_data(data: Any, *, base_dir: str | Path | None = None) -> PipelineSpec:
    if isinstance(data, dict) and base_dir is not None:
        resolved_base_dir = _resolve_base_dir(base_dir)
        data = expand_compact_nodes(data, base_dir=resolved_base_dir)
        data = _resolve_file_relative_paths(data, resolved_base_dir)
        data = {**data, "base_dir": str(resolved_base_dir)}
    return PipelineSpec.model_validate(data)


def _resolve_base_dir(base_dir: str | Path) -> Path:
    return Path(base_dir).expanduser().resolve()


def _resolve_file_relative_paths(parsed: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(parsed)
    working_dir_value = resolved.get("working_dir", ".")
    working_dir = Path(working_dir_value).expanduser()
    if not working_dir.is_absolute():
        working_dir = (base_dir / working_dir).resolve()
        resolved["working_dir"] = str(working_dir)
    else:
        working_dir = working_dir.resolve()
        resolved["working_dir"] = str(working_dir)

    def _resolve_target_payload(target: Any) -> Any:
        if not isinstance(target, dict):
            return target

        kind = target.get("kind", "local")
        if kind == "local":
            cwd = target.get("cwd")
            if not isinstance(cwd, str) or not cwd:
                return target
            expanded_cwd = Path(cwd).expanduser()
            updated_target = dict(target)
            if expanded_cwd.is_absolute():
                updated_target["cwd"] = str(expanded_cwd.resolve())
            else:
                updated_target["cwd"] = str((working_dir / expanded_cwd).resolve())
            return updated_target

        if kind not in {"docker", "cloud_hypervisor"}:
            return target

        updated_target = dict(target)
        raw_mounts = updated_target.get("mounts")
        if isinstance(raw_mounts, list):
            resolved_mounts: list[Any] = []
            for raw_mount in raw_mounts:
                if not isinstance(raw_mount, dict):
                    resolved_mounts.append(raw_mount)
                    continue
                mount = dict(raw_mount)
                source = mount.get("source")
                if isinstance(source, str) and source.strip():
                    expanded_source = Path(source.strip()).expanduser()
                    if not expanded_source.is_absolute():
                        expanded_source = working_dir / expanded_source
                    mount["source"] = str(expanded_source.resolve())
                resolved_mounts.append(mount)
            updated_target["mounts"] = resolved_mounts

        if kind == "docker":
            daemon_socket = updated_target.get("docker_daemon_socket")
            if isinstance(daemon_socket, str) and daemon_socket.strip():
                expanded_socket = Path(daemon_socket.strip()).expanduser()
                updated_target["docker_daemon_socket"] = (
                    str(expanded_socket.resolve()) if expanded_socket.is_absolute() else str(expanded_socket)
                )
        else:
            for field_name in ("kernel", "rootfs"):
                raw_path = updated_target.get(field_name)
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                expanded_path = Path(raw_path.strip()).expanduser()
                if not expanded_path.is_absolute():
                    expanded_path = working_dir / expanded_path
                updated_target[field_name] = str(expanded_path.resolve())
        return updated_target

    local_target_defaults = resolved.get("local_target_defaults")
    if local_target_defaults is not None:
        resolved["local_target_defaults"] = _resolve_target_payload(local_target_defaults)

    node_defaults = resolved.get("node_defaults")
    if isinstance(node_defaults, dict):
        updated_node_defaults = dict(node_defaults)
        if "target" in updated_node_defaults:
            updated_node_defaults["target"] = _resolve_target_payload(updated_node_defaults.get("target"))
        resolved["node_defaults"] = updated_node_defaults

    raw_agent_defaults = resolved.get("agent_defaults")
    if isinstance(raw_agent_defaults, dict):
        updated_agent_defaults: dict[str, Any] = {}
        for agent_name, defaults in raw_agent_defaults.items():
            if not isinstance(defaults, dict):
                updated_agent_defaults[agent_name] = defaults
                continue
            updated_defaults = dict(defaults)
            if "target" in updated_defaults:
                updated_defaults["target"] = _resolve_target_payload(updated_defaults.get("target"))
            updated_agent_defaults[agent_name] = updated_defaults
        resolved["agent_defaults"] = updated_agent_defaults

    nodes: list[Any] = []
    for node in resolved.get("nodes", []):
        if not isinstance(node, dict):
            nodes.append(node)
            continue
        updated = dict(node)
        if "target" in updated:
            updated["target"] = _resolve_target_payload(updated.get("target"))
        nodes.append(updated)
    resolved["nodes"] = nodes
    return resolved
