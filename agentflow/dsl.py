"""Python DSL helpers for building AgentFlow pipelines."""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
from types import TracebackType
from typing import Any

from agentflow.specs import AgentKind, LocalTarget, NodeSpec, PipelineSpec, normalize_agent_name


_CURRENT_GRAPH: ContextVar["Graph | None"] = ContextVar("_CURRENT_GRAPH", default=None)


@dataclass(frozen=True, slots=True)
class InferenceSetup:
    """Graph-level OpenAI-compatible inference service launched via SkyPilot."""

    gpu: str
    model: str
    engine: str = "vllm"
    use_spot: bool = True
    max_hourly_cost: float | None = None
    image_id: str | None = None
    name: str | None = None
    cluster_name: str | None = None
    api_key: str | None = None
    port: int = 8000
    idle_minutes_to_autostop: int = 60
    retry_until_up: bool = False
    endpoint_timeout_seconds: int = 600

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gpu": self.gpu,
            "model": self.model,
            "engine": self.engine,
            "use_spot": self.use_spot,
            "port": self.port,
            "idle_minutes_to_autostop": self.idle_minutes_to_autostop,
            "retry_until_up": self.retry_until_up,
            "endpoint_timeout_seconds": self.endpoint_timeout_seconds,
        }
        optional_fields = {
            "max_hourly_cost": self.max_hourly_cost,
            "image_id": self.image_id,
            "name": self.name,
            "cluster_name": self.cluster_name,
            "api_key": self.api_key,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return payload


@dataclass
class _FailureEdge:
    """Proxy returned by ``node.on_failure`` for building back-edges."""

    source: "NodeBuilder"

    def __rshift__(self, target: "NodeBuilder | list[NodeBuilder]") -> "NodeBuilder | list[NodeBuilder]":
        if isinstance(target, list):
            for t in target:
                self.source.kwargs.setdefault("on_failure_restart", []).append(t.id)
            return target
        self.source.kwargs.setdefault("on_failure_restart", []).append(target.id)
        return target


@dataclass
class NodeBuilder:
    dag: "Graph"
    id: str
    agent: AgentKind | str
    prompt: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dag._register(self)

    def __repr__(self) -> str:
        return f"NodeBuilder(id={json.dumps(self.id)}, agent={json.dumps(normalize_agent_name(self.agent))})"

    def __rshift__(self, other: "NodeBuilder | list[NodeBuilder]") -> "NodeBuilder | list[NodeBuilder]":
        if isinstance(other, list):
            for item in other:
                item.depends_on.append(self.id)
            return other
        other.depends_on.append(self.id)
        return other

    def __rrshift__(self, other: list["NodeBuilder"]) -> "NodeBuilder":
        if isinstance(other, list):
            for item in other:
                self.depends_on.append(item.id)
            return self
        raise TypeError(f"unsupported dependency source {type(other)!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "prompt": self.prompt,
            "depends_on": list(self.depends_on),
            **_normalize_node_kwargs(self.kwargs),
        }

    @property
    def on_failure(self) -> _FailureEdge:
        """Return a proxy for ``node.on_failure >> target`` back-edges."""
        return _FailureEdge(source=self)

    def to_spec(self) -> NodeSpec:
        return NodeSpec.model_validate(self.to_payload())


class Graph:
    def __init__(
        self,
        name: str,
        *,
        description: str | None = None,
        working_dir: str = ".",
        optimizer: str | None = None,
        n_run: int = 1,
        concurrency: int = 4,
        fail_fast: bool = False,
        max_iterations: int = 10,
        scratchboard: bool = False,
        use_worktree: bool = False,
        node_defaults: dict[str, Any] | None = None,
        agent_defaults: dict[str | AgentKind, dict[str, Any]] | None = None,
        local_target_defaults: dict[str, Any] | LocalTarget | None = None,
        inference: InferenceSetup | dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.working_dir = working_dir
        self.optimizer = optimizer
        self.n_run = n_run
        self.concurrency = concurrency
        self.fail_fast = fail_fast
        self.max_iterations = max_iterations
        self.scratchboard = scratchboard
        self.use_worktree = use_worktree
        self.node_defaults = node_defaults
        self.agent_defaults = agent_defaults
        self.local_target_defaults = local_target_defaults
        self.inference = inference
        self._nodes: dict[str, NodeBuilder] = {}
        self._token: Token[Graph | None] | None = None

    def __repr__(self) -> str:
        return f"Graph(name={json.dumps(self.name)}, nodes={len(self._nodes)})"

    def __enter__(self) -> "Graph":
        self._token = _CURRENT_GRAPH.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _CURRENT_GRAPH.reset(self._token)

    def _register(self, node: NodeBuilder) -> None:
        if node.id in self._nodes:
            raise ValueError(f"node {node.id!r} already exists")
        self._nodes[node.id] = node

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
        }
        if self.description is not None:
            payload["description"] = self.description
        payload["working_dir"] = self.working_dir
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer
        payload["n_run"] = self.n_run
        payload["concurrency"] = self.concurrency
        payload["fail_fast"] = self.fail_fast
        payload["max_iterations"] = self.max_iterations
        payload["scratchboard"] = self.scratchboard
        payload["use_worktree"] = self.use_worktree
        if self.node_defaults is not None:
            payload["node_defaults"] = _normalize_node_defaults(self.node_defaults)
        if self.agent_defaults:
            payload["agent_defaults"] = _normalize_agent_defaults(self.agent_defaults)
        if self.local_target_defaults is not None:
            payload["local_target_defaults"] = self.local_target_defaults
        if self.inference is not None:
            payload["inference"] = _normalize_inference_setup(self.inference)
        payload["nodes"] = [node.to_payload() for node in self._nodes.values()]
        return payload

    def to_spec(self) -> PipelineSpec:
        return PipelineSpec.model_validate(self.to_payload())

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_payload(), indent=indent)


def _normalize_local_target(value: Any) -> Any:
    if not isinstance(value, dict):
        return deepcopy(value)
    if "kind" in value:
        return deepcopy(value)
    return {"kind": "local", **deepcopy(value)}


def _normalize_node_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "target" not in kwargs:
        return deepcopy(kwargs)

    normalized = deepcopy({key: value for key, value in kwargs.items() if key != "target"})
    normalized["target"] = _normalize_local_target(kwargs["target"])
    return normalized


def _normalize_node_defaults(defaults: dict[str, Any] | None) -> dict[str, Any] | None:
    if defaults is None:
        return None
    return _normalize_node_kwargs(defaults)


def _normalize_inference_setup(value: InferenceSetup | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, InferenceSetup):
        return value.to_payload()
    return deepcopy(value)


def _normalize_agent_defaults(
    defaults: dict[str | AgentKind, dict[str, Any]] | None,
) -> dict[str | AgentKind, dict[str, Any]] | None:
    if defaults is None:
        return None
    return {
        agent: _normalize_node_kwargs(agent_defaults)
        for agent, agent_defaults in defaults.items()
    }


DAG = Graph  # backward compatibility


def _current_graph() -> Graph:
    g = _CURRENT_GRAPH.get()
    if g is None:
        raise RuntimeError("No active Graph context. Use `with Graph(...):`.")
    return g


def _node(agent: AgentKind | str, *, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    return NodeBuilder(dag=_current_graph(), id=task_id, agent=agent, prompt=prompt, kwargs=kwargs)


# ---------------------------------------------------------------------------
# Fanout & merge
# ---------------------------------------------------------------------------


def fanout(
    node: NodeBuilder,
    source: int | list[Any] | dict[str, list[Any]],
    *,
    derive: dict[str, Any] | None = None,
    include: list[dict[str, Any]] | None = None,
    exclude: list[dict[str, Any]] | None = None,
) -> NodeBuilder:
    """Fan a node out into many parallel copies.

    *source* selects the expansion mode:

    - ``int``  -- uniform count  (``fanout(node, 128)``)
    - ``list`` -- explicit values (``fanout(node, [{"repo": "api"}, ...])``)
    - ``dict`` -- cartesian matrix (``fanout(node, {"axis": [...], ...})``)

    Every expanded copy gets an ``item`` template variable with these fields:

    ======== ===== ============================================
    Field    Type  Example
    ======== ===== ============================================
    index    int   0, 1, 2, ...
    number   int   1, 2, 3, ... (1-indexed)
    count    int   total copies
    suffix   str   "0", "01", "001" (zero-padded)
    node_id  str   "fuzzer_001"
    value    Any   the raw iteration value
    *(keys)* Any   dict keys from value are lifted
    *(keys)* Any   keys from *derive* are added
    ======== ===== ============================================

    Use ``{{ item.number }}``, ``{{ item.suffix }}``, etc. in prompts
    and target paths.
    """
    if isinstance(source, int):
        mode: dict[str, Any] = {"count": source}
    elif isinstance(source, list):
        mode = {"values": deepcopy(source)}
    elif isinstance(source, dict):
        mode = {"matrix": deepcopy(source)}
    else:
        raise TypeError(f"fanout source must be int, list, or dict; got {type(source).__name__}")

    if include is not None and not isinstance(source, dict):
        raise TypeError("include is only valid for matrix fanout (dict source)")
    if exclude is not None and not isinstance(source, dict):
        raise TypeError("exclude is only valid for matrix fanout (dict source)")

    payload: dict[str, Any] = {"as": "item", **mode}
    if derive is not None:
        payload["derive"] = deepcopy(derive)
    if include is not None:
        payload["include"] = deepcopy(include)
    if exclude is not None:
        payload["exclude"] = deepcopy(exclude)
    node.kwargs["fanout"] = payload
    return node


def merge(
    node: NodeBuilder,
    source: NodeBuilder,
    *,
    by: list[str] | None = None,
    size: int | None = None,
    derive: dict[str, Any] | None = None,
) -> NodeBuilder:
    """Merge (reduce) the results of a prior fanout group.

    Exactly one of *by* or *size* must be given:

    - ``by=["field", ...]`` -- one reducer per unique field combination
    - ``size=N`` -- one reducer per N-item batch

    The ``item`` template variable has all the same fields as a fanout
    ``item``, plus these reducer-specific fields:

    ============= ====== ============================================
    Field         Type   Description
    ============= ====== ============================================
    source_group  str    task_id of the fanout being reduced
    source_count  int    total members in the source fanout
    member_ids    list   node IDs of the members in this group/batch
    members       list   full member objects
    size          int    members in this group/batch
    ============= ====== ============================================

    With ``by=``, the grouping field values are also on ``item``
    (e.g. ``{{ item.target }}``).

    With ``size=``, batch range fields are added:
    ``start_number``, ``end_number``, ``start_index``, ``end_index``,
    ``start_suffix``, ``end_suffix``.

    At runtime, ``item.scope`` provides the aggregated results of the
    source members in this group/batch:

    ============== ====== ============================================
    Field          Type   Description
    ============== ====== ============================================
    scope.ids      list   member node IDs
    scope.size     int    count
    scope.nodes    list   full member objects with status/output
    scope.outputs  list   output strings
    scope.summary  dict   {total, completed, failed, with_output, ...}
    scope.with_output    subset with non-empty output
    scope.without_output subset with empty output
    ============== ====== ============================================

    Use ``{{ item.scope.with_output.nodes }}`` in Jinja2 loops to
    iterate over completed source members.
    """
    if by is not None and size is not None:
        raise TypeError("specify either by= or size=, not both")

    if by is not None:
        mode: dict[str, Any] = {"group_by": {"from": source.id, "fields": list(by)}}
    elif size is not None:
        mode = {"batches": {"from": source.id, "size": size}}
    else:
        raise TypeError("merge() requires either by= or size=")

    payload: dict[str, Any] = {"as": "item", **mode}
    if derive is not None:
        payload["derive"] = deepcopy(derive)
    node.kwargs["fanout"] = payload
    # Auto-add dependency on source so edges are drawn and scheduling works
    if source.id not in node.depends_on:
        node.depends_on.append(source.id)
    return node


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def codex(*, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    return _node(AgentKind.CODEX, task_id=task_id, prompt=prompt, **kwargs)


def claude(*, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    return _node(AgentKind.CLAUDE, task_id=task_id, prompt=prompt, **kwargs)


def kimi(*, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    return _node(AgentKind.KIMI, task_id=task_id, prompt=prompt, **kwargs)


def pi(*, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    """Run the Pi coding agent (https://pi.dev).

    Supports external providers (anthropic, openai, openai-codex, groq, etc.) and
    local endpoints registered in the user's ``~/.pi/agent/models.json`` (e.g.
    LMStudio, Ollama). The ``model`` kwarg accepts Pi's ``provider/id[:thinking]``
    syntax (e.g. ``"lmstudio/mythos-26b"`` or ``"anthropic/claude-sonnet-4-6:high"``).

    For an inline, ephemeral provider declaration, pass a full ``ProviderConfig``
    via ``provider={"name": ..., "base_url": ..., "api_key_env": ..., "wire_api": ...}``
    and AgentFlow will materialize a scoped ``models.json`` for the run.
    """
    return _node(AgentKind.PI, task_id=task_id, prompt=prompt, **kwargs)


def python_node(*, task_id: str, code: str, **kwargs: Any) -> NodeBuilder:
    """Run Python code directly. The ``code`` is executed as ``python3 -c <code>``."""
    return _node(AgentKind.PYTHON, task_id=task_id, prompt=code, **kwargs)


def shell(*, task_id: str, script: str, **kwargs: Any) -> NodeBuilder:
    """Run a shell script directly. The ``script`` is executed as ``bash -c <script>``."""
    return _node(AgentKind.SHELL, task_id=task_id, prompt=script, **kwargs)


def sync(*, task_id: str, mode: str = "full", **kwargs: Any) -> NodeBuilder:
    """Sync local git repo to a remote target.

    *mode* is ``"repo"`` (.git + stash only) or ``"full"`` (entire directory).
    The node ``target`` must be SSH, EC2, or ECS with a ``remote_workdir``.
    Uses rclone if available, falls back to tar + scp.
    """
    return _node(AgentKind.SYNC, task_id=task_id, prompt=mode, **kwargs)


def agent(name: str, *, task_id: str, prompt: str, **kwargs: Any) -> NodeBuilder:
    normalized = name.strip()
    if not normalized:
        raise ValueError("agent name must not be empty")
    return _node(normalized, task_id=task_id, prompt=prompt, **kwargs)


def evolve(
    nodes: NodeBuilder | list[NodeBuilder],
    *,
    target: str = "codex",
    optimizer: str = "codex",
    tuned_agent: str | None = None,
    task_id: str | None = None,
    **kwargs: Any,
) -> NodeBuilder:
    if isinstance(nodes, NodeBuilder):
        source_nodes = [nodes]
    else:
        source_nodes = list(nodes)
    if not source_nodes:
        raise ValueError("evolve() requires at least one source node")

    normalized_target = target.strip()
    filtered_nodes = [node for node in source_nodes if normalize_agent_name(node.agent) == normalized_target]
    if not filtered_nodes:
        raise ValueError(f"evolve() found no source nodes with target agent `{normalized_target}`")

    profile = (tuned_agent or target).strip()
    if not profile:
        raise ValueError("evolve() requires a non-empty tuned agent profile")

    payload = {
        "profile": profile,
        "target": normalized_target,
        "optimizer": optimizer.strip(),
        "source_nodes": [node.id for node in filtered_nodes],
        "trace_paths": {
            node.id: f"{{{{ nodes.{node.id}.artifacts.trace_jsonl }}}}"
            for node in filtered_nodes
        },
        "workspace_dir": "{{ pipeline.working_dir }}",
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    code = (
        "import json\n"
        "from agentflow.tuned_agents import run_evolution_from_payload\n\n"
        f"payload = json.loads(r'''{payload_json}''')\n"
        "result = run_evolution_from_payload(payload)\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    evolve_task_id = task_id or f"evolve_{profile.replace('-', '_')}"
    node = python_node(task_id=evolve_task_id, code=code, **kwargs)
    for source in filtered_nodes:
        if source.id not in node.depends_on:
            node.depends_on.append(source.id)
    return node
