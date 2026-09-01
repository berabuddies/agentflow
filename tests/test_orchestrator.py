from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

import agentflow.inference
from agentflow import Graph, codex
from agentflow.agents.base import AgentAdapter
from agentflow.agents.registry import AdapterRegistry
from agentflow.inference import SkyInferenceService
from agentflow.orchestrator import Orchestrator
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.runners.registry import RunnerRegistry
from agentflow.specs import AgentKind, PipelineSpec
from agentflow.store import RunStore


class MockAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        script = r'''
import json
import sys
import sys
import time
from pathlib import Path

node_id = sys.argv[1]
prompt = sys.argv[2]
agent = sys.argv[3]
workdir = Path.cwd()
if node_id in {"alpha", "beta"}:
    time.sleep(0.25)
if node_id == "slow":
    for _ in range(200):
        time.sleep(0.05)
if node_id == "flaky":
    marker = workdir / ".flaky"
    if not marker.exists():
        marker.write_text("first failure", encoding="utf-8")
        print("transient failure", file=sys.stderr)
        raise SystemExit(3)
if node_id == "flaky_silent":
    marker = workdir / ".flaky_silent"
    if not marker.exists():
        marker.write_text("first failure", encoding="utf-8")
        print("stale stdout from failed attempt")
        raise SystemExit(3)
    raise SystemExit(0)
if node_id == "fail_always":
    raise SystemExit(3)
if node_id == "writer":
    (workdir / "artifact.txt").write_text("file data", encoding="utf-8")
if agent == "codex":
    print(json.dumps({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": prompt}]}}))
elif agent == "claude":
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": prompt}]}}))
    print(json.dumps({"type": "result", "result": prompt}))
else:
    print(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"type": "ContentPart", "payload": {"type": "text", "text": prompt}}}))
'''
        return PreparedExecution(
            command=["python3", "-c", script, node.id, prompt, node.agent.value],
            env={},
            cwd=paths.target_workdir,
            trace_kind=node.agent.value,
        )


class ProviderEchoAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        payload = {
            "provider": node.provider.model_dump(mode="json") if hasattr(node.provider, "model_dump") else node.provider,
            "model": node.model,
        }
        script = (
            "import json, sys\n"
            "text = sys.argv[1]\n"
            "print(json.dumps({'type': 'message_end', 'message': {"
            "'role': 'assistant', "
            "'content': [{'type': 'text', 'text': text}]"
            "}}))\n"
        )
        return PreparedExecution(
            command=["python3", "-c", script, json.dumps(payload)],
            env={},
            cwd=paths.target_workdir,
            trace_kind="pi",
        )


class LaunchPlanAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        return PreparedExecution(
            command=["python3", "-c", 'print("launch plan ok")'],
            env={
                "OPENAI_API_KEY": "super-secret",
                "UPSTREAM_AUTH_HEADER": "Bearer top-secret",
                "ANTHROPIC_CUSTOM_HEADERS": '{"x-api-key": "super-secret"}',
                "VISIBLE_FLAG": "visible",
            },
            cwd=paths.target_workdir,
            trace_kind=node.agent.value,
            runtime_files={"config/runtime.env": "OPENAI_API_KEY=super-secret\n"},
        )


class SlowPeriodicAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        script = r'''
import json
import sys
import time

node_id = sys.argv[1]
prompt = sys.argv[2]

if node_id.startswith("worker"):
    time.sleep(1.6)
    text = f"{node_id} finished"
else:
    text = prompt

print(json.dumps({
    "type": "response.output_item.done",
    "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
}))
'''
        return PreparedExecution(
            command=["python3", "-c", script, node.id, prompt],
            env={},
            cwd=paths.target_workdir,
            trace_kind=node.agent.value,
        )


class PeriodicControlAdapter(AgentAdapter):
    def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        script = r'''
import json
import re
import sys
import time
from pathlib import Path

node_id = sys.argv[1]
prompt = sys.argv[2]
workdir = Path.cwd()

if node_id.startswith("worker"):
    marker = workdir / f".{node_id}.rerun"
    if marker.exists():
        text = "healthy"
        print(json.dumps({
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
        }))
        raise SystemExit(0)

    marker.write_text("rerun", encoding="utf-8")
    text = "underperforming"
    print(json.dumps({
        "type": "response.output_item.done",
        "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    }))
    sys.stdout.flush()
    for _ in range(80):
        time.sleep(0.05)
    raise SystemExit(0)

match = re.search(r"(/[^\s]+stdout\.log)", prompt)
stdout_log = Path(match.group(1)) if match else None
payload = {"analysis": "stable", "actions": []}
if stdout_log is not None and stdout_log.exists():
    content = stdout_log.read_text(encoding="utf-8")
    if "underperforming" in content and "healthy" not in content:
        payload = {
            "analysis": "cancel and rerun",
            "actions": [
                {"kind": "cancel", "node_ids": ["worker_0"]},
                {"kind": "rerun", "node_ids": ["worker_0"]},
            ],
        }

print(json.dumps({
    "type": "response.output_item.done",
    "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(payload)}]}
}))
'''
        return PreparedExecution(
            command=["python3", "-c", script, node.id, prompt],
            env={},
            cwd=paths.target_workdir,
            trace_kind=node.agent.value,
        )


def make_orchestrator(tmp_path: Path) -> Orchestrator:
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, MockAdapter())
    adapters.register(AgentKind.CLAUDE, MockAdapter())
    adapters.register(AgentKind.KIMI, MockAdapter())
    return Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())


class BlockingTerminalPersistRunStore(RunStore):
    def __init__(self, base_dir: str | Path) -> None:
        super().__init__(base_dir)
        self.terminal_persist_started = threading.Event()
        self.allow_terminal_persist = threading.Event()

    async def persist_run(self, run_id: str) -> None:
        record = self._runs[run_id]
        if record.status in {
            "completed",
            "failed",
            "cancelled",
        } or getattr(record.status, "value", None) in {"completed", "failed", "cancelled"}:
            self.terminal_persist_started.set()
            while not self.allow_terminal_persist.is_set():
                await asyncio.sleep(0.01)
        await super().persist_run(run_id)


@pytest.mark.asyncio
async def test_orchestrator_runs_parallel_and_templates_outputs(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "parallel",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {"id": "alpha", "agent": "codex", "prompt": "alpha"},
                {"id": "beta", "agent": "claude", "prompt": "beta"},
                {
                    "id": "gamma",
                    "agent": "kimi",
                    "depends_on": ["alpha", "beta"],
                    "prompt": "merge {{ nodes.alpha.output }} + {{ nodes.beta.output }}",
                },
            ],
        }
    )
    started = asyncio.get_running_loop().time()
    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    elapsed = asyncio.get_running_loop().time() - started

    alpha = completed.nodes["alpha"]
    beta = completed.nodes["beta"]
    gamma = completed.nodes["gamma"]
    assert completed.status.value == "completed"
    assert alpha.output == "alpha"
    assert beta.output == "beta"
    assert "alpha" in gamma.output
    assert "beta" in gamma.output
    alpha_start = datetime.fromisoformat(alpha.started_at)
    beta_start = datetime.fromisoformat(beta.started_at)
    assert abs((alpha_start - beta_start).total_seconds()) < 0.15
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_graph_codex_goal_mode_answers_from_goal_thread(tmp_path: Path):
    marker = tmp_path / "goal-set"
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(
        r'''#!/usr/bin/env bash

case "${1:-}:${2:-}" in
app-server:--listen)
    read -r _
    echo '{"id":1,"result":{}}'
    read -r _
    echo '{"id":2,"result":{"thread":{"id":"thread-123"}}}'
    read -r _
    touch "$FAKE_CODEX_GOAL_MARKER"
    echo '{"id":3,"result":{}}'
    ;;
exec:resume)
    if [[ -f "$FAKE_CODEX_GOAL_MARKER" && "${@: -1}" == "am i in /goal ? yes or no" ]]; then
        echo '{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"yes"}]}}'
    else
        echo '{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"no"}]}}'
    fi
    ;;
*)
    exit 2
    ;;
esac
''',
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=AdapterRegistry(), runners=RunnerRegistry())

    with Graph("goal-mode", working_dir=str(tmp_path)) as graph:
        codex(
            task_id="ask",
            prompt="am i in /goal ? yes or no",
            goal=True,
            executable=str(fake_codex),
            env={"FAKE_CODEX_GOAL_MARKER": str(marker)},
        )

    run = await orchestrator.submit(graph.to_spec())
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert "yes" in (completed.nodes["ask"].output or "").lower()


@pytest.mark.asyncio
async def test_orchestrator_launches_graph_inference_and_injects_pi_provider(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def _launch(request):
        captured["request"] = request
        return SkyInferenceService(
            name="agentflow-inference-qwen",
            cluster_name="agentflow-infer-qwen",
            base_url="https://inference.example/v1",
            api_key=request.api_key or "generated-key",
            model_id=request.model_id,
            engine=request.engine,
            gpu=request.gpu,
            port=request.port,
            use_spot=request.use_spot,
            request_id="request-1",
            provider={
                "name": "agentflow-inference-qwen",
                "base_url": "https://inference.example/v1",
                "api_key_env": "OPENAI_API_KEY",
                "wire_api": "openai-completions",
                "env": {
                    "OPENAI_API_KEY": request.api_key or "generated-key",
                    "OPENAI_BASE_URL": "https://inference.example/v1",
                },
            },
        )

    monkeypatch.setattr(agentflow.inference, "launch_sky_inference_service", _launch)
    adapters = AdapterRegistry()
    adapters.register(AgentKind.PI, ProviderEchoAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-inference",
            "working_dir": str(tmp_path),
            "inference": {
                "gpu": "aws:8x8xb200@us-east-2",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "engine": "sglang",
                "api_key": "test-key",
            },
            "nodes": [
                {"id": "solve", "agent": "pi", "prompt": "solve"},
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    request = captured["request"]
    assert request.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert request.gpu.num_nodes == 8
    assert request.gpu.accelerators == "B200:8"
    assert request.engine == "sglang"
    assert request.use_spot is True
    assert request.api_key == "test-key"
    assert completed.status.value == "completed"
    output = json.loads(completed.nodes["solve"].output)
    assert output["provider"]["name"] == "agentflow-inference-qwen"
    assert output["provider"]["base_url"] == "https://inference.example/v1"
    assert output["provider"]["env"]["OPENAI_API_KEY"] == "test-key"
    assert output["model"] == "agentflow-inference-qwen/Qwen/Qwen2.5-0.5B-Instruct"
    assert completed.pipeline.node_map["solve"].provider.base_url == "https://inference.example/v1"


@pytest.mark.asyncio
async def test_orchestrator_renders_fanout_group_context_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout",
            "working_dir": str(tmp_path),
            "concurrency": 3,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "count": 3,
                        "as": "shard",
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.number }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "fanout={{ fanouts.worker.size }} :: "
                        "{% for shard in fanouts.worker.nodes %}"
                        "{{ shard.id }}={{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "worker_2", "merge"}
    assert completed.nodes["worker_0"].output == "worker 1"
    assert completed.nodes["worker_1"].output == "worker 2"
    assert completed.nodes["worker_2"].output == "worker 3"
    assert completed.nodes["merge"].output == (
        "fanout=3 :: worker_0=worker 1;worker_1=worker 2;worker_2=worker 3;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_fanout_values_context_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-values",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "as": "shard",
                        "values": [
                            {"target": "libpng", "seed": 1001},
                            {"target": "sqlite", "seed": 2002},
                        ],
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.target }} seed {{ shard.seed }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "fanout={{ fanouts.worker.size }} :: "
                        "{% for shard in fanouts.worker.nodes %}"
                        "{{ shard.id }}={{ shard.target }}:{{ shard.seed }}:{{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "merge"}
    assert completed.nodes["worker_0"].output == "worker libpng seed 1001"
    assert completed.nodes["worker_1"].output == "worker sqlite seed 2002"
    assert completed.nodes["merge"].output == (
        "fanout=2 :: worker_0=libpng:1001:worker libpng seed 1001;"
        "worker_1=sqlite:2002:worker sqlite seed 2002;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_curated_fanout_matrix_context_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-matrix-curated",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [{"target": "libpng"}, {"target": "sqlite"}],
                            "variant": [{"sanitizer": "asan"}, {"sanitizer": "ubsan"}],
                        },
                        "exclude": [{"target": "sqlite", "sanitizer": "ubsan"}],
                        "include": [
                            {
                                "family": {"target": "openssl"},
                                "variant": {"sanitizer": "asan"},
                            }
                        ],
                        "derive": {
                            "label": "{{ shard.target }}/{{ shard.sanitizer }}",
                        },
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.label }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "fanout={{ fanouts.worker.size }} :: "
                        "{% for shard in fanouts.worker.nodes %}"
                        "{{ shard.id }}={{ shard.target }}:{{ shard.sanitizer }}:{{ shard.label }}:{{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "worker_2", "worker_3", "merge"}
    assert completed.nodes["worker_0"].output == "worker libpng/asan"
    assert completed.nodes["worker_1"].output == "worker libpng/ubsan"
    assert completed.nodes["worker_2"].output == "worker sqlite/asan"
    assert completed.nodes["worker_3"].output == "worker openssl/asan"
    assert completed.nodes["merge"].output == (
        "fanout=4 :: worker_0=libpng:asan:libpng/asan:worker libpng/asan;"
        "worker_1=libpng:ubsan:libpng/ubsan:worker libpng/ubsan;"
        "worker_2=sqlite:asan:sqlite/asan:worker sqlite/asan;"
        "worker_3=openssl:asan:openssl/asan:worker openssl/asan;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_fanout_values_context_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-values",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "as": "shard",
                        "values": [
                            {"target": "libpng", "seed": 1001},
                            {"target": "sqlite", "seed": 2002},
                        ],
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.target }} seed {{ shard.seed }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "fanout={{ fanouts.worker.size }} :: "
                        "{% for shard in fanouts.worker.nodes %}"
                        "{{ shard.id }}={{ shard.target }}:{{ shard.seed }}:{{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "merge"}
    assert completed.nodes["worker_0"].output == "worker libpng seed 1001"
    assert completed.nodes["worker_1"].output == "worker sqlite seed 2002"
    assert completed.nodes["merge"].output == (
        "fanout=2 :: worker_0=libpng:1001:worker libpng seed 1001;"
        "worker_1=sqlite:2002:worker sqlite seed 2002;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_fanout_matrix_context_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-matrix",
            "working_dir": str(tmp_path),
            "concurrency": 4,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [
                                {"target": "libpng", "corpus": "png"},
                                {"target": "sqlite", "corpus": "sql"},
                            ],
                            "variant": [
                                {"sanitizer": "asan", "seed": 1001},
                                {"sanitizer": "ubsan", "seed": 2002},
                            ],
                        },
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.target }} {{ shard.sanitizer }} seed {{ shard.seed }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "fanout={{ fanouts.worker.size }} :: "
                        "{% for shard in fanouts.worker.nodes %}"
                        "{{ shard.id }}={{ shard.family.target }}/{{ shard.variant.seed }}/{{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "worker_2", "worker_3", "merge"}
    assert completed.nodes["worker_0"].output == "worker libpng asan seed 1001"
    assert completed.nodes["worker_1"].output == "worker libpng ubsan seed 2002"
    assert completed.nodes["worker_2"].output == "worker sqlite asan seed 1001"
    assert completed.nodes["worker_3"].output == "worker sqlite ubsan seed 2002"
    assert completed.nodes["merge"].output == (
        "fanout=4 :: worker_0=libpng/1001/worker libpng asan seed 1001;"
        "worker_1=libpng/2002/worker libpng ubsan seed 2002;"
        "worker_2=sqlite/1001/worker sqlite asan seed 1001;"
        "worker_3=sqlite/2002/worker sqlite ubsan seed 2002;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_grouped_fanout_members_in_merge_prompt(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-group-by",
            "working_dir": str(tmp_path),
            "concurrency": 4,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [{"target": "libpng"}, {"target": "sqlite"}],
                            "variant": [{"sanitizer": "asan"}, {"sanitizer": "ubsan"}],
                        },
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.target }} {{ shard.sanitizer }}",
                },
                {
                    "id": "family_merge",
                    "fanout": {
                        "as": "family",
                        "group_by": {
                            "from": "worker",
                            "fields": ["target"],
                        },
                    },
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": "family {{ family.target }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["family_merge"],
                    "prompt": (
                        "groups={{ fanouts.family_merge.size }} :: "
                        "{% for family in fanouts.family_merge.nodes %}"
                        "{{ family.id }}={{ family.target }}:{{ family.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {
        "worker_0",
        "worker_1",
        "worker_2",
        "worker_3",
        "family_merge_0",
        "family_merge_1",
        "merge",
    }
    assert completed.nodes["family_merge_0"].output == "family libpng"
    assert completed.nodes["family_merge_1"].output == "family sqlite"
    assert completed.nodes["merge"].output == (
        "groups=2 :: family_merge_0=libpng:family libpng;"
        "family_merge_1=sqlite:family sqlite;"
    )


@pytest.mark.asyncio
async def test_orchestrator_renders_current_scope_for_batched_reducers(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-batches-scope",
            "working_dir": str(tmp_path),
            "concurrency": 3,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "count": 3,
                        "as": "shard",
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.number }}",
                },
                {
                    "id": "batch_merge",
                    "fanout": {
                        "as": "batch",
                        "batches": {
                            "from": "worker",
                            "size": 2,
                        },
                    },
                    "agent": "codex",
                    "depends_on": ["worker"],
                    "prompt": (
                        "done={{ item.scope.summary.completed }}/{{ item.scope.size }} :: "
                        "{% for shard in item.scope.with_output.nodes %}"
                        "{{ shard.node_id }}={{ shard.output }};"
                        "{% endfor %}"
                    ),
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    assert set(completed.nodes) == {"worker_0", "worker_1", "worker_2", "batch_merge_0", "batch_merge_1"}
    assert completed.nodes["batch_merge_0"].output == "done=2/2 :: worker_0=worker 1;worker_1=worker 2;"
    assert completed.nodes["batch_merge_1"].output == "done=1/1 :: worker_2=worker 3;"


@pytest.mark.asyncio
async def test_orchestrator_reschedules_periodic_nodes_until_watched_fanout_settles(tmp_path: Path):
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, SlowPeriodicAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "periodic-monitor",
            "working_dir": str(tmp_path),
            "concurrency": 3,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "count": 2,
                        "as": "shard",
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.number }}",
                },
                {
                    "id": "monitor",
                    "agent": "codex",
                    "schedule": {
                        "every_seconds": 1,
                        "until_fanout_settles_from": "worker",
                    },
                    "prompt": "tick {{ item.tick_number }} running={{ fanouts.worker.summary.running }}",
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=10)

    monitor = completed.nodes["monitor"]
    assert completed.status.value == "completed"
    assert monitor.status.value == "completed"
    assert monitor.tick_count >= 2
    assert len(monitor.attempts) >= 2
    assert monitor.output is not None and monitor.output.startswith("tick ")
    events = orchestrator.store.get_events(completed.id)
    tick_events = [event for event in events if event.type == "node_tick_completed" and event.node_id == "monitor"]
    assert len(tick_events) >= 2


@pytest.mark.asyncio
async def test_orchestrator_periodic_nodes_can_cancel_and_rerun_watched_members_from_logs(tmp_path: Path):
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, PeriodicControlAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "periodic-control",
            "working_dir": str(tmp_path),
            "concurrency": 2,
            "nodes": [
                {
                    "id": "worker",
                    "fanout": {
                        "count": 1,
                        "as": "shard",
                    },
                    "agent": "codex",
                    "prompt": "worker {{ shard.number }}",
                    "timeout_seconds": 30,
                },
                {
                    "id": "monitor",
                    "agent": "codex",
                    "schedule": {
                        "every_seconds": 1,
                        "until_fanout_settles_from": "worker",
                        "actuation": "output_json",
                    },
                    "prompt": "watch {{ fanouts.worker.nodes[0].artifacts.stdout_log }}",
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=10)

    worker = completed.nodes["worker_0"]
    monitor = completed.nodes["monitor"]

    assert completed.status.value == "completed"
    assert worker.status.value == "completed"
    assert len(worker.attempts) == 2
    assert worker.attempts[0].status.value == "cancelled"
    assert worker.attempts[1].status.value == "completed"
    assert worker.output == "healthy"
    assert monitor.status.value == "completed"
    assert monitor.output == "cancel and rerun"

    action_artifact = json.loads(
        orchestrator.store.read_artifact_text(completed.id, "monitor", "periodic-actions-tick-1.json")
    )
    assert action_artifact["actions"] == [
        {"kind": "cancel", "node_ids": ["worker_0"], "reason": None},
        {"kind": "rerun", "node_ids": ["worker_0"], "reason": None},
    ]
    events = orchestrator.store.get_events(completed.id)
    assert any(event.type == "node_control_actions_applied" and event.node_id == "monitor" for event in events)


@pytest.mark.asyncio
async def test_orchestrator_waits_for_terminal_persist_before_returning(tmp_path: Path):
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, MockAdapter())
    store = BlockingTerminalPersistRunStore(tmp_path / "runs")
    orchestrator = Orchestrator(store=store, adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "persist-before-wait-return",
            "working_dir": str(tmp_path),
            "nodes": [
                {"id": "alpha", "agent": "codex", "prompt": "alpha"},
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    wait_task = asyncio.create_task(orchestrator.wait(run.id, timeout=5))

    await asyncio.to_thread(store.terminal_persist_started.wait, 5)
    await asyncio.sleep(0.1)
    assert wait_task.done() is False

    store.allow_terminal_persist.set()
    completed = await asyncio.wait_for(wait_task, timeout=5)

    assert completed.status.value == "completed"
    reloaded = RunStore(tmp_path / "runs")
    assert reloaded.get_run(run.id).status.value == "completed"


@pytest.mark.asyncio
async def test_orchestrator_applies_success_criteria(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "writer",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "writer",
                    "agent": "codex",
                    "prompt": "success",
                    "success_criteria": [
                        {"kind": "file_exists", "path": "artifact.txt"},
                        {"kind": "file_contains", "path": "artifact.txt", "value": "file data"},
                    ],
                }
            ],
        }
    )
    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    assert completed.nodes["writer"].status.value == "completed"
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "file data"


@pytest.mark.asyncio
async def test_orchestrator_applies_file_success_criteria_in_local_target_cwd(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    pipeline = PipelineSpec.model_validate(
        {
            "name": "writer-target-cwd",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "writer",
                    "agent": "codex",
                    "prompt": "success",
                    "target": {"kind": "local", "cwd": str(task_dir)},
                    "success_criteria": [
                        {"kind": "file_exists", "path": "artifact.txt"},
                        {"kind": "file_contains", "path": "artifact.txt", "value": "file data"},
                    ],
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.nodes["writer"].status.value == "completed"
    assert (task_dir / "artifact.txt").read_text(encoding="utf-8") == "file data"


@pytest.mark.asyncio
async def test_orchestrator_resolves_relative_local_target_cwd_from_pipeline_workdir(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    pipeline = PipelineSpec.model_validate(
        {
            "name": "writer-relative-target-cwd",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "writer",
                    "agent": "codex",
                    "prompt": "success",
                    "target": {"kind": "local", "cwd": "task"},
                    "success_criteria": [
                        {"kind": "file_exists", "path": "artifact.txt"},
                    ],
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.nodes["writer"].status.value == "completed"
    assert (task_dir / "artifact.txt").read_text(encoding="utf-8") == "file data"


@pytest.mark.asyncio
async def test_orchestrator_retries_failed_nodes(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "retry",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "flaky",
                    "agent": "codex",
                    "prompt": "recovered",
                    "retries": 1,
                    "retry_backoff_seconds": 0.01,
                }
            ],
        }
    )
    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    node = completed.nodes["flaky"]
    assert completed.status.value == "completed"
    assert node.status.value == "completed"
    assert node.current_attempt == 2
    assert len(node.attempts) == 2
    assert node.attempts[0].status.value == "failed"
    assert node.attempts[1].status.value == "completed"
    assert orchestrator.store.read_artifact_text(completed.id, "flaky", "output.txt") == "recovered"


@pytest.mark.asyncio
async def test_orchestrator_retry_isolates_final_capture_from_failed_attempt_stdout(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "retry-silent-success",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "flaky_silent",
                    "agent": "codex",
                    "prompt": "unused",
                    "retries": 1,
                    "retry_backoff_seconds": 0.01,
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    node = completed.nodes["flaky_silent"]

    assert completed.status.value == "completed"
    assert node.status.value == "completed"
    assert node.final_response == ""
    assert node.output == ""
    assert node.stdout_lines == []
    assert node.attempts[0].output == "stale stdout from failed attempt"
    assert node.attempts[1].output == ""
    assert orchestrator.store.read_artifact_text(completed.id, "flaky_silent", "output.txt") == ""


@pytest.mark.asyncio
async def test_orchestrator_preserves_per_attempt_launch_artifacts_on_retry(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "retry-launch-artifacts",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "flaky",
                    "agent": "codex",
                    "prompt": "recovered",
                    "retries": 1,
                    "retry_backoff_seconds": 0.01,
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    latest_launch = json.loads(orchestrator.store.read_artifact_text(completed.id, "flaky", "launch.json"))
    first_attempt_launch = json.loads(
        orchestrator.store.read_artifact_text(completed.id, "flaky", "launch-attempt-1.json")
    )
    second_attempt_launch = json.loads(
        orchestrator.store.read_artifact_text(completed.id, "flaky", "launch-attempt-2.json")
    )

    assert completed.status.value == "completed"
    assert latest_launch["attempt"] == 2
    assert first_attempt_launch["attempt"] == 1
    assert second_attempt_launch["attempt"] == 2
    assert latest_launch == second_attempt_launch


@pytest.mark.asyncio
async def test_orchestrator_cancels_running_nodes(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cancel",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "slow",
                    "agent": "codex",
                    "prompt": "eventually cancelled",
                    "timeout_seconds": 30,
                }
            ],
        }
    )
    run = await orchestrator.submit(pipeline)
    for _ in range(50):
        snapshot = orchestrator.store.get_run(run.id)
        if snapshot.status.value == "running":
            break
        await asyncio.sleep(0.05)
    await orchestrator.cancel(run.id)
    completed = await orchestrator.wait(run.id, timeout=5)
    assert completed.status.value == "cancelled"
    assert completed.nodes["slow"].status.value == "cancelled"


@pytest.mark.asyncio
async def test_orchestrator_honors_cancel_request_from_fresh_instance(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cancel-cross-process",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "slow",
                    "agent": "codex",
                    "prompt": "eventually cancelled",
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    await asyncio.sleep(0.2)

    external_store = RunStore(tmp_path / "runs")
    external_orchestrator = Orchestrator(store=external_store, adapters=orchestrator.adapters, runners=RunnerRegistry())
    requested = await external_orchestrator.cancel(run.id)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert requested.status.value == "cancelling"
    assert completed.status.value == "cancelled"
    assert completed.nodes["slow"].status.value == "cancelled"
    assert external_store.cancel_requested(run.id) is False
    assert "Cancelled by user" in orchestrator.store.read_artifact_text(completed.id, "slow", "stderr.log")


@pytest.mark.asyncio
async def test_orchestrator_persists_cancel_stderr_before_first_attempt(tmp_path: Path, monkeypatch):
    original_publish = Orchestrator._publish
    node_started = asyncio.Event()
    release_node = asyncio.Event()

    async def gated_publish(self, run_id: str, event_type: str, *, node_id: str | None = None, **data):
        await original_publish(self, run_id, event_type, node_id=node_id, **data)
        if event_type == "node_started" and node_id == "slow":
            node_started.set()
            await release_node.wait()

    monkeypatch.setattr(Orchestrator, "_publish", gated_publish)

    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cancel-before-attempt",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "slow",
                    "agent": "codex",
                    "prompt": "cancel before attempt",
                    "timeout_seconds": 30,
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    await asyncio.wait_for(node_started.wait(), timeout=5)
    await orchestrator.cancel(run.id)
    release_node.set()

    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "cancelled"
    assert completed.nodes["slow"].status.value == "cancelled"
    assert "Cancelled by user" in orchestrator.store.read_artifact_text(completed.id, "slow", "stderr.log")


@pytest.mark.asyncio
async def test_orchestrator_writes_redacted_launch_artifact(tmp_path: Path):
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, LaunchPlanAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "launch-artifact",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "alpha",
                    "agent": "codex",
                    "prompt": "launch",
                    "target": {"kind": "local", "shell": "bash"},
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    launch_artifact = json.loads(orchestrator.store.read_artifact_text(completed.id, "alpha", "launch.json"))
    assert launch_artifact == {
        "attempt": 1,
        "kind": "process",
        "command": ["bash", "-c", "python3 -c 'print(\"launch plan ok\")'"],
        "env": {
            "ANTHROPIC_CUSTOM_HEADERS": "<redacted>",
            "OPENAI_API_KEY": "<redacted>",
            "UPSTREAM_AUTH_HEADER": "<redacted>",
            "VISIBLE_FLAG": "visible",
        },
        "cwd": str(tmp_path),
        "stdin": None,
        "runtime_files": ["config/runtime.env"],
        "payload": None,
    }


@pytest.mark.asyncio
async def test_orchestrator_redacts_inline_shell_bootstrap_secrets_in_launch_artifact(tmp_path: Path):
    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, LaunchPlanAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "launch-artifact-shell-secret",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "alpha",
                    "agent": "codex",
                    "prompt": "launch",
                    "target": {
                        "kind": "local",
                        "shell": "bash",
                        "shell_init": ["export ANTHROPIC_API_KEY=super-secret-inline"],
                    },
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)

    assert completed.status.value == "completed"
    launch_artifact = json.loads(orchestrator.store.read_artifact_text(completed.id, "alpha", "launch.json"))
    assert launch_artifact["command"] == [
        "bash",
        "-c",
        'export ANTHROPIC_API_KEY=<redacted> && eval "$AGENTFLOW_TARGET_COMMAND"',
    ]


@pytest.mark.asyncio
async def test_orchestrator_does_not_use_kimi_control_events_as_final_output(tmp_path: Path):
    class ControlOnlyKimiAdapter(AgentAdapter):
        def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
            script = r'''
import json
import sys

print(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"type": "TurnBegin", "payload": {"user_input": sys.argv[1]}}}))
print(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"type": "StepBegin", "payload": {"n": 1}}}))
raise SystemExit(1)
'''
            return PreparedExecution(
                command=["python3", "-c", script, prompt],
                env={},
                cwd=paths.target_workdir,
                trace_kind=node.agent.value,
            )

    adapters = AdapterRegistry()
    adapters.register(AgentKind.KIMI, ControlOnlyKimiAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "kimi-control-events",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "review",
                    "agent": "kimi",
                    "prompt": "Reply with exactly: kimi ok",
                    "success_criteria": [{"kind": "output_contains", "value": "kimi ok"}],
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    node = completed.nodes["review"]

    assert completed.status.value == "failed"
    assert node.status.value == "failed"
    assert node.final_response == ""
    assert node.output == ""
    assert node.success is False
    assert node.success_details == ["output_contains('kimi ok')=False"]
    assert node.attempts[0].output == ""


@pytest.mark.asyncio
async def test_orchestrator_filters_ignored_claude_hook_stdout_lines(tmp_path: Path):
    class HookyClaudeAdapter(AgentAdapter):
        def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
            script = r'''
import json

print(json.dumps({"type": "system", "subtype": "hook_started", "hook_name": "SessionStart:startup"}))
print(json.dumps({"type": "system", "subtype": "hook_response", "hook_name": "SessionStart:startup", "output": "very large startup payload"}))
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "claude ok"}]}}))
print(json.dumps({"type": "result", "result": "claude ok"}))
'''
            return PreparedExecution(
                command=["python3", "-c", script],
                env={},
                cwd=paths.target_workdir,
                trace_kind=node.agent.value,
            )

    adapters = AdapterRegistry()
    adapters.register(AgentKind.CLAUDE, HookyClaudeAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "claude-hook-filter",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "review",
                    "agent": "claude",
                    "prompt": "Reply with exactly: claude ok",
                    "capture": "trace",
                    "success_criteria": [{"kind": "output_contains", "value": "claude ok"}],
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    node = completed.nodes["review"]
    stdout_log = orchestrator.store.read_artifact_text(completed.id, "review", "stdout.log")

    assert completed.status.value == "completed"
    assert all("hook_" not in line for line in node.stdout_lines)
    assert node.output is not None and "hook_response" not in node.output
    assert "hook_response" in stdout_log


@pytest.mark.asyncio
async def test_orchestrator_filters_ignored_codex_warning_stdout_lines(tmp_path: Path):
    class WarningCodexAdapter(AgentAdapter):
        def prepare(self, node, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
            script = r'''
import json

print(json.dumps({
    "type": "item.completed",
    "item": {
        "id": "item_0",
        "type": "error",
        "message": "Under-development features enabled: responses_websockets_v2. To suppress this warning, set suppress_unstable_features_warning = true."
    }
}))
print(json.dumps({
    "type": "response.output_item.done",
    "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "codex ok"}]}
}))
'''
            return PreparedExecution(
                command=["python3", "-c", script],
                env={},
                cwd=paths.target_workdir,
                trace_kind=node.agent.value,
            )

    adapters = AdapterRegistry()
    adapters.register(AgentKind.CODEX, WarningCodexAdapter())
    orchestrator = Orchestrator(store=RunStore(tmp_path / "runs"), adapters=adapters, runners=RunnerRegistry())
    pipeline = PipelineSpec.model_validate(
        {
            "name": "codex-warning-filter",
            "working_dir": str(tmp_path),
            "nodes": [
                {
                    "id": "plan",
                    "agent": "codex",
                    "prompt": "Reply with exactly: codex ok",
                    "capture": "trace",
                    "success_criteria": [{"kind": "output_contains", "value": "codex ok"}],
                }
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=5)
    node = completed.nodes["plan"]
    stdout_log = orchestrator.store.read_artifact_text(completed.id, "plan", "stdout.log")

    assert completed.status.value == "completed"
    assert all("Under-development features enabled:" not in line for line in node.stdout_lines)
    assert node.output is not None and "Under-development features enabled:" not in node.output
    assert "Under-development features enabled:" in stdout_log


@pytest.mark.asyncio
async def test_cycle_tail_success_does_not_block_downstream_skip(tmp_path: Path):
    """Regression: once a cycle tail succeeds, downstream nodes that also
    depend on a *separate* failed node must be skipped normally — not held
    pending forever.

    Pipeline:
        cycle_target -> cycle_tail (on_failure_restart -> cycle_target)
        fail_always  (independent, always fails)
        downstream   (depends_on: [cycle_tail, fail_always])

    cycle_tail fails on first iteration (triggering restart), then succeeds
    on iteration 2.  fail_always always fails.  downstream should be SKIPPED
    because fail_always failed — not stuck PENDING because the cycle tail
    was once active.
    """
    orchestrator = make_orchestrator(tmp_path)
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cycle-tail-success-skip",
            "working_dir": str(tmp_path),
            "max_iterations": 5,
            "nodes": [
                {"id": "cycle_target", "agent": "codex", "prompt": "target"},
                {
                    "id": "flaky",
                    "agent": "codex",
                    "prompt": "tail",
                    "depends_on": ["cycle_target"],
                    "on_failure_restart": ["cycle_target"],
                },
                {"id": "fail_always", "agent": "codex", "prompt": "doomed"},
                {
                    "id": "downstream",
                    "agent": "codex",
                    "prompt": "should be skipped",
                    "depends_on": ["flaky", "fail_always"],
                },
            ],
        }
    )

    run = await orchestrator.submit(pipeline)
    completed = await orchestrator.wait(run.id, timeout=10)

    assert completed.nodes["flaky"].status.value == "completed"
    assert completed.nodes["fail_always"].status.value == "failed"
    assert completed.nodes["downstream"].status.value == "skipped"
    assert completed.status.value in {"completed", "failed"}
