from pathlib import Path
import subprocess
import sys

import pytest

from agentflow import (
    Graph,
    InferenceSetup,
    claude,
    codex,
    fanout,
    kimi,
    merge,
    pi,
)
from agentflow.loader import load_pipeline_from_text


def _run_example(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "examples" / name)],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return load_pipeline_from_text(completed.stdout, base_dir=repo_root)


def test_fanout_and_merge_build_correct_payloads():
    with Graph("payload-test") as dag:
        # count
        n = fanout(codex(task_id="c", prompt="p"), 3)
        assert n.kwargs["fanout"] == {"count": 3, "as": "item"}

        # count with derive
        n = fanout(
            codex(task_id="cd", prompt="p"),
            3,
            derive={"workspace": "agents/{{ item.suffix }}"},
        )
        assert n.kwargs["fanout"] == {
            "count": 3,
            "as": "item",
            "derive": {"workspace": "agents/{{ item.suffix }}"},
        }

        # values
        n = fanout(
            codex(task_id="v", prompt="p"),
            [{"target": "libpng"}, {"target": "sqlite"}],
        )
        assert n.kwargs["fanout"] == {
            "values": [{"target": "libpng"}, {"target": "sqlite"}],
            "as": "item",
        }

        # matrix with include/exclude
        n = fanout(
            codex(task_id="m", prompt="p"),
            {
                "family": [{"target": "libpng"}],
                "variant": [{"sanitizer": "asan"}],
            },
            include=[{"family": {"target": "openssl"}, "variant": {"sanitizer": "asan"}}],
            exclude=[{"family": {"target": "sqlite"}}],
        )
        assert n.kwargs["fanout"] == {
            "matrix": {
                "family": [{"target": "libpng"}],
                "variant": [{"sanitizer": "asan"}],
            },
            "as": "item",
            "include": [{"family": {"target": "openssl"}, "variant": {"sanitizer": "asan"}}],
            "exclude": [{"family": {"target": "sqlite"}}],
        }

        # merge with size (batches)
        src = codex(task_id="src", prompt="p")
        n = merge(codex(task_id="b", prompt="p"), src, size=4)
        assert n.kwargs["fanout"] == {
            "batches": {"from": "src", "size": 4},
            "as": "item",
        }

        # merge with by (group_by)
        n = merge(
            codex(task_id="g", prompt="p"),
            src,
            by=["target", "corpus"],
            derive={"label": "{{ item.target }} / {{ item.corpus }}"},
        )
        assert n.kwargs["fanout"] == {
            "group_by": {"from": "src", "fields": ["target", "corpus"]},
            "as": "item",
            "derive": {"label": "{{ item.target }} / {{ item.corpus }}"},
        }


def test_fanout_rejects_invalid_source_type():
    with Graph("err-test") as dag:
        with pytest.raises(TypeError, match="int, list, or dict"):
            fanout(codex(task_id="bad", prompt="p"), "not_valid")


def test_fanout_rejects_include_on_non_matrix():
    with Graph("err-test2") as dag:
        with pytest.raises(TypeError, match="include is only valid for matrix"):
            fanout(codex(task_id="bad", prompt="p"), [1, 2], include=[{}])


def test_merge_rejects_both_by_and_size():
    with Graph("err-test3") as dag:
        src = codex(task_id="src", prompt="p")
        with pytest.raises(TypeError, match="either by= or size="):
            merge(codex(task_id="bad", prompt="p"), src, by=["x"], size=2)


def test_merge_rejects_neither_by_nor_size():
    with Graph("err-test4") as dag:
        src = codex(task_id="src", prompt="p")
        with pytest.raises(TypeError, match="requires either by= or size="):
            merge(codex(task_id="bad", prompt="p"), src)


def test_airflow_like_dag_builds_dependencies():
    with Graph("demo", working_dir="/tmp/work", concurrency=2) as dag:
        plan = codex(task_id="plan", prompt="plan")
        implement = claude(task_id="implement", prompt="implement")
        review = kimi(task_id="review", prompt="review")
        final = codex(task_id="merge", prompt="merge")
        plan >> [implement, review]
        [implement, review] >> final

    spec = dag.to_spec()
    nodes = spec.node_map

    assert spec.name == "demo"
    assert spec.working_dir == "/tmp/work"
    assert nodes["implement"].depends_on == ["plan"]
    assert nodes["review"].depends_on == ["plan"]
    assert set(nodes["merge"].depends_on) == {"implement", "review"}


def test_dag_and_node_repr_and_payload_isolation():
    with Graph("demo") as dag:
        plan = codex(task_id="plan", prompt="plan")

    assert repr(plan) == 'NodeBuilder(id="plan", agent="codex")'
    assert repr(dag) == 'Graph(name="demo", nodes=1)'

    payload = plan.to_payload()
    payload["depends_on"].append("other")
    assert plan.depends_on == []


def test_graph_supports_optimizer_and_n_run():
    with Graph("opt-demo", optimizer="codex", n_run=3) as dag:
        codex(task_id="plan", prompt="plan")

    payload = dag.to_payload()
    spec = dag.to_spec()

    assert payload["optimizer"] == "codex"
    assert payload["n_run"] == 3
    assert spec.optimizer == "codex"
    assert spec.n_run == 3


def test_graph_supports_inference_setup():
    with Graph(
        "my-pipeline",
        concurrency=3,
        inference=InferenceSetup(
            gpu="aws:8x8xb200@us-east-2",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            engine="sglang",
            max_hourly_cost=20.0,
        ),
    ) as dag:
        pi(task_id="solve", prompt="solve")

    payload = dag.to_payload()
    spec = dag.to_spec()

    assert payload["concurrency"] == 3
    assert payload["inference"] == {
        "gpu": "aws:8x8xb200@us-east-2",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "engine": "sglang",
        "use_spot": True,
        "port": 8000,
        "idle_minutes_to_autostop": 60,
        "retry_until_up": False,
        "endpoint_timeout_seconds": 600,
        "max_hourly_cost": 20.0,
    }
    assert spec.inference is not None
    assert spec.inference.gpu == "aws:8x8xb200@us-east-2"
    assert spec.inference.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec.inference.use_spot is True


def test_airflow_like_dag_applies_local_target_defaults():
    with Graph(
        "local-defaults",
        local_target_defaults={
            "shell": "bash",
            "shell_login": True,
            "shell_interactive": True,
            "shell_init": ["command -v kimi >/dev/null 2>&1", "kimi"],
        },
    ) as dag:
        codex(task_id="plan", prompt="plan")
        claude(task_id="review", prompt="review", target={"cwd": "review-work"})

    spec = dag.to_spec()

    assert spec.local_target_defaults is not None
    assert spec.nodes[0].target.shell == "bash"
    assert spec.nodes[0].target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]
    assert spec.nodes[1].target.shell == "bash"
    assert spec.nodes[1].target.cwd == "review-work"


def test_airflow_like_dag_supports_pipeline_defaults_and_count_fanout():
    with Graph(
        "fanout-defaults",
        working_dir="/tmp/fanout-work",
        concurrency=16,
        fail_fast=True,
        node_defaults={
            "tools": "read_only",
            "capture": "final",
        },
        agent_defaults={
            "codex": {
                "model": "gpt-5-codex",
                "retries": 1,
                "retry_backoff_seconds": 2,
                "extra_args": ["--search"],
            }
        },
    ) as dag:
        prepare = codex(task_id="prepare", prompt="prepare")
        fuzzer = fanout(
            codex(task_id="fuzzer", prompt="fuzz item {{ item.number }}"),
            4,
        )
        final = codex(task_id="merge", prompt="merge")
        prepare >> fuzzer
        fuzzer >> final

    payload = dag.to_payload()
    spec = dag.to_spec()
    nodes = spec.node_map

    assert payload["fail_fast"] is True
    assert payload["node_defaults"] == {"tools": "read_only", "capture": "final"}
    assert payload["agent_defaults"]["codex"]["model"] == "gpt-5-codex"
    assert payload["nodes"][1]["fanout"] == {"count": 4, "as": "item"}
    assert spec.working_dir == "/tmp/fanout-work"
    assert spec.concurrency == 16
    assert spec.fail_fast is True
    assert spec.node_defaults == {"tools": "read_only", "capture": "final"}
    assert spec.agent_defaults["codex"] == {
        "model": "gpt-5-codex",
        "retries": 1,
        "retry_backoff_seconds": 2,
        "extra_args": ["--search"],
    }
    assert spec.fanouts == {"fuzzer": ["fuzzer_0", "fuzzer_1", "fuzzer_2", "fuzzer_3"]}
    assert nodes["prepare"].model == "gpt-5-codex"
    assert nodes["prepare"].tools == "read_only"
    assert nodes["prepare"].capture == "final"
    assert nodes["prepare"].extra_args == ["--search"]
    assert nodes["fuzzer_0"].model == "gpt-5-codex"
    assert nodes["fuzzer_0"].depends_on == ["prepare"]
    assert nodes["fuzzer_3"].fanout_group == "fuzzer"
    assert nodes["fuzzer_3"].fanout_member["number"] == 4
    assert nodes["merge"].depends_on == ["fuzzer_0", "fuzzer_1", "fuzzer_2", "fuzzer_3"]
    assert nodes["merge"].retries == 1


def test_airflow_like_dag_supports_matrix_and_batch_merge():
    with Graph(
        "batched-fuzz",
        working_dir="/tmp/batched-fuzz",
        concurrency=8,
        node_defaults={
            "agent": "codex",
            "tools": "read_only",
        },
        agent_defaults={
            "codex": {
                "model": "gpt-5-codex",
                "extra_args": ["--search"],
            }
        },
    ) as dag:
        init = codex(task_id="init", prompt="init", tools="read_write")
        fuzzer = fanout(
            codex(
                task_id="fuzzer",
                prompt="fuzz {{ item.target }} {{ item.sanitizer }} inside {{ item.workspace }}",
                target={"cwd": "{{ item.workspace }}"},
            ),
            {
                "family": [
                    {"target": "libpng"},
                    {"target": "sqlite"},
                ],
                "variant": [
                    {"sanitizer": "asan"},
                    {"sanitizer": "ubsan"},
                ],
            },
            derive={"workspace": "agents/{{ item.target }}_{{ item.sanitizer }}_{{ item.suffix }}"},
        )
        batch_merge = merge(
            codex(
                task_id="batch_merge",
                prompt="reduce batch {{ item.number }} covering {{ item.member_ids | join(', ') }}",
            ),
            fuzzer,
            size=2,
        )
        final = codex(task_id="merge", prompt="merge")
        init >> fuzzer
        fuzzer >> batch_merge
        batch_merge >> final

    spec = dag.to_spec()
    nodes = spec.node_map

    assert spec.fanouts == {
        "fuzzer": ["fuzzer_0", "fuzzer_1", "fuzzer_2", "fuzzer_3"],
        "batch_merge": ["batch_merge_0", "batch_merge_1"],
    }
    assert nodes["init"].tools == "read_write"
    assert nodes["init"].model == "gpt-5-codex"
    assert nodes["fuzzer_0"].prompt == "fuzz libpng asan inside agents/libpng_asan_0"
    assert nodes["fuzzer_0"].target.cwd == "agents/libpng_asan_0"
    assert nodes["fuzzer_3"].prompt == "fuzz sqlite ubsan inside agents/sqlite_ubsan_3"
    assert nodes["fuzzer_3"].extra_args == ["--search"]
    assert nodes["batch_merge_0"].depends_on == ["fuzzer_0", "fuzzer_1"]
    assert nodes["batch_merge_0"].fanout_member["member_ids"] == ["fuzzer_0", "fuzzer_1"]
    assert nodes["batch_merge_1"].depends_on == ["fuzzer_2", "fuzzer_3"]
    assert nodes["merge"].depends_on == ["batch_merge_0", "batch_merge_1"]


def test_airflow_like_dag_supports_grouped_merge():
    with Graph(
        "grouped-fuzz",
        working_dir="/tmp/grouped-fuzz",
        concurrency=8,
        node_defaults={
            "agent": "codex",
            "tools": "read_only",
        },
        agent_defaults={
            "codex": {
                "model": "gpt-5-codex",
                "extra_args": ["--search"],
            }
        },
    ) as dag:
        init = codex(task_id="init", prompt="init", tools="read_write")
        fuzzer = fanout(
            codex(
                task_id="fuzzer",
                prompt="fuzz {{ item.target }} {{ item.sanitizer }} inside {{ item.workspace }}",
                target={"cwd": "{{ item.workspace }}"},
            ),
            {
                "family": [
                    {"target": "libpng", "corpus": "png"},
                    {"target": "sqlite", "corpus": "sql"},
                ],
                "variant": [
                    {"sanitizer": "asan"},
                    {"sanitizer": "ubsan"},
                ],
            },
            derive={"workspace": "agents/{{ item.target }}_{{ item.sanitizer }}_{{ item.suffix }}"},
        )
        family_merge = merge(
            codex(
                task_id="family_merge",
                prompt="group {{ item.target }} has {{ item.scope.size }} shards",
            ),
            fuzzer,
            by=["target", "corpus"],
        )
        final = codex(task_id="merge", prompt="merge")
        init >> fuzzer
        fuzzer >> family_merge
        family_merge >> final

    spec = dag.to_spec()
    nodes = spec.node_map

    assert spec.fanouts == {
        "fuzzer": ["fuzzer_0", "fuzzer_1", "fuzzer_2", "fuzzer_3"],
        "family_merge": ["family_merge_0", "family_merge_1"],
    }
    assert nodes["family_merge_0"].depends_on == ["fuzzer_0", "fuzzer_1"]
    assert nodes["family_merge_0"].fanout_member["member_ids"] == ["fuzzer_0", "fuzzer_1"]
    assert nodes["family_merge_0"].fanout_member["target"] == "libpng"
    assert nodes["family_merge_1"].depends_on == ["fuzzer_2", "fuzzer_3"]
    assert nodes["family_merge_1"].fanout_member["target"] == "sqlite"
    assert nodes["merge"].depends_on == ["family_merge_0", "family_merge_1"]


def test_airflow_like_dag_can_render_json():
    with Graph(
        "render-demo",
        description="render helpers",
        working_dir="/tmp/render-demo",
        node_defaults={"tools": "read_only"},
    ) as dag:
        codex(task_id="plan", prompt="line one\nline two")

    rendered_json = dag.to_json()
    spec_from_json = load_pipeline_from_text(rendered_json)

    assert '"description": "render helpers"' in rendered_json
    assert '"working_dir": "/tmp/render-demo"' in rendered_json
    assert spec_from_json.name == "render-demo"
    assert spec_from_json.node_map["plan"].prompt == "line one\nline two"


def test_airflow_like_fuzz_batched_example_emits_valid_pipeline():
    spec = _run_example("airflow_like_fuzz_batched.py")

    assert spec.name == "airflow-like-fuzz-batched-128"
    assert spec.fail_fast is True
    assert spec.concurrency == 32
    assert len(spec.fanouts["fuzzer"]) == 128
    assert spec.fanouts["fuzzer"][:3] == ["fuzzer_000", "fuzzer_001", "fuzzer_002"]
    assert spec.fanouts["fuzzer"][-1] == "fuzzer_127"
    assert len(spec.fanouts["batch_merge"]) == 8
    assert spec.node_map["init"].tools == "read_write"
    assert spec.node_map["init"].success_criteria[0].value == "INIT_OK"
    assert spec.node_map["fuzzer_000"].fanout_member["workspace"] == "agents/agent_000"
    assert spec.node_map["fuzzer_127"].fanout_member["workspace"] == "agents/agent_127"
    assert spec.node_map["fuzzer_000"].target.cwd.endswith("/codex_fuzz_python_128/agents/agent_000")
    assert spec.node_map["fuzzer_127"].target.cwd.endswith("/codex_fuzz_python_128/agents/agent_127")
    assert spec.node_map["batch_merge_0"].depends_on == spec.fanouts["fuzzer"][:16]
    assert spec.node_map["batch_merge_0"].fanout_member["member_ids"] == spec.fanouts["fuzzer"][:16]
    assert spec.node_map["batch_merge_7"].fanout_member["member_ids"] == spec.fanouts["fuzzer"][112:]
    assert spec.node_map["monitor"].schedule is not None
    assert spec.node_map["monitor"].schedule.every_seconds == 600
    assert spec.node_map["monitor"].schedule.until_fanout_settles_from == "fuzzer"
    assert spec.node_map["batch_merge_0"].prompt.startswith(
        "Prepare the maintainer handoff for shard batch 1 of 8."
    )
    assert spec.node_map["merge"].depends_on == [*spec.fanouts["batch_merge"], "monitor"]


def test_airflow_like_fuzz_grouped_example_emits_valid_pipeline():
    spec = _run_example("airflow_like_fuzz_grouped.py")

    assert spec.name == "airflow-like-fuzz-grouped-128"
    assert spec.fail_fast is True
    assert spec.concurrency == 32
    assert len(spec.fanouts["fuzzer"]) == 128
    assert spec.fanouts["fuzzer"][:3] == ["fuzzer_000", "fuzzer_001", "fuzzer_002"]
    assert spec.fanouts["fuzzer"][-1] == "fuzzer_127"
    assert len(spec.fanouts["family_merge"]) == 4
    assert spec.node_map["fuzzer_000"].fanout_member["target"] == "libpng"
    assert spec.node_map["fuzzer_000"].fanout_member["corpus"] == "png"
    assert spec.node_map["fuzzer_000"].fanout_member["sanitizer"] == "asan"
    assert spec.node_map["fuzzer_000"].fanout_member["focus"] == "parser"
    assert spec.node_map["fuzzer_000"].fanout_member["bucket"] == "seed_a"
    assert spec.node_map["fuzzer_000"].fanout_member["label"] == "libpng / asan / parser / seed_a"
    assert spec.node_map["fuzzer_000"].target.cwd.endswith(
        "/codex_fuzz_python_grouped_128/agents/libpng_asan_seed_a_000"
    )
    assert spec.node_map["family_merge_0"].fanout_member["target"] == "libpng"
    assert spec.node_map["family_merge_0"].fanout_member["corpus"] == "png"
    assert spec.node_map["family_merge_0"].fanout_member["member_ids"] == spec.fanouts["fuzzer"][:32]
    assert spec.node_map["family_merge_0"].depends_on == spec.fanouts["fuzzer"][:32]
    assert spec.node_map["family_merge_3"].fanout_member["target"] == "sqlite"
    assert spec.node_map["family_merge_0"].prompt.startswith(
        "Prepare the maintainer handoff for target family libpng (corpus png)."
    )
    assert spec.node_map["merge"].depends_on == spec.fanouts["family_merge"]


def test_pi_factory_produces_pi_agent_node():
    with Graph("pi-smoke") as g:
        node = pi(
            task_id="scan",
            prompt="Scan repo for TODOs",
            model="lmstudio/mythos-26b-a4b-prism-pro-dq-mlx",
            tools="read_only",
        )

    payload = g.to_payload()
    assert len(payload["nodes"]) == 1
    spec = payload["nodes"][0]
    assert spec["agent"] == "pi"
    assert spec["id"] == "scan"
    assert spec["model"] == "lmstudio/mythos-26b-a4b-prism-pro-dq-mlx"
    assert spec["tools"] == "read_only"

    # Pipeline round-trips through the loader.
    rendered = g.to_json()
    loaded = load_pipeline_from_text(rendered, base_dir=Path(__file__).resolve().parents[1])
    assert loaded.nodes[0].agent == "pi"
