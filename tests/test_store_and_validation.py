from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from agentflow.specs import LocalTarget, PipelineSpec, RunEvent, RunRecord
from agentflow.store import RunStore


def test_pipeline_validation_applies_node_defaults_and_agent_defaults():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "node-defaults",
            "working_dir": ".",
            "node_defaults": {
                "agent": "codex",
                "tools": "read_only",
                "capture": "final",
                "env": {"SHARED_ENV": "1"},
                "extra_args": ["--search"],
                "target": {
                    "kind": "local",
                    "shell": "bash",
                },
            },
            "agent_defaults": {
                "codex": {
                    "model": "gpt-5-codex",
                    "retries": 1,
                    "retry_backoff_seconds": 2,
                    "env": {"CODEX_ENV": "1"},
                    "extra_args": ["-c", 'model_reasoning_effort="high"'],
                }
            },
            "nodes": [
                {
                    "id": "review",
                    "fanout": {
                        "count": 2,
                        "as": "shard",
                    },
                    "prompt": "review {{ shard.number }}",
                    "target": {
                        "kind": "local",
                        "cwd": "agents/{{ shard.suffix }}",
                    },
                },
                {
                    "id": "merge",
                    "prompt": "merge",
                    "tools": "read_write",
                    "env": {"MERGE_ENV": "1"},
                    "extra_args": ["--freeform"],
                },
            ],
        }
    )

    assert pipeline.fanouts == {"review": ["review_0", "review_1"]}
    assert pipeline.node_map["review_0"].agent == "codex"
    assert pipeline.node_map["review_0"].model == "gpt-5-codex"
    assert pipeline.node_map["review_0"].tools == "read_only"
    assert pipeline.node_map["review_0"].capture == "final"
    assert pipeline.node_map["review_0"].retries == 1
    assert pipeline.node_map["review_0"].retry_backoff_seconds == 2
    assert pipeline.node_map["review_0"].env == {"SHARED_ENV": "1", "CODEX_ENV": "1"}
    assert pipeline.node_map["review_0"].extra_args == ["--search", "-c", 'model_reasoning_effort="high"']
    assert pipeline.node_map["review_0"].target.shell == "bash"
    assert pipeline.node_map["review_0"].target.cwd == "agents/0"
    assert pipeline.node_map["merge"].tools == "read_write"
    assert pipeline.node_map["merge"].env == {"SHARED_ENV": "1", "CODEX_ENV": "1", "MERGE_ENV": "1"}
    assert pipeline.node_map["merge"].extra_args == [
        "--search",
        "-c",
        'model_reasoning_effort="high"',
        "--freeform",
    ]


def test_pipeline_validation_rejects_forbidden_node_default_fields():
    with pytest.raises(ValueError, match="node_defaults.*prompt"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-node-defaults",
                "working_dir": ".",
                "node_defaults": {
                    "prompt": "shared prompt",
                },
                "nodes": [
                    {"id": "plan", "agent": "codex", "prompt": "plan"},
                ],
            }
        )


def test_pipeline_validation_rejects_agent_field_inside_agent_defaults():
    with pytest.raises(ValueError, match="agent_defaults\\.codex.*agent"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-agent-defaults",
                "working_dir": ".",
                "agent_defaults": {
                    "codex": {
                        "agent": "codex",
                    }
                },
                "nodes": [
                    {"id": "plan", "agent": "codex", "prompt": "plan"},
                ],
            }
        )


def test_pipeline_validation_accepts_cycles():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "cycle",
            "working_dir": ".",
            "nodes": [
                {"id": "a", "agent": "codex", "prompt": "a", "depends_on": ["b"]},
                {"id": "b", "agent": "codex", "prompt": "b", "depends_on": ["a"]},
            ],
        }
    )
    assert pipeline.node_map["a"].depends_on == ["b"]
    assert pipeline.node_map["b"].depends_on == ["a"]


def test_pipeline_validation_rejects_codex_kimi_provider_alias():
    with pytest.raises(ValueError, match="provider 'kimi' is not supported for codex nodes"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-provider",
                "working_dir": ".",
                "nodes": [
                    {"id": "plan", "agent": "codex", "prompt": "plan", "provider": "kimi"},
                ],
            }
        )


@pytest.mark.parametrize("provider", ["minimax", "minimax-cn"])
def test_pipeline_validation_rejects_codex_minimax_provider_alias(provider):
    with pytest.raises(ValueError, match=rf"provider '{provider}' is only supported for `pi` and `claude` nodes"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-provider",
                "working_dir": ".",
                "nodes": [
                    {"id": "plan", "agent": "codex", "prompt": "plan", "provider": provider},
                ],
            }
        )


def test_pipeline_validation_accepts_graph_inference_setup():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "graph-inference",
            "working_dir": ".",
            "inference": {
                "gpu": "aws:8xb200@us-east-1",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "engine": "vllm",
            },
            "nodes": [
                {"id": "solve", "agent": "pi", "prompt": "solve"},
            ],
        }
    )

    assert pipeline.inference is not None
    assert pipeline.inference.gpu == "aws:8xb200@us-east-1"
    assert pipeline.inference.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert pipeline.inference.use_spot is True
    assert pipeline.inference.port == 8000


def test_pipeline_validation_rejects_invalid_graph_inference_setup():
    with pytest.raises(ValidationError):
        PipelineSpec.model_validate(
            {
                "name": "invalid-graph-inference",
                "working_dir": ".",
                "inference": {
                    "gpu": "",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "engine": "bad",
                },
                "nodes": [
                    {"id": "solve", "agent": "pi", "prompt": "solve"},
                ],
            }
        )


@pytest.mark.parametrize(
    ("target_patch", "expected_field"),
    [
        ({"shell_login": True}, "shell_login"),
        ({"shell_interactive": True}, "shell_interactive"),
        ({"shell_init": "kimi"}, "shell_init"),
        ({"shell_init": ["kimi"]}, "shell_init"),
    ],
)
def test_pipeline_validation_rejects_local_shell_bootstrap_without_shell(target_patch, expected_field):
    with pytest.raises(ValueError, match=rf"{expected_field}.*target\.shell"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-local-shell-target",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "target": {"kind": "local", **target_patch},
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    ("shell", "expected_message"),
    [
        ("bash --command 'kimi && {command}'", r"unsupported bash long option.*--command.*use `-c`"),
        ("bash --command='kimi && {command}'", r"unsupported bash long option.*--command.*use `-c`"),
        (
            "env BASH_ENV=/tmp/kimi.env bash --interactive -c 'kimi && {command}'",
            r"unsupported bash long option.*--interactive.*target\.shell_interactive",
        ),
        (
            "bash --rcfile=\"$HOME/.bashrc\" -ic 'kimi && {command}'",
            r"unsupported bash long option.*--rcfile=.*pass `--rcfile` and its value as separate arguments",
        ),
        (
            "bash --login=1 -c 'kimi && {command}'",
            r"unsupported bash long option.*--login=.*use `--login` without `=`",
        ),
    ],
)
def test_pipeline_validation_rejects_unsupported_bash_long_options(shell, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        PipelineSpec.model_validate(
            {
                "name": "invalid-bash-long-options",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "target": {"kind": "local", "shell": shell, "shell_init": "kimi"},
                    },
                ],
            }
        )


def test_pipeline_validation_accepts_supported_bash_long_options_with_separate_value():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "valid-bash-long-options",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "plan",
                    "agent": "claude",
                    "prompt": "plan",
                    "target": {
                        "kind": "local",
                        "shell": "bash --rcfile $HOME/.bashrc -ic 'kimi && {command}'",
                        "shell_init": "kimi",
                    },
                },
            ],
        }
    )

    assert pipeline.nodes[0].target.shell == "bash --rcfile $HOME/.bashrc -ic 'kimi && {command}'"


@pytest.mark.parametrize(
    "shell",
    [
        "bash -lc 'echo pre'",
        "env BASH_ENV=/tmp/shell.env bash -lc 'echo pre'",
    ],
)
def test_pipeline_validation_rejects_shell_command_payload_without_command_placeholder(shell):
    with pytest.raises(ValueError, match=r"shell command payload.*\{command\}"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-shell-command-payload",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "target": {"kind": "local", "shell": shell},
                    },
                ],
            }
        )


def test_pipeline_validation_applies_local_target_defaults_to_local_nodes():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "local-target-defaults",
            "working_dir": ".",
            "local_target_defaults": {
                "shell": "bash",
                "shell_login": True,
                "shell_interactive": True,
                "shell_init": ["command -v kimi >/dev/null 2>&1", "kimi"],
            },
            "nodes": [
                {"id": "plan", "agent": "codex", "prompt": "plan"},
                {
                    "id": "review",
                    "agent": "claude",
                    "prompt": "review",
                    "target": {"cwd": "review-work", "shell_init": "custom-kimi"},
                },
                {
                    "id": "remote",
                    "agent": "kimi",
                    "prompt": "remote",
                    "target": {"kind": "container", "image": "python:3.12"},
                },
            ],
        }
    )

    assert pipeline.local_target_defaults is not None
    assert pipeline.nodes[0].target.shell == "bash"
    assert pipeline.nodes[0].target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]
    assert pipeline.nodes[1].target.shell == "bash"
    assert pipeline.nodes[1].target.cwd == "review-work"
    assert pipeline.nodes[1].target.shell_init == "custom-kimi"
    assert pipeline.nodes[2].target.kind == "container"


def test_pipeline_validation_expands_kimi_bootstrap_shorthand():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "local-bootstrap-shorthand",
            "working_dir": ".",
            "local_target_defaults": {
                "bootstrap": "kimi",
            },
            "nodes": [
                {"id": "plan", "agent": "codex", "prompt": "plan"},
                {
                    "id": "review",
                    "agent": "claude",
                    "prompt": "review",
                    "target": {"bootstrap": "kimi", "cwd": "review-work"},
                },
            ],
        }
    )

    assert pipeline.local_target_defaults is not None
    assert pipeline.local_target_defaults.bootstrap == "kimi"
    assert pipeline.local_target_defaults.shell == "bash"
    assert pipeline.local_target_defaults.shell_login is True
    assert pipeline.local_target_defaults.shell_interactive is True
    assert pipeline.local_target_defaults.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]
    assert pipeline.nodes[0].target.bootstrap == "kimi"
    assert pipeline.nodes[0].target.shell == "bash"
    assert pipeline.nodes[0].target.shell_login is True
    assert pipeline.nodes[0].target.shell_interactive is True
    assert pipeline.nodes[0].target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]
    assert pipeline.nodes[1].target.bootstrap == "kimi"
    assert pipeline.nodes[1].target.cwd == "review-work"
    assert pipeline.nodes[1].target.shell == "bash"
    assert pipeline.nodes[1].target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]


def test_pipeline_validation_expands_fanout_nodes_and_group_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-validation",
            "working_dir": ".",
            "local_target_defaults": {
                "bootstrap": "kimi",
            },
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "count": 3,
                        "as": "shard",
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.number }}/{{ shard.count }} in {{ shard.node_id }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {"fuzz": ["fuzz_0", "fuzz_1", "fuzz_2"]}
    assert [node.id for node in pipeline.nodes] == ["fuzz_0", "fuzz_1", "fuzz_2", "merge"]
    assert pipeline.nodes[0].prompt == "fuzz 1/3 in fuzz_0"
    assert pipeline.nodes[1].prompt == "fuzz 2/3 in fuzz_1"
    assert pipeline.nodes[2].prompt == "fuzz 3/3 in fuzz_2"
    assert pipeline.nodes[0].target.bootstrap == "kimi"
    assert pipeline.nodes[1].target.bootstrap == "kimi"
    assert pipeline.node_map["merge"].depends_on == ["fuzz_0", "fuzz_1", "fuzz_2"]


def test_pipeline_validation_expands_fanout_value_nodes_and_group_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-values-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "as": "shard",
                        "values": [
                            {"target": "libpng", "seed": 101},
                            {"target": "sqlite", "seed": 202},
                        ],
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.target }} with {{ shard.value.seed }} in {{ shard.node_id }}",
                    "target": {
                        "kind": "local",
                        "cwd": "agents/{{ shard.target }}_{{ shard.suffix }}",
                    },
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {"fuzz": ["fuzz_0", "fuzz_1"]}
    assert [node.id for node in pipeline.nodes] == ["fuzz_0", "fuzz_1", "merge"]
    assert pipeline.nodes[0].prompt == "fuzz libpng with 101 in fuzz_0"
    assert pipeline.nodes[1].prompt == "fuzz sqlite with 202 in fuzz_1"
    assert pipeline.nodes[0].fanout_group == "fuzz"
    assert pipeline.nodes[0].fanout_member == {
        "index": 0,
        "number": 1,
        "count": 2,
        "suffix": "0",
        "value": {"target": "libpng", "seed": 101},
        "template_id": "fuzz",
        "node_id": "fuzz_0",
        "target": "libpng",
        "seed": 101,
    }
    assert pipeline.nodes[0].target.cwd == "agents/libpng_0"
    assert pipeline.nodes[1].target.cwd == "agents/sqlite_1"
    assert pipeline.node_map["merge"].depends_on == ["fuzz_0", "fuzz_1"]


def test_pipeline_validation_expands_fanout_matrix_nodes_and_group_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-matrix-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [
                                {"target": "libpng", "corpus": "png"},
                                {"target": "sqlite", "corpus": "sql"},
                            ],
                            "variant": [
                                {"sanitizer": "asan", "seed": 101},
                                {"sanitizer": "ubsan", "seed": 202},
                            ],
                        },
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.target }} {{ shard.sanitizer }} seed {{ shard.variant.seed }} in {{ shard.node_id }}",
                    "target": {
                        "kind": "local",
                        "cwd": "agents/{{ shard.target }}_{{ shard.sanitizer }}_{{ shard.suffix }}",
                    },
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {"fuzz": ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3"]}
    assert [node.id for node in pipeline.nodes] == ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3", "merge"]
    assert pipeline.nodes[0].prompt == "fuzz libpng asan seed 101 in fuzz_0"
    assert pipeline.nodes[1].prompt == "fuzz libpng ubsan seed 202 in fuzz_1"
    assert pipeline.nodes[2].prompt == "fuzz sqlite asan seed 101 in fuzz_2"
    assert pipeline.nodes[3].prompt == "fuzz sqlite ubsan seed 202 in fuzz_3"
    assert pipeline.nodes[0].fanout_group == "fuzz"
    assert pipeline.nodes[0].fanout_member is not None
    assert pipeline.nodes[0].fanout_member["family"] == {"target": "libpng", "corpus": "png"}
    assert pipeline.nodes[0].fanout_member["variant"] == {"sanitizer": "asan", "seed": 101}
    assert pipeline.nodes[0].fanout_member["target"] == "libpng"
    assert pipeline.nodes[0].fanout_member["corpus"] == "png"
    assert pipeline.nodes[0].fanout_member["sanitizer"] == "asan"
    assert pipeline.nodes[0].fanout_member["seed"] == 101
    assert pipeline.nodes[0].target.cwd == "agents/libpng_asan_0"
    assert pipeline.nodes[3].target.cwd == "agents/sqlite_ubsan_3"
    assert pipeline.node_map["merge"].depends_on == ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3"]


def test_pipeline_validation_expands_fanout_derived_fields():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-derived-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [
                                {"target": "libpng", "corpus": "png"},
                                {"target": "sqlite", "corpus": "sql"},
                            ],
                            "variant": [
                                {"sanitizer": "asan", "seed": 101},
                                {"sanitizer": "ubsan", "seed": 202},
                            ],
                        },
                        "derive": {
                            "workspace": "agents/{{ shard.target }}_{{ shard.sanitizer }}_{{ shard.suffix }}",
                            "label": "{{ shard.target }}/{{ shard.sanitizer }}/{{ shard.seed }}",
                            "paths": {
                                "workspace": "{{ shard.workspace }}",
                                "report": "reports/{{ shard.target }}_{{ shard.suffix }}.md",
                            },
                        },
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.label }} in {{ shard.paths.workspace }}",
                    "target": {
                        "kind": "local",
                        "cwd": "{{ shard.workspace }}",
                    },
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {"fuzz": ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3"]}
    assert pipeline.nodes[0].prompt == "fuzz libpng/asan/101 in agents/libpng_asan_0"
    assert pipeline.nodes[3].prompt == "fuzz sqlite/ubsan/202 in agents/sqlite_ubsan_3"
    assert pipeline.nodes[0].fanout_member is not None
    assert pipeline.nodes[0].fanout_member["workspace"] == "agents/libpng_asan_0"
    assert pipeline.nodes[0].fanout_member["label"] == "libpng/asan/101"
    assert pipeline.nodes[0].fanout_member["paths"] == {
        "workspace": "agents/libpng_asan_0",
        "report": "reports/libpng_0.md",
    }
    assert pipeline.nodes[3].fanout_member["workspace"] == "agents/sqlite_ubsan_3"
    assert pipeline.nodes[3].fanout_member["label"] == "sqlite/ubsan/202"
    assert pipeline.nodes[0].target.cwd == "agents/libpng_asan_0"
    assert pipeline.nodes[3].target.cwd == "agents/sqlite_ubsan_3"
    assert pipeline.node_map["merge"].depends_on == ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3"]


def test_pipeline_validation_expands_grouped_fanout_nodes_and_group_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-group-by-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [
                                {"target": "libpng", "corpus": "png"},
                                {"target": "sqlite", "corpus": "sql"},
                            ],
                            "variant": [
                                {"sanitizer": "asan", "seed": 101},
                                {"sanitizer": "ubsan", "seed": 202},
                            ],
                        },
                        "derive": {
                            "family_label": "{{ shard.target }}/{{ shard.corpus }}",
                        },
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.family_label }} {{ shard.sanitizer }}",
                },
                {
                    "id": "family_merge",
                    "fanout": {
                        "as": "family",
                        "group_by": {
                            "from": "fuzz",
                            "fields": ["target", "corpus", "family_label"],
                        },
                    },
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge {{ family.family_label }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["family_merge"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {
        "fuzz": ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3"],
        "family_merge": ["family_merge_0", "family_merge_1"],
    }
    assert [node.id for node in pipeline.nodes] == [
        "fuzz_0",
        "fuzz_1",
        "fuzz_2",
        "fuzz_3",
        "family_merge_0",
        "family_merge_1",
        "merge",
    ]
    assert pipeline.node_map["family_merge_0"].prompt == "merge libpng/png"
    assert pipeline.node_map["family_merge_1"].prompt == "merge sqlite/sql"
    assert pipeline.node_map["family_merge_0"].fanout_member == {
        "index": 0,
        "number": 1,
        "count": 2,
        "suffix": "0",
        "value": {
            "source_group": "fuzz",
            "source_count": 4,
            "size": 2,
            "member_ids": ["fuzz_0", "fuzz_1"],
            "members": [
                {
                    "index": 0,
                    "number": 1,
                    "count": 4,
                    "suffix": "0",
                    "value": {
                        "family": {"target": "libpng", "corpus": "png"},
                        "variant": {"sanitizer": "asan", "seed": 101},
                        "target": "libpng",
                        "corpus": "png",
                        "sanitizer": "asan",
                        "seed": 101,
                    },
                    "template_id": "fuzz",
                    "node_id": "fuzz_0",
                    "family": {"target": "libpng", "corpus": "png"},
                    "variant": {"sanitizer": "asan", "seed": 101},
                    "target": "libpng",
                    "corpus": "png",
                    "sanitizer": "asan",
                    "seed": 101,
                    "family_label": "libpng/png",
                },
                {
                    "index": 1,
                    "number": 2,
                    "count": 4,
                    "suffix": "1",
                    "value": {
                        "family": {"target": "libpng", "corpus": "png"},
                        "variant": {"sanitizer": "ubsan", "seed": 202},
                        "target": "libpng",
                        "corpus": "png",
                        "sanitizer": "ubsan",
                        "seed": 202,
                    },
                    "template_id": "fuzz",
                    "node_id": "fuzz_1",
                    "family": {"target": "libpng", "corpus": "png"},
                    "variant": {"sanitizer": "ubsan", "seed": 202},
                    "target": "libpng",
                    "corpus": "png",
                    "sanitizer": "ubsan",
                    "seed": 202,
                    "family_label": "libpng/png",
                },
            ],
            "target": "libpng",
            "corpus": "png",
            "family_label": "libpng/png",
        },
        "template_id": "family_merge",
        "node_id": "family_merge_0",
        "source_group": "fuzz",
        "source_count": 4,
        "size": 2,
        "member_ids": ["fuzz_0", "fuzz_1"],
        "members": [
            {
                "index": 0,
                "number": 1,
                "count": 4,
                "suffix": "0",
                "value": {
                    "family": {"target": "libpng", "corpus": "png"},
                    "variant": {"sanitizer": "asan", "seed": 101},
                    "target": "libpng",
                    "corpus": "png",
                    "sanitizer": "asan",
                    "seed": 101,
                },
                "template_id": "fuzz",
                "node_id": "fuzz_0",
                "family": {"target": "libpng", "corpus": "png"},
                "variant": {"sanitizer": "asan", "seed": 101},
                "target": "libpng",
                "corpus": "png",
                "sanitizer": "asan",
                "seed": 101,
                "family_label": "libpng/png",
            },
            {
                "index": 1,
                "number": 2,
                "count": 4,
                "suffix": "1",
                "value": {
                    "family": {"target": "libpng", "corpus": "png"},
                    "variant": {"sanitizer": "ubsan", "seed": 202},
                    "target": "libpng",
                    "corpus": "png",
                    "sanitizer": "ubsan",
                    "seed": 202,
                },
                "template_id": "fuzz",
                "node_id": "fuzz_1",
                "family": {"target": "libpng", "corpus": "png"},
                "variant": {"sanitizer": "ubsan", "seed": 202},
                "target": "libpng",
                "corpus": "png",
                "sanitizer": "ubsan",
                "seed": 202,
                "family_label": "libpng/png",
            },
        ],
        "target": "libpng",
        "corpus": "png",
        "family_label": "libpng/png",
    }
    assert pipeline.node_map["family_merge_1"].fanout_member["target"] == "sqlite"
    assert pipeline.node_map["family_merge_1"].fanout_member["member_ids"] == ["fuzz_2", "fuzz_3"]
    assert pipeline.node_map["family_merge_0"].depends_on == ["fuzz_0", "fuzz_1"]
    assert pipeline.node_map["family_merge_1"].depends_on == ["fuzz_2", "fuzz_3"]
    assert pipeline.node_map["merge"].depends_on == ["family_merge_0", "family_merge_1"]


def test_pipeline_validation_expands_batched_fanout_nodes_and_scoped_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-batches-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "count": 5,
                        "as": "shard",
                        "derive": {
                            "workspace": "agents/agent_{{ shard.suffix }}",
                        },
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.number }} in {{ shard.workspace }}",
                    "target": {
                        "kind": "local",
                        "cwd": "{{ shard.workspace }}",
                    },
                },
                {
                    "id": "batch_merge",
                    "fanout": {
                        "as": "batch",
                        "batches": {
                            "from": "fuzz",
                            "size": 2,
                        },
                    },
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge batch {{ batch.start_number }}-{{ batch.end_number }}",
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["batch_merge"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {
        "fuzz": ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3", "fuzz_4"],
        "batch_merge": ["batch_merge_0", "batch_merge_1", "batch_merge_2"],
    }
    assert [node.id for node in pipeline.nodes] == [
        "fuzz_0",
        "fuzz_1",
        "fuzz_2",
        "fuzz_3",
        "fuzz_4",
        "batch_merge_0",
        "batch_merge_1",
        "batch_merge_2",
        "merge",
    ]
    assert pipeline.node_map["batch_merge_0"].prompt == "merge batch 1-2"
    assert pipeline.node_map["batch_merge_1"].prompt == "merge batch 3-4"
    assert pipeline.node_map["batch_merge_2"].prompt == "merge batch 5-5"
    assert pipeline.node_map["batch_merge_0"].fanout_member == {
        "index": 0,
        "number": 1,
        "count": 3,
        "suffix": "0",
        "value": {
            "source_group": "fuzz",
            "source_count": 5,
            "size": 2,
            "member_ids": ["fuzz_0", "fuzz_1"],
            "members": [
                {
                    "index": 0,
                    "number": 1,
                    "count": 5,
                    "suffix": "0",
                    "value": 0,
                    "template_id": "fuzz",
                    "node_id": "fuzz_0",
                    "workspace": "agents/agent_0",
                },
                {
                    "index": 1,
                    "number": 2,
                    "count": 5,
                    "suffix": "1",
                    "value": 1,
                    "template_id": "fuzz",
                    "node_id": "fuzz_1",
                    "workspace": "agents/agent_1",
                },
            ],
            "start_index": 0,
            "end_index": 1,
            "start_number": 1,
            "end_number": 2,
            "start_suffix": "0",
            "end_suffix": "1",
        },
        "template_id": "batch_merge",
        "node_id": "batch_merge_0",
        "source_group": "fuzz",
        "source_count": 5,
        "size": 2,
        "member_ids": ["fuzz_0", "fuzz_1"],
        "members": [
            {
                "index": 0,
                "number": 1,
                "count": 5,
                "suffix": "0",
                "value": 0,
                "template_id": "fuzz",
                "node_id": "fuzz_0",
                "workspace": "agents/agent_0",
            },
            {
                "index": 1,
                "number": 2,
                "count": 5,
                "suffix": "1",
                "value": 1,
                "template_id": "fuzz",
                "node_id": "fuzz_1",
                "workspace": "agents/agent_1",
            },
        ],
        "start_index": 0,
        "end_index": 1,
        "start_number": 1,
        "end_number": 2,
        "start_suffix": "0",
        "end_suffix": "1",
    }
    assert pipeline.node_map["batch_merge_1"].fanout_member["member_ids"] == ["fuzz_2", "fuzz_3"]
    assert pipeline.node_map["batch_merge_2"].fanout_member["member_ids"] == ["fuzz_4"]
    assert pipeline.node_map["batch_merge_0"].depends_on == ["fuzz_0", "fuzz_1"]
    assert pipeline.node_map["batch_merge_1"].depends_on == ["fuzz_2", "fuzz_3"]
    assert pipeline.node_map["batch_merge_2"].depends_on == ["fuzz_4"]
    assert pipeline.node_map["merge"].depends_on == ["batch_merge_0", "batch_merge_1", "batch_merge_2"]


def test_pipeline_validation_expands_curated_fanout_matrix_nodes_and_group_dependencies():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "fanout-matrix-curated-validation",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "fuzz",
                    "fanout": {
                        "as": "shard",
                        "matrix": {
                            "family": [
                                {"target": "libpng", "corpus": "png"},
                                {"target": "sqlite", "corpus": "sql"},
                            ],
                            "variant": [
                                {"sanitizer": "asan", "seed": 101},
                                {"sanitizer": "ubsan", "seed": 202},
                            ],
                            "seed_bucket": [
                                {"bucket": "seed_a"},
                                {"bucket": "seed_b"},
                            ],
                        },
                        "exclude": [
                            {
                                "family": {"target": "sqlite"},
                                "variant": {"sanitizer": "ubsan"},
                            }
                        ],
                        "include": [
                            {
                                "family": {"target": "openssl", "corpus": "tls"},
                                "variant": {"sanitizer": "asan", "seed": 909},
                                "seed_bucket": {"bucket": "seed_tls"},
                            }
                        ],
                        "derive": {
                            "workspace": "agents/{{ shard.target }}_{{ shard.sanitizer }}_{{ shard.bucket }}_{{ shard.suffix }}"
                        },
                    },
                    "agent": "codex",
                    "prompt": "fuzz {{ shard.target }} {{ shard.sanitizer }} {{ shard.bucket }} {{ shard.workspace }}",
                    "target": {
                        "kind": "local",
                        "cwd": "{{ shard.workspace }}",
                    },
                },
                {
                    "id": "merge",
                    "agent": "codex",
                    "depends_on": ["fuzz"],
                    "prompt": "merge",
                },
            ],
        }
    )

    assert pipeline.fanouts == {"fuzz": ["fuzz_0", "fuzz_1", "fuzz_2", "fuzz_3", "fuzz_4", "fuzz_5", "fuzz_6"]}
    assert [node.id for node in pipeline.nodes] == [
        "fuzz_0",
        "fuzz_1",
        "fuzz_2",
        "fuzz_3",
        "fuzz_4",
        "fuzz_5",
        "fuzz_6",
        "merge",
    ]
    assert pipeline.node_map["fuzz_0"].prompt == "fuzz libpng asan seed_a agents/libpng_asan_seed_a_0"
    assert pipeline.node_map["fuzz_5"].prompt == "fuzz sqlite asan seed_b agents/sqlite_asan_seed_b_5"
    assert pipeline.node_map["fuzz_6"].prompt == "fuzz openssl asan seed_tls agents/openssl_asan_seed_tls_6"
    assert pipeline.node_map["fuzz_6"].fanout_member == {
        "index": 6,
        "number": 7,
        "count": 7,
        "suffix": "6",
        "value": {
            "family": {"target": "openssl", "corpus": "tls"},
            "variant": {"sanitizer": "asan", "seed": 909},
            "seed_bucket": {"bucket": "seed_tls"},
            "target": "openssl",
            "corpus": "tls",
            "sanitizer": "asan",
            "seed": 909,
            "bucket": "seed_tls",
        },
        "template_id": "fuzz",
        "node_id": "fuzz_6",
        "family": {"target": "openssl", "corpus": "tls"},
        "variant": {"sanitizer": "asan", "seed": 909},
        "seed_bucket": {"bucket": "seed_tls"},
        "target": "openssl",
        "corpus": "tls",
        "sanitizer": "asan",
        "seed": 909,
        "bucket": "seed_tls",
        "workspace": "agents/openssl_asan_seed_tls_6",
    }
    assert pipeline.node_map["fuzz_6"].target.cwd == "agents/openssl_asan_seed_tls_6"
    assert pipeline.node_map["merge"].depends_on == [
        "fuzz_0",
        "fuzz_1",
        "fuzz_2",
        "fuzz_3",
        "fuzz_4",
        "fuzz_5",
        "fuzz_6",
    ]


def test_pipeline_validation_rejects_fanout_with_multiple_expansion_modes():
    with pytest.raises(
        ValueError,
        match=r"fanout accepts exactly one of `count`, `values`, `matrix`, `group_by`, `batches`",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-shape",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "values": ["a", "b"],
                            "matrix": {"axis": ["a", "b"]},
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_grouped_fanout_with_unknown_source_group():
    with pytest.raises(ValueError, match=r"`fanout\.group_by\.from` references unknown prior fanout group `fuzz`"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-group-by-source",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "family_merge",
                        "fanout": {
                            "as": "family",
                            "group_by": {
                                "from": "fuzz",
                                "fields": ["target"],
                            },
                        },
                        "agent": "codex",
                        "prompt": "merge {{ family.target }}",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_batched_fanout_with_unknown_source_group():
    with pytest.raises(ValueError, match=r"`fanout\.batches\.from` references unknown prior fanout group `fuzz`"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-batches-source",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "batch_merge",
                        "fanout": {
                            "as": "batch",
                            "batches": {
                                "from": "fuzz",
                                "size": 4,
                            },
                        },
                        "agent": "codex",
                        "prompt": "merge batch {{ batch.number }}",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_grouped_fanout_with_missing_source_field():
    with pytest.raises(
        ValueError,
        match=r"`fanout\.group_by\.fields` references `corpus`, but fanout group `fuzz` does not expose that field",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-group-by-field",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "as": "shard",
                        },
                        "agent": "codex",
                        "prompt": "fuzz {{ shard.number }}",
                    },
                    {
                        "id": "family_merge",
                        "fanout": {
                            "as": "family",
                            "group_by": {
                                "from": "fuzz",
                                "fields": ["corpus"],
                            },
                        },
                        "agent": "codex",
                        "depends_on": ["fuzz"],
                        "prompt": "merge {{ family.corpus }}",
                    },
                ],
            }
        )


def test_pipeline_validation_rejects_grouped_fanout_with_reserved_scoped_metadata_field():
    with pytest.raises(
        ValueError,
        match=r"`fanout\.group_by\.fields` cannot use reserved scoped reducer metadata fields `members`",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-group-by-reserved-field",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "as": "shard",
                            "values": [
                                {"members": "libpng"},
                                {"members": "sqlite"},
                            ],
                        },
                        "agent": "codex",
                        "prompt": "fuzz {{ shard.members }}",
                    },
                    {
                        "id": "family_merge",
                        "fanout": {
                            "as": "family",
                            "group_by": {
                                "from": "fuzz",
                                "fields": ["members"],
                            },
                        },
                        "agent": "codex",
                        "depends_on": ["fuzz"],
                        "prompt": "merge {{ family.members }}",
                    },
                ],
            }
        )


def test_pipeline_validation_rejects_fanout_alias_that_shadows_reserved_context():
    with pytest.raises(
        ValueError,
        match=r"`fanout\.as` uses a reserved template variable name; choose something other than `fanout`, `fanouts`, `nodes`, `pipeline`, or `item`",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-alias-nodes",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "as": "nodes",
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_curated_fanout_without_matrix():
    with pytest.raises(
        ValueError,
        match=r"`fanout.include` and `fanout.exclude` require `fanout.matrix`",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-curation-shape",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "include": [{"target": "libpng"}],
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_curated_fanout_matrix_without_remaining_members():
    with pytest.raises(
        ValueError,
        match=r"`fanout.matrix` produced no members after applying `fanout.exclude`",
    ):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-curation-empty",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "matrix": {
                                "family": [{"target": "libpng"}],
                            },
                            "exclude": [{"target": "libpng"}],
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_reserved_fanout_matrix_axis_name():
    with pytest.raises(ValueError, match=r"fanout\.matrix.*reserved member fields"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-matrix-axis",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "matrix": {
                                "suffix": ["a", "b"],
                            },
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_reserved_fanout_derive_field_name():
    with pytest.raises(ValueError, match=r"fanout\.derive.*reserved member fields"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-derive-field",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "derive": {
                                "suffix": "agent_{{ fanout.suffix }}",
                            },
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_conflicting_fanout_derive_field():
    with pytest.raises(ValueError, match=r"fanout\.derive field `target` conflicts with an existing member field"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout-derive-conflict",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "as": "shard",
                            "values": [{"target": "libpng"}],
                            "derive": {
                                "target": "sqlite",
                            },
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_invalid_fanout_alias():
    with pytest.raises(ValueError, match=r"fanout\.as.*valid template variable name"):
        PipelineSpec.model_validate(
            {
                "name": "bad-fanout",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                            "as": "bad-alias",
                        },
                        "agent": "codex",
                        "prompt": "hi",
                    }
                ],
            }
        )


def test_pipeline_validation_rejects_fanout_group_id_collision_with_normal_node():
    with pytest.raises(ValueError, match=r"duplicate node ids: \['fuzz'\]"):
        PipelineSpec.model_validate(
            {
                "name": "fanout-id-collision",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "fuzz",
                        "fanout": {
                            "count": 2,
                        },
                        "agent": "codex",
                        "prompt": "fanout",
                    },
                    {
                        "id": "fuzz",
                        "agent": "codex",
                        "prompt": "plain",
                    },
                ],
            }
        )


def test_pipeline_validation_merges_extra_shell_init_into_kimi_bootstrap_shorthand():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "local-bootstrap-shell-init-merge",
            "working_dir": ".",
            "local_target_defaults": {
                "bootstrap": "kimi",
            },
            "nodes": [
                {
                    "id": "review",
                    "agent": "claude",
                    "prompt": "review",
                    "target": {
                        "shell_init": ["export EXTRA_FLAG=1"],
                    },
                }
            ],
        }
    )

    assert pipeline.nodes[0].target.bootstrap == "kimi"
    assert pipeline.nodes[0].target.shell == "bash"
    assert pipeline.nodes[0].target.shell_login is True
    assert pipeline.nodes[0].target.shell_interactive is True
    assert pipeline.nodes[0].target.shell_init == [
        "export EXTRA_FLAG=1",
        "command -v kimi >/dev/null 2>&1",
        "kimi",
    ]


def test_pipeline_validation_allows_per_node_opt_out_from_inherited_kimi_bootstrap_defaults():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "local-bootstrap-opt-out",
            "working_dir": ".",
            "local_target_defaults": LocalTarget.model_validate(
                {
                    "bootstrap": "kimi",
                    "cwd": "shared-work",
                }
            ),
            "nodes": [
                {"id": "shared", "agent": "claude", "prompt": "shared"},
                {
                    "id": "plain",
                    "agent": "codex",
                    "prompt": "plain",
                    "target": {"bootstrap": None},
                },
            ],
        }
    )

    shared_target = pipeline.nodes[0].target
    plain_target = pipeline.nodes[1].target

    assert shared_target.bootstrap == "kimi"
    assert shared_target.shell == "bash"
    assert shared_target.shell_login is True
    assert shared_target.shell_interactive is True
    assert shared_target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]
    assert shared_target.cwd == "shared-work"

    assert plain_target.bootstrap is None
    assert plain_target.shell is None
    assert plain_target.shell_login is False
    assert plain_target.shell_interactive is False
    assert plain_target.shell_init is None
    assert plain_target.cwd == "shared-work"


def test_pipeline_validation_opt_out_drops_customized_kimi_shell_defaults():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "local-bootstrap-custom-opt-out",
            "working_dir": ".",
            "local_target_defaults": LocalTarget.model_validate(
                {
                    "bootstrap": "kimi",
                    "cwd": "shared-work",
                    "shell_init": ["export EXTRA_FLAG=1"],
                }
            ),
            "nodes": [
                {"id": "shared", "agent": "claude", "prompt": "shared"},
                {
                    "id": "plain",
                    "agent": "codex",
                    "prompt": "plain",
                    "target": {"bootstrap": None},
                },
            ],
        }
    )

    shared_target = pipeline.nodes[0].target
    plain_target = pipeline.nodes[1].target

    assert shared_target.bootstrap == "kimi"
    assert shared_target.shell == "bash"
    assert shared_target.shell_login is True
    assert shared_target.shell_interactive is True
    assert shared_target.shell_init == [
        "export EXTRA_FLAG=1",
        "command -v kimi >/dev/null 2>&1",
        "kimi",
    ]
    assert shared_target.cwd == "shared-work"

    assert plain_target.bootstrap is None
    assert plain_target.shell is None
    assert plain_target.shell_login is False
    assert plain_target.shell_interactive is False
    assert plain_target.shell_init is None
    assert plain_target.cwd == "shared-work"


@pytest.mark.parametrize(
    ("target_patch", "expected_message"),
    [
        (
            {"bootstrap": "kimi", "shell": "sh"},
            r"target\.bootstrap: kimi.*requires bash-style shell bootstrap.*target\.shell.*`sh`",
        ),
        (
            {"bootstrap": "kimi", "shell": "bash", "shell_interactive": False},
            r"target\.bootstrap: kimi.*requires interactive bash startup",
        ),
        (
            {"bootstrap": "kimi", "shell": "bash --noprofile -lic '{command}'"},
            r"target\.bootstrap: kimi.*--noprofile",
        ),
        (
            {"bootstrap": "kimi", "shell": "bash --norc -ic '{command}'", "shell_login": False},
            r"target\.bootstrap: kimi.*--norc",
        ),
    ],
)
def test_pipeline_validation_rejects_incompatible_kimi_bootstrap_overrides(target_patch, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        PipelineSpec.model_validate(
            {
                "name": "invalid-kimi-bootstrap-override",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "review",
                        "agent": "claude",
                        "prompt": "review",
                        "target": {"kind": "local", **target_patch},
                    },
                ],
            }
        )


def test_pipeline_validation_rejects_non_bash_shell_overriding_local_kimi_bootstrap_defaults():
    with pytest.raises(ValueError, match=r"target\.bootstrap: kimi.*requires bash-style shell bootstrap.*`sh`"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-kimi-bootstrap-default-override",
                "working_dir": ".",
                "local_target_defaults": {
                    "bootstrap": "kimi",
                },
                "nodes": [
                    {
                        "id": "review",
                        "agent": "claude",
                        "prompt": "review",
                        "target": {"kind": "local", "shell": "sh"},
                    },
                ],
            }
        )


def test_pipeline_validation_rejects_unknown_local_bootstrap():
    with pytest.raises(ValidationError, match=r"target\.bootstrap.*must be `kimi`"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-local-bootstrap",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "target": {"kind": "local", "bootstrap": "other"},
                    },
                ],
            }
        )


def test_pipeline_validation_accepts_shell_init_command_lists():
    pipeline = PipelineSpec.model_validate(
        {
            "name": "valid-shell-init-list",
            "working_dir": ".",
            "nodes": [
                {
                    "id": "plan",
                    "agent": "claude",
                    "prompt": "plan",
                    "target": {
                        "kind": "local",
                        "shell": "bash",
                        "shell_init": [" command -v kimi >/dev/null 2>&1 ", " kimi "],
                    },
                },
            ],
        }
    )

    assert pipeline.nodes[0].target.shell_init == ["command -v kimi >/dev/null 2>&1", "kimi"]


def test_pipeline_validation_rejects_empty_shell_init_list_entries():
    with pytest.raises(ValidationError, match="non-empty strings"):
        PipelineSpec.model_validate(
            {
                "name": "invalid-shell-init-list",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "target": {
                            "kind": "local",
                            "shell": "bash",
                            "shell_init": ["kimi", "   "],
                        },
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    ("mcp_patch", "expected_message"),
    [
        ({"transport": "stdio"}, "stdio MCP servers require `command`"),
        (
            {"transport": "stdio", "command": "npx", "url": "https://example.com/mcp"},
            "stdio MCP servers do not support `url`",
        ),
        ({"transport": "streamable_http"}, "streamable_http MCP servers require `url`"),
        (
            {"transport": "streamable_http", "url": "https://example.com/mcp", "command": "npx"},
            "streamable_http MCP servers do not support `command`",
        ),
    ],
)
def test_pipeline_validation_rejects_invalid_mcp_server_shape(mcp_patch, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        PipelineSpec.model_validate(
            {
                "name": "invalid-mcp-shape",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "mcps": [{"name": "github", **mcp_patch}],
                    },
                ],
            }
        )


def test_pipeline_validation_rejects_duplicate_mcp_server_names():
    with pytest.raises(ValueError, match="duplicate MCP server names"):
        PipelineSpec.model_validate(
            {
                "name": "duplicate-mcps",
                "working_dir": ".",
                "nodes": [
                    {
                        "id": "plan",
                        "agent": "claude",
                        "prompt": "plan",
                        "mcps": [
                            {"name": "github", "command": "npx"},
                            {"name": "github", "command": "node"},
                        ],
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    ("pipeline_patch", "node_patch", "expected_loc", "expected_type"),
    [
        ({"concurrency": 0}, {}, ("concurrency",), "greater_than_equal"),
        ({}, {"timeout_seconds": 0}, ("nodes", 0, "timeout_seconds"), "greater_than"),
        ({}, {"retries": -1}, ("nodes", 0, "retries"), "greater_than_equal"),
        ({}, {"retry_backoff_seconds": -0.1}, ("nodes", 0, "retry_backoff_seconds"), "greater_than_equal"),
    ],
)
def test_pipeline_validation_rejects_invalid_numeric_runtime_settings(
    pipeline_patch,
    node_patch,
    expected_loc,
    expected_type,
):
    payload = {
        "name": "invalid-runtime-settings",
        "working_dir": ".",
        "nodes": [
            {
                "id": "plan",
                "agent": "codex",
                "prompt": "plan",
                **node_patch,
            }
        ],
        **pipeline_patch,
    }

    with pytest.raises(ValidationError) as exc_info:
        PipelineSpec.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == expected_loc
    assert exc_info.value.errors()[0]["type"] == expected_type


def test_pipeline_validation_rejects_scheduled_node_before_watched_fanout():
    payload = {
        "name": "scheduled-before-fanout",
        "working_dir": ".",
        "nodes": [
            {
                "id": "monitor",
                "agent": "codex",
                "schedule": {
                    "every_seconds": 600,
                    "until_fanout_settles_from": "worker",
                },
                "prompt": "monitor",
            },
            {
                "id": "worker",
                "fanout": {"count": 2, "as": "shard"},
                "agent": "codex",
                "prompt": "worker {{ shard.number }}",
            },
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        PipelineSpec.model_validate(payload)

    assert "must appear after the watched fanout group" in str(exc_info.value)


def test_pipeline_validation_rejects_scheduled_node_with_non_local_target():
    payload = {
        "name": "scheduled-non-local",
        "working_dir": ".",
        "nodes": [
            {
                "id": "worker",
                "fanout": {"count": 1, "as": "shard"},
                "agent": "codex",
                "prompt": "worker {{ shard.number }}",
            },
            {
                "id": "monitor",
                "agent": "codex",
                "schedule": {
                    "every_seconds": 600,
                    "until_fanout_settles_from": "worker",
                },
                "target": {"kind": "container", "image": "python:3.11"},
                "prompt": "monitor",
            },
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        PipelineSpec.model_validate(payload)

    assert "scheduled nodes currently require a local target" in str(exc_info.value)


@pytest.mark.asyncio
async def test_store_loads_runs_and_artifacts_from_disk(tmp_path):
    pipeline = PipelineSpec.model_validate(
        {
            "name": "persisted",
            "working_dir": str(tmp_path),
            "nodes": [{"id": "alpha", "agent": "codex", "prompt": "hi"}],
        }
    )
    original = RunStore(tmp_path / "runs")
    record = RunRecord(id="run-1", pipeline=pipeline)
    await original.create_run(record)
    await original.append_event("run-1", RunEvent(run_id="run-1", type="run_started"))
    await original.write_artifact_text("run-1", "alpha", "output.txt", "hello persisted")

    reloaded = RunStore(tmp_path / "runs")
    assert reloaded.get_run("run-1").pipeline.name == "persisted"
    assert reloaded.get_events("run-1")[0].type == "run_started"
    assert reloaded.read_artifact_text("run-1", "alpha", "output.txt") == "hello persisted"


def test_store_expands_user_in_base_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    store = RunStore("~/agentflow-runs")

    assert store.base_dir == home / "agentflow-runs"
    assert store.base_dir.exists()
