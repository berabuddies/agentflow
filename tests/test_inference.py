from __future__ import annotations

from pathlib import Path

import pytest

from agentflow.inference.skypilot import (
    SkyInferenceRequest,
    SkyInferenceServiceRequest,
    build_agentflow_provider,
    build_sky_resources_kwargs,
    build_sky_service_task,
    build_sky_task,
    parse_gpu_selector,
    resolve_default_image_id,
)


def test_parse_single_node_cloud_region_gpu_selector():
    selector = parse_gpu_selector("aws:8xb200@us-east-1")

    assert selector.cloud == "aws"
    assert selector.location == "us-east-1"
    assert selector.infra == "aws/us-east-1"
    assert selector.accelerator == "B200"
    assert selector.count == 8
    assert selector.num_nodes is None
    assert selector.accelerators == "B200:8"


def test_parse_multi_node_multi_gpu_selector():
    selector = parse_gpu_selector("aws:8x8xb200@us-east-2")

    assert selector.cloud == "aws"
    assert selector.location == "us-east-2"
    assert selector.infra == "aws/us-east-2"
    assert selector.accelerator == "B200"
    assert selector.count == 8
    assert selector.num_nodes == 8
    assert selector.accelerators == "B200:8"


def test_parse_provider_agnostic_selector():
    selector = parse_gpu_selector("1xl4")

    assert selector.cloud is None
    assert selector.infra is None
    assert selector.accelerator == "L4"
    assert selector.count == 1


def test_build_sky_resources_kwargs_defaults_to_spot():
    selector = parse_gpu_selector("aws:1xl4@us-east-1")

    assert build_sky_resources_kwargs(selector, use_spot=True, max_hourly_cost=2.5) == {
        "infra": "aws/us-east-1",
        "accelerators": "L4:1",
        "use_spot": True,
        "max_hourly_cost": 2.5,
    }


def test_build_sky_task_sets_multi_node_and_resources():
    class FakeResources:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSky:
        Resources = FakeResources
        Task = FakeTask

    request = SkyInferenceRequest(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        gpu=parse_gpu_selector("aws:8x8xb200@us-east-2"),
        input_path=Path("/tmp/prompts.jsonl"),
        use_spot=True,
        max_hourly_cost=10.0,
        image_id="ami-explicit",
    )

    task = build_sky_task(request, sky_module=FakeSky)

    assert task.kwargs["num_nodes"] == 8
    assert task.kwargs["resources"].kwargs == {
        "infra": "aws/us-east-2",
        "accelerators": "B200:8",
        "use_spot": True,
        "max_hourly_cost": 10.0,
        "image_id": "ami-explicit",
    }
    assert "agentflow.inference.worker" in task.kwargs["run"]
    assert "--tensor-parallel-size 8" in task.kwargs["run"]


def test_build_sky_service_task_exposes_port_and_env_key():
    class FakeResources:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSky:
        Resources = FakeResources
        Task = FakeTask

    request = SkyInferenceServiceRequest(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        gpu=parse_gpu_selector("aws:1xl4@us-east-1"),
        api_key="test-key",
        image_id="ami-explicit",
        port=9000,
    )

    task = build_sky_service_task(request, sky_module=FakeSky)

    assert task.kwargs["resources"].kwargs == {
        "infra": "aws/us-east-1",
        "accelerators": "L4:1",
        "use_spot": True,
        "image_id": "ami-explicit",
        "ports": 9000,
    }
    assert task.kwargs["secrets"] == {"AGENTFLOW_INFERENCE_API_KEY": "test-key"}
    assert "vllm.entrypoints.openai.api_server" in task.kwargs["run"]
    assert "--api-key ${AGENTFLOW_INFERENCE_API_KEY}" in task.kwargs["run"]
    assert "http://127.0.0.1:9000/v1/models" in task.kwargs["run"]
    assert "nohup" not in task.kwargs["run"]
    assert "wait \"$(cat /tmp/agentflow-inference/server.pid)\"" in task.kwargs["run"]


def test_build_agentflow_provider_returns_base_url_and_key_env():
    provider = build_agentflow_provider(
        name="agentflow-inference-qwen",
        base_url="https://inference.example/v1",
        api_key="test-key",
    )

    assert provider == {
        "name": "agentflow-inference-qwen",
        "base_url": "https://inference.example/v1",
        "api_key_env": "OPENAI_API_KEY",
        "wire_api": "openai-completions",
        "env": {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://inference.example/v1",
        },
    }


def test_resolve_default_image_id_uses_blackwell_dlami_for_aws_b200(monkeypatch):
    monkeypatch.setattr("agentflow.inference.skypilot.resolve_blackwell_image", lambda region: f"ami-{region}")

    assert resolve_default_image_id(parse_gpu_selector("aws:8xb200@us-east-2")) == "ami-us-east-2"
    assert resolve_default_image_id(parse_gpu_selector("aws:8x8xb200@us-east-2")) == "ami-us-east-2"


def test_resolve_default_image_id_skips_non_blackwell_or_non_aws(monkeypatch):
    def _unexpected(region):
        raise AssertionError("should not resolve AMI")

    monkeypatch.setattr("agentflow.inference.skypilot.resolve_blackwell_image", _unexpected)

    assert resolve_default_image_id(parse_gpu_selector("aws:1xl4@us-east-1")) is None
    assert resolve_default_image_id(parse_gpu_selector("gcp:8xb200@us-central1")) is None


@pytest.mark.parametrize("value", ["", "aws:", "aws:0xb200", "aws:8x0xb200", "aws:8x"])
def test_parse_gpu_selector_rejects_invalid_values(value: str):
    with pytest.raises(ValueError):
        parse_gpu_selector(value)
