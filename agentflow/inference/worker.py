from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Write one short sentence confirming AgentFlow batch inference is running."


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow SkyPilot batch inference worker")
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    requests = list(_load_requests(args.input, args.prompt))
    if args.engine == "vllm":
        results = _run_vllm(
            args.model_id,
            requests,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:
        results = _run_sglang(
            args.model_id,
            requests,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    _write_results(args.output, results)


def _load_requests(input_path: str | None, prompt: str | None) -> Iterator[dict[str, Any]]:
    if input_path is None:
        yield {"id": "smoke", "prompt": prompt or DEFAULT_PROMPT}
        return

    with Path(input_path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            prompt_value = payload.get("prompt")
            if not isinstance(prompt_value, str) or not prompt_value:
                raise ValueError(f"Input line {index + 1} is missing a non-empty `prompt` field")
            yield {
                "id": payload.get("id", str(index)),
                "prompt": prompt_value,
                "metadata": payload.get("metadata", {}),
            }


def _run_vllm(
    model_id: str,
    requests: list[dict[str, Any]],
    *,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
) -> Iterator[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    for batch in _chunks(requests, batch_size):
        outputs = llm.generate([request["prompt"] for request in batch], sampling_params)
        for request, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            yield _result_payload(request, text)


def _run_sglang(
    model_id: str,
    requests: list[dict[str, Any]],
    *,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
) -> Iterator[dict[str, Any]]:
    import sglang as sgl

    engine = sgl.Engine(model_path=model_id, tp_size=tensor_parallel_size)
    sampling_params = {"max_new_tokens": max_tokens, "temperature": temperature}
    try:
        for batch in _chunks(requests, batch_size):
            outputs = engine.generate(
                [request["prompt"] for request in batch],
                sampling_params=sampling_params,
            )
            for request, output in zip(batch, outputs):
                text = output.get("text", "") if isinstance(output, dict) else str(output)
                yield _result_payload(request, text)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


def _write_results(output_path: str, results: Iterable[dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            line = json.dumps(result, ensure_ascii=False)
            print(f"AGENTFLOW_INFERENCE_RESULT {line}", flush=True)
            handle.write(line + "\n")


def _chunks(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _result_payload(request: dict[str, Any], text: str) -> dict[str, Any]:
    payload = {
        "id": request["id"],
        "prompt": request["prompt"],
        "text": text,
    }
    metadata = request.get("metadata")
    if metadata:
        payload["metadata"] = metadata
    return payload


if __name__ == "__main__":
    main()
