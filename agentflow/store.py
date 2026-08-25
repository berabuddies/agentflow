from __future__ import annotations

import json
import queue
import threading
from collections import defaultdict
from pathlib import Path, PurePath
from uuid import uuid4

from pydantic import ValidationError

from agentflow.specs import RunEvent, RunRecord
from agentflow.utils import ensure_dir


def _safe_path_segment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path segment")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {label} path segment")
    if PurePath(value).name != value:
        raise ValueError(f"invalid {label} path segment")
    return value


class RunStore:
    def __init__(self, base_dir: str | Path = ".agentflow/runs") -> None:
        self.base_dir = ensure_dir(Path(base_dir).expanduser())
        self._runs: dict[str, RunRecord] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._subscribers: defaultdict[str, set[queue.Queue[RunEvent]]] = defaultdict(set)
        self._events_cache: defaultdict[str, list[RunEvent]] = defaultdict(list)
        self._load_existing_runs()

    def _load_existing_runs(self) -> None:
        for run_file in sorted(self.base_dir.glob("*/run.json")):
            run_id = run_file.parent.name
            try:
                run = RunRecord.model_validate_json(run_file.read_text(encoding="utf-8"))
                self._runs[run_id] = run
                events_path = run_file.parent / "events.jsonl"
                if events_path.exists():
                    events = [
                        RunEvent.model_validate_json(line)
                        for line in events_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    self._events_cache[run_id] = events
            except (OSError, ValidationError, json.JSONDecodeError, KeyError):
                continue

    async def create_run(self, record: RunRecord | None = None) -> RunRecord:
        if record is None:
            raise ValueError("create_run requires a RunRecord")
        _safe_path_segment(record.id, "run_id")
        self._runs[record.id] = record
        await self.persist_run(record.id)
        return record

    def new_run_id(self) -> str:
        return uuid4().hex

    def run_dir(self, run_id: str) -> Path:
        return ensure_dir(self.base_dir / _safe_path_segment(run_id, "run_id"))

    def node_artifact_dir(self, run_id: str, node_id: str) -> Path:
        return ensure_dir(self.run_dir(run_id) / "artifacts" / _safe_path_segment(node_id, "node_id"))

    def artifact_path(self, run_id: str, node_id: str, name: str) -> Path:
        return self.node_artifact_dir(run_id, node_id) / _safe_path_segment(name, "artifact name")

    def cancel_request_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "cancel.requested"

    async def persist_run(self, run_id: str) -> None:
        record = self._runs[run_id]
        run_dir = self.run_dir(run_id)
        lock = self._locks[run_id]
        with lock:
            (run_dir / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    async def append_event(self, run_id: str, event: RunEvent) -> None:
        lock = self._locks[run_id]
        with lock:
            run_dir = self.run_dir(run_id)
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json())
                handle.write("\n")
            self._events_cache[run_id].append(event)
        for subscriber in list(self._subscribers[run_id]):
            subscriber.put_nowait(event)

    async def request_cancel(self, run_id: str) -> None:
        lock = self._locks[run_id]
        with lock:
            self.cancel_request_path(run_id).write_text("cancel\n", encoding="utf-8")

    def cancel_requested(self, run_id: str) -> bool:
        return self.cancel_request_path(run_id).exists()

    async def clear_cancel_request(self, run_id: str) -> None:
        lock = self._locks[run_id]
        with lock:
            self.cancel_request_path(run_id).unlink(missing_ok=True)

    async def append_artifact_text(self, run_id: str, node_id: str, name: str, content: str) -> None:
        path = self.artifact_path(run_id, node_id, name)
        lock = self._locks[run_id]
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)

    async def write_artifact_text(self, run_id: str, node_id: str, name: str, content: str) -> None:
        path = self.artifact_path(run_id, node_id, name)
        lock = self._locks[run_id]
        with lock:
            path.write_text(content, encoding="utf-8")

    async def write_artifact_json(self, run_id: str, node_id: str, name: str, payload: object) -> None:
        await self.write_artifact_text(run_id, node_id, name, json.dumps(payload, ensure_ascii=False, indent=2))

    def read_artifact_text(self, run_id: str, node_id: str, name: str) -> str:
        return self.artifact_path(run_id, node_id, name).read_text(encoding="utf-8")

    def get_run(self, run_id: str) -> RunRecord:
        return self._runs[run_id]

    def list_runs(self) -> list[RunRecord]:
        return sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)

    def get_events(self, run_id: str) -> list[RunEvent]:
        return list(self._events_cache[run_id])

    async def subscribe(self, run_id: str) -> queue.Queue[RunEvent]:
        subscriber: queue.Queue[RunEvent] = queue.Queue()
        self._subscribers[run_id].add(subscriber)
        return subscriber

    async def unsubscribe(self, run_id: str, subscriber: queue.Queue[RunEvent]) -> None:
        self._subscribers[run_id].discard(subscriber)
