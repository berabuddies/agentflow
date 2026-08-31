#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
verify_async_codex.sh

Smoke-like validation for the async AgentFlow mainline using Codex only.

What it verifies:
- Detached submission: `agentflow run ... -d`
- Store-backed process view: `agentflow status <run_id>`
- PR11 process visibility: evolution progress rendered in status
- PR12 process visibility: optimization session / round events rendered in status

Required environment:
- OPENAI_API_KEY
- Optional if you use a custom gateway:
  - OPENAI_BASE_URL
  - AGENTFLOW_OPENAI_BASE_URL

Usage:
  bash scripts/verify_async_codex.sh
  bash scripts/verify_async_codex.sh /tmp/agentflow-async-verify

Outputs:
- Writes all generated pipelines, run ids, summaries, and json payloads under the chosen workdir.
- Prints the key artifact paths you should use for screenshots in the PR description.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="${1:-${REPO_ROOT}/.tmp/verify_async_codex}"
ARTIFACT_DIR="${WORKDIR}/artifacts"
PR11_WORKSPACE="${WORKDIR}/pr11_workspace"
PR11_SEED_REPO="${WORKDIR}/pr11_seed_repo"
PR12_WORKSPACE="${WORKDIR}/pr12_workspace"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

choose_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

AF() {
  "${PYTHON_BIN}" -m agentflow.cli "$@"
}

json_field() {
  local json_path="$1"
  local expr="$2"
  "${PYTHON_BIN}" - "$json_path" "$expr" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
expr = sys.argv[2]
data = json.loads(json_path.read_text(encoding="utf-8"))
value = eval(expr, {"__builtins__": {}}, {"data": data})
if isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}

wait_for_condition() {
  local run_id="$1"
  local output_json="$2"
  local python_expr="$3"
  local timeout_seconds="$4"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    AF status "$run_id" --output json-summary > "$output_json"
    if "${PYTHON_BIN}" - "$output_json" "$python_expr" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expr = sys.argv[2]
safe_globals = {
    "__builtins__": {},
    "len": len,
    "any": any,
    "all": all,
    "bool": bool,
}
ok = bool(eval(expr, safe_globals, {"data": payload}))
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi

    if (( "$(date +%s)" - start_ts >= timeout_seconds )); then
      echo "Timed out waiting for condition on run ${run_id}: ${python_expr}" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_terminal() {
  local run_id="$1"
  local output_json="$2"
  local timeout_seconds="$3"
  wait_for_condition \
    "$run_id" \
    "$output_json" \
    'data["status"] in {"completed", "failed", "cancelled"}' \
    "$timeout_seconds"
}

capture_latest_status() {
  local run_id="$1"
  local json_path="$2"
  local summary_path="$3"
  AF status "$run_id" --output json-summary > "$json_path"
  AF status "$run_id" --output summary > "$summary_path"
}

best_effort_terminal_snapshot() {
  local run_id="$1"
  local json_path="$2"
  local summary_path="$3"
  local timeout_seconds="$4"

  if wait_for_terminal "$run_id" "$json_path" "$timeout_seconds"; then
    AF status "$run_id" --output summary > "$summary_path"
    return 0
  fi

  capture_latest_status "$run_id" "$json_path" "$summary_path"
  return 0
}

rm -rf "$ARTIFACT_DIR" "$PR11_WORKSPACE" "$PR11_SEED_REPO" "$PR12_WORKSPACE" "${WORKDIR}/runs"
mkdir -p "$ARTIFACT_DIR" "$PR11_WORKSPACE" "$PR12_WORKSPACE"

export AGENTFLOW_RUNS_DIR="${WORKDIR}/runs"
export AGENTFLOW_DAEMON_HOST="${AGENTFLOW_DAEMON_HOST:-127.0.0.1}"
export AGENTFLOW_DAEMON_PORT="${AGENTFLOW_DAEMON_PORT:-$(choose_port)}"
export AGENTFLOW_DAEMON_METADATA_PATH="${AGENTFLOW_RUNS_DIR}/daemon.json"
VERIFY_LATEST_TIMEOUT_SECONDS="${VERIFY_LATEST_TIMEOUT_SECONDS:-15}"

echo "[verify] repo root: ${REPO_ROOT}"
echo "[verify] python: ${PYTHON_BIN}"
echo "[verify] runs dir: ${AGENTFLOW_RUNS_DIR}"
echo "[verify] daemon: ${AGENTFLOW_DAEMON_HOST}:${AGENTFLOW_DAEMON_PORT}"

echo "[verify] preparing PR11 local seed repo"
rm -rf "$PR11_SEED_REPO"
mkdir -p "$PR11_SEED_REPO"
git -C "$PR11_SEED_REPO" init -b main >/dev/null
cat > "${PR11_SEED_REPO}/README.md" <<'EOF'
# local-codex-smoke
EOF
git -C "$PR11_SEED_REPO" add README.md
git -c user.name="AgentFlow Verify" -c user.email="verify@example.invalid" -C "$PR11_SEED_REPO" commit -m "init local smoke repo" >/dev/null

mkdir -p "${PR11_WORKSPACE}/agent_tuner"
cat > "${PR11_WORKSPACE}/agent_tuner/local_codex_smoke.yaml" <<EOF
name: local_codex_smoke
base_agent: codex
repo_url: "${PR11_SEED_REPO}"
default_branch: main
build_command: mkdir -p .venv/bin && printf '#!/bin/sh\nprintf "local-codex-smoke\\n"\n' > .venv/bin/codex && chmod +x .venv/bin/codex
test_command: test -f README.md
smoke_command: "{executable} >/dev/null"
executable_path: .venv/bin/codex
max_attempts: 2
evolution_prompt: |
  Make the smallest coherent change you can based on the copied traces.
  Keep the repository valid and preserve a working local smoke flow.
EOF

cat > "${PR11_WORKSPACE}/pr11_evolve_demo.py" <<EOF
import sys

sys.path.insert(0, ${REPO_ROOT@Q})

from agentflow import Graph, codex, evolve

with Graph("pr11-evolve-demo", concurrency=1) as g:
    plan = codex(
        task_id="plan",
        prompt="Reply with exactly one short line: async validation trace.",
        model="gpt-5.2-codex",
    )
    evolve(
        plan,
        target="codex",
        optimizer="codex",
        tuned_agent="local_codex_smoke",
        task_id="evolve_codex",
    )

print(g.to_json())
EOF

cat > "${PR12_WORKSPACE}/pr12_optimize_demo.py" <<EOF
import sys

sys.path.insert(0, ${REPO_ROOT@Q})

from agentflow import Graph, python_node

with Graph("pr12-optimize-demo", concurrency=1, optimizer="codex", n_run=2) as g:
    prepare = python_node(task_id="prepare", code="print('prepare ok')")
    summarize = python_node(task_id="summarize", code="print('summarize ok')")
    prepare >> summarize

print(g.to_json())
EOF

echo "[verify] submitting PR11 evolve demo via detached run"
AF run "${PR11_WORKSPACE}/pr11_evolve_demo.py" -d --output json > "${ARTIFACT_DIR}/pr11.run.json"
PR11_RUN_ID="$(json_field "${ARTIFACT_DIR}/pr11.run.json" 'data["id"]')"
echo "[verify] PR11 run id: ${PR11_RUN_ID}"

wait_for_condition \
  "$PR11_RUN_ID" \
  "${ARTIFACT_DIR}/pr11.status.process.json" \
  'len(data.get("evolution_progress", [])) >= 3 and any(event.get("stage") == "optimizer" for event in data.get("evolution_progress", []))' \
  900
AF status "$PR11_RUN_ID" --output summary > "${ARTIFACT_DIR}/pr11.status.process.summary.txt"

best_effort_terminal_snapshot \
  "$PR11_RUN_ID" \
  "${ARTIFACT_DIR}/pr11.status.latest.json" \
  "${ARTIFACT_DIR}/pr11.status.latest.summary.txt" \
  "$VERIFY_LATEST_TIMEOUT_SECONDS"
PR11_LATEST_STATUS="$(json_field "${ARTIFACT_DIR}/pr11.status.latest.json" 'data["status"]')"

echo "[verify] submitting PR12 optimization demo via detached run"
AF run "${PR12_WORKSPACE}/pr12_optimize_demo.py" -d --output json > "${ARTIFACT_DIR}/pr12.run.json"
PR12_RUN_ID="$(json_field "${ARTIFACT_DIR}/pr12.run.json" 'data["id"]')"
echo "[verify] PR12 run id: ${PR12_RUN_ID}"

wait_for_condition \
  "$PR12_RUN_ID" \
  "${ARTIFACT_DIR}/pr12.status.process.json" \
  'bool(data.get("optimization")) and any(event.get("type") == "optimization_optimizer_started" for event in data.get("events", []))' \
  900
AF status "$PR12_RUN_ID" --output summary > "${ARTIFACT_DIR}/pr12.status.process.summary.txt"

best_effort_terminal_snapshot \
  "$PR12_RUN_ID" \
  "${ARTIFACT_DIR}/pr12.status.latest.json" \
  "${ARTIFACT_DIR}/pr12.status.latest.summary.txt" \
  "$VERIFY_LATEST_TIMEOUT_SECONDS"
PR12_LATEST_STATUS="$(json_field "${ARTIFACT_DIR}/pr12.status.latest.json" 'data["status"]')"

cat > "${ARTIFACT_DIR}/verification_report.txt" <<EOF
PR11 run id: ${PR11_RUN_ID}
PR11 latest status: ${PR11_LATEST_STATUS}
PR11 detached submission record: ${ARTIFACT_DIR}/pr11.run.json
PR11 process summary (screenshot this): ${ARTIFACT_DIR}/pr11.status.process.summary.txt
PR11 latest summary: ${ARTIFACT_DIR}/pr11.status.latest.summary.txt
PR11 latest json-summary: ${ARTIFACT_DIR}/pr11.status.latest.json

PR12 run id: ${PR12_RUN_ID}
PR12 latest status: ${PR12_LATEST_STATUS}
PR12 detached submission record: ${ARTIFACT_DIR}/pr12.run.json
PR12 process summary (screenshot this): ${ARTIFACT_DIR}/pr12.status.process.summary.txt
PR12 latest summary: ${ARTIFACT_DIR}/pr12.status.latest.summary.txt
PR12 latest json-summary: ${ARTIFACT_DIR}/pr12.status.latest.json
EOF

echo
echo "[verify] validation completed"
echo "[verify] PR11 latest status: ${PR11_LATEST_STATUS}"
echo "[verify] PR12 latest status: ${PR12_LATEST_STATUS}"
echo
echo "[verify] key artifacts"
cat "${ARTIFACT_DIR}/verification_report.txt"
