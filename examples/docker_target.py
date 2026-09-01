"""Smoke-test AgentFlow's structured Docker target in one of three modes.

Build the image first:

    docker build -t agentflow-agents:latest .

Then select a mode (``isolated`` is the default):

    agentflow run examples/docker_target.py --output summary
    AGENTFLOW_DOCKER_MODE=daemon agentflow run examples/docker_target.py --output summary
    AGENTFLOW_DOCKER_MODE=dind agentflow run examples/docker_target.py --output summary

The example uses a shell node so it can validate the image and Docker plumbing
without model-provider credentials. ``daemon`` grants the node root-equivalent
control over the host Docker daemon. ``dind`` runs the node as a privileged
container. Use both modes only with trusted code and inputs.
"""

from __future__ import annotations

import os

from agentflow import Graph, shell

MODE = os.environ.get("AGENTFLOW_DOCKER_MODE", "isolated").strip().lower()

targets: dict[str, dict[str, object]] = {
    "isolated": {
        "kind": "docker",
        "workdir_read_only": True,
        "mounts": [
            {"source": "./docs", "target": "/reference", "read_only": True},
        ],
        "network_policy": "none",
    },
    "daemon": {
        "kind": "docker",
        "workdir_read_only": True,
        "network_policy": "bridge",
        "mount_docker_daemon": True,
    },
    "dind": {
        "kind": "docker",
        "workdir_read_only": True,
        "network_policy": "bridge",
        "privileged": True,
        "dind": True,
    },
}

if MODE not in targets:
    choices = ", ".join(targets)
    raise SystemExit(f"AGENTFLOW_DOCKER_MODE must be one of: {choices}")

check_toolchain = r"""
set -eu
for executable in agentflow codex claude kimi pi docker dockerd; do
    path="$(command -v "$executable")"
    printf '%-10s %s\n' "$executable" "$path"
done
python3 -c 'import os, pwd; print("runtime user", pwd.getpwuid(os.getuid()).pw_name)'
ssh -G example.invalid >/dev/null
test -w "$HOME"
test -f README.md
test -f /reference/pipelines.md
printf 'isolated Docker target is ready\n'
"""

check_host_daemon = r"""
set -eu
docker version
docker info
test -n "${AGENTFLOW_HOST_WORKDIR:-}"
docker run --rm \
    --mount "type=bind,src=${AGENTFLOW_HOST_WORKDIR},dst=/host-workspace,readonly" \
    busybox:1.36 test -f /host-workspace/README.md
docker run --rm busybox:1.36 echo 'host-daemon sibling container is ready'
"""

check_dind = r"""
set -eu
test "$(id -u)" = "$AGENTFLOW_RUN_UID"
docker version
docker info
if grep -Eq ':(0947|0948) ' /proc/net/tcp /proc/net/tcp6; then
    echo 'DinD unexpectedly exposed TCP port 2375/2376' >&2
    exit 70
fi
docker run --rm busybox:1.36 echo 'nested Docker container is ready'
"""

scripts = {
    "isolated": check_toolchain,
    "daemon": check_host_daemon,
    "dind": check_dind,
}

with Graph("docker-target-smoke", working_dir="..") as dag:
    shell(
        task_id=f"docker_{MODE}",
        script=scripts[MODE],
        target=targets[MODE],
    )

print(dag.to_json())
