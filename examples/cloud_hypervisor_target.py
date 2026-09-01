"""Credential-free smoke pipeline for the Cloud Hypervisor target.

Prepare a Cloud Hypervisor kernel and exported AgentFlow rootfs first, then set:

    AGENTFLOW_CH_KERNEL=/opt/agentflow-vm/vmlinux-x86_64
    AGENTFLOW_CH_ROOTFS=/opt/agentflow-vm/rootfs

The target defaults to no network device. Set ``AGENTFLOW_CH_TAP`` to attach a
pre-created TAP interface; guest address/routing should then be supplied in the
target configuration for that host network.
"""

from __future__ import annotations

import os

from agentflow import Graph, shell

KERNEL = os.environ.get(
    "AGENTFLOW_CH_KERNEL", ".agentflow/cloud-hypervisor/vmlinux-x86_64"
)
ROOTFS = os.environ.get("AGENTFLOW_CH_ROOTFS", ".agentflow/cloud-hypervisor/rootfs")
TAP = os.environ.get("AGENTFLOW_CH_TAP", "").strip()

network_policy: str | dict[str, object]
if TAP:
    network_policy = {
        "mode": "tap",
        "tap": TAP,
        "dhcp": True,
    }
else:
    network_policy = "none"

with Graph("cloud-hypervisor-smoke", working_dir="..") as dag:
    shell(
        task_id="vm_smoke",
        script=r"""
set -eu
for executable in agentflow codex claude kimi pi docker; do
    printf '%-10s %s\n' "$executable" "$(command -v "$executable")"
done
python3 -c 'import os, pwd; print("guest user", pwd.getpwuid(os.getuid()).pw_name, os.getuid(), os.getgid())'
test -f README.md
test -w "$HOME"
printf 'cloud-hypervisor target is ready\n'
""",
        target={
            "kind": "cloud_hypervisor",
            "kernel": KERNEL,
            "rootfs": ROOTFS,
            "workdir_read_only": True,
            "network_policy": network_policy,
        },
    )

print(dag.to_json())
