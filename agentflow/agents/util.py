"""Adapters for utility nodes: python, shell, sync."""

from __future__ import annotations

import shlex

from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.specs import NodeSpec


class PythonAdapter:
    """Run a Python script. The prompt is the Python code."""

    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        return PreparedExecution(
            command=["python3", "-c", prompt],
            env=dict(node.env or {}),
            cwd=paths.target_workdir,
            trace_kind="python",
            runtime_files={},
            stdin=None,
        )


class ShellAdapter:
    """Run a shell command. The prompt is the bash script."""

    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        return PreparedExecution(
            command=["bash", "-c", prompt],
            env=dict(node.env or {}),
            cwd=paths.target_workdir,
            trace_kind="shell",
            runtime_files={},
            stdin=None,
        )


class SyncAdapter:
    """Sync local git repo to a remote target.

    The prompt is the sync mode: "repo" (just .git + stash) or "full" (entire directory).
    The target must be SSH, EC2, or ECS with a remote_workdir.
    """

    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        if node.target.kind not in {"ssh", "ec2", "ecs"}:
            raise ValueError(
                "sync nodes require an `ssh`, `ec2`, or `ecs` target; container, Docker, and Cloud "
                "Hypervisor targets already share the local pipeline workspace"
            )
        mode = prompt.strip().lower()
        if mode not in ("repo", "full"):
            mode = "full"

        target = node.target
        host = getattr(target, "host", None)
        username = getattr(target, "username", None)
        identity_file = getattr(target, "identity_file", None)
        remote_workdir = getattr(target, "remote_workdir", None) or str(paths.target_workdir)
        dest = f"{username}@{host}" if username else host

        ssh_opts = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
        if identity_file:
            ssh_opts += f" -i {shlex.quote(identity_file)}"

        source_dir = str(paths.host_workdir)

        if mode == "repo":
            # Sync .git directory + create a stash of current changes
            script = f"""
set -e
cd {shlex.quote(source_dir)}

# Stash current changes
STASH_OUTPUT=$(git stash create 2>/dev/null || true)

# Ensure remote dir exists
ssh {ssh_opts} {dest} "mkdir -p {shlex.quote(remote_workdir)}"

# Try rclone first, fall back to tar+ssh
if command -v rclone &>/dev/null; then
    rclone sync {shlex.quote(source_dir + '/.git')} :sftp:{remote_workdir}/.git \\
        --sftp-host={host} {f'--sftp-user={username}' if username else ''} \\
        {f'--sftp-key-file={identity_file}' if identity_file else ''} \\
        --transfers=8 --checkers=16 2>&1
else
    tar cf - .git | ssh {ssh_opts} {dest} "cd {shlex.quote(remote_workdir)} && tar xf -"
fi

# Checkout and apply stash on remote
ssh {ssh_opts} {dest} "cd {shlex.quote(remote_workdir)} && git checkout -f HEAD"
if [ -n "$STASH_OUTPUT" ]; then
    git stash show -p $STASH_OUTPUT | ssh {ssh_opts} {dest} \\
        "cd {shlex.quote(remote_workdir)} && git apply --allow-empty 2>/dev/null || true"
fi

echo "SYNC_OK mode=repo"
"""
        else:
            # Full sync: entire directory
            script = f"""
set -e
cd {shlex.quote(source_dir)}

# Ensure remote dir exists
ssh {ssh_opts} {dest} "mkdir -p {shlex.quote(remote_workdir)}"

# Try rclone first, fall back to tar+ssh
if command -v rclone &>/dev/null; then
    rclone sync {shlex.quote(source_dir)}/ :sftp:{remote_workdir}/ \\
        --sftp-host={host} {f'--sftp-user={username}' if username else ''} \\
        {f'--sftp-key-file={identity_file}' if identity_file else ''} \\
        --exclude='.agentflow/**' --exclude='node_modules/**' --exclude='.venv/**' \\
        --transfers=8 --checkers=16 2>&1
else
    tar cf - --exclude='.agentflow' --exclude='node_modules' --exclude='.venv' . | \\
        ssh {ssh_opts} {dest} "cd {shlex.quote(remote_workdir)} && tar xf -"
fi

echo "SYNC_OK mode=full"
"""

        return PreparedExecution(
            command=["bash", "-c", script],
            env=dict(node.env or {}),
            cwd=str(paths.host_workdir),
            trace_kind="sync",
            runtime_files={},
            stdin=None,
        )
