# Pipeline Reference

Pipeline authoring details, execution targets, and per-agent launch behavior.

## Python DAG

```python
from agentflow import DAG, claude, codex, kimi

with DAG("demo", working_dir=".", concurrency=3) as dag:
    plan = codex(task_id="plan", prompt="Inspect the repo and plan the work.")
    implement = claude(
        task_id="implement",
        prompt="Implement the plan:\n\n{{ nodes.plan.output }}",
        tools="read_write",
    )
    review = kimi(
        task_id="review",
        prompt="Review the plan:\n\n{{ nodes.plan.output }}",
        capture="trace",
    )
    merge = codex(
        task_id="merge",
        prompt="Merge the implementation and review outputs.",
    )

    plan >> [implement, review]
    [implement, review] >> merge

spec = dag.to_spec()
```

Use `fanout(node, source)` to fan a node into parallel copies and `merge(node, source, by=...|size=...)` to reduce them.
`DAG(...)` also accepts `fail_fast`, `node_defaults`, `agent_defaults`, and `local_target_defaults`.
Use `dag.to_json()` to serialize a compact runnable pipeline, `dag.to_payload()` for the raw object structure, and `dag.to_spec()` for the fully expanded in-memory pipeline object.

See `examples/airflow_like.py` for the small static DAG. `examples/airflow_like_fuzz_batched.py` and `examples/airflow_like_fuzz_grouped.py` are advanced fanout examples.

## Pipeline schema

Each node supports:

- `agent`: `codex`, `claude`, `kimi`, or `pi`; utility nodes also use `python`, `shell`, and `sync`
- `fanout`: `count`, `values`, `matrix`, `group_by`, or `batches`, plus optional `as`, `derive`, and matrix-only `include` / `exclude`
- `schedule`: optional periodic execution for local nodes with `every_seconds`, `until_fanout_settles_from`, and optional `actuation`
- `model`: any model string understood by the backend
- `provider`: a string or a structured provider config with `base_url`, `api_key_env`, headers, and env
- `tools`: `read_only` or `read_write`
- `repo_instructions_mode`: `inherit` (default) or `ignore` for agent CLIs that should not absorb repo-local instruction files such as `AGENTS.md`, `CLAUDE.md`, or project skills
- `mcps`: a list of MCP server definitions
- `skills`: a list of local skill paths or names
- `target`: `local`, `docker`, `container`, `ssh`, `ec2`, or `ecs`
- local target fields: `cwd`, `bootstrap`, `shell`, `shell_login`, `shell_interactive`, and `shell_init`
- `capture`: `final` or `trace`
- `retries` and `retry_backoff_seconds`
- `success_criteria`: output or filesystem checks evaluated after execution

Skill entries are resolved from the pipeline `working_dir`. You can point `skills:` at a plain file, a `.md` file, a home-relative path such as `~/.codex/skills/release-skill`, or a directory that contains `SKILL.md`.

Top-level pipeline controls include:

- `concurrency`: max parallel nodes within a run
- `fail_fast`: skip downstream work after the first failed node
- `node_defaults`: shared node fields merged into every node before validation
- `agent_defaults`: agent-specific shared node fields keyed by `codex`, `claude`, or `kimi`
- `optimizer`: optional optimizer backend, one of `codex`, `claude`, or `kimi`
- `n_run`: optional integer; when `> 1`, runs optimization rounds before execution

`node_defaults` is the pipeline-wide baseline. `agent_defaults` is the agent-specific override layer. Explicit node values always win.

```python
DAG(
    "demo",
    node_defaults={
        "agent": "codex",
        "tools": "read_only",
        "capture": "final",
    },
    agent_defaults={
        "codex": {
            "model": "gpt-5-codex",
            "retries": 1,
            "retry_backoff_seconds": 1,
            "extra_args": ["--search", "-c", 'model_reasoning_effort="high"'],
        }
    },
)
```

## Graph optimization rounds

Set top-level `optimizer` and `n_run` to run optimization rounds over the graph before execution. When `n_run > 1`, AgentFlow runs per-round optimization behavior and writes artifacts under `.agentflow/runs/<run_id>/optimization/round-XXX/`:

- `pipeline.original.py`
- `pipeline.edited.py`
- `graph_report.json`
- `optimizer-prompt.txt`
- `optimizer-result.json`
- `optimizer-validation.json`

Validation safeguards ensure the edited pipeline still loads and matches the pipeline schema. These checks only validate loader and schema correctness; they do not guarantee semantic improvement or better runtime results.

## Fan-out and merge

Use `fanout()` when a DAG needs many nearly identical nodes. Use `merge()` to reduce them. AgentFlow expands those nodes into a concrete DAG before validation and execution.

```python
from agentflow import DAG, codex, fanout, merge

with DAG("sweep-demo", concurrency=8) as dag:
    review = fanout(
        codex(task_id="review", prompt="Shard {{ item.number }} of {{ item.count }}."),
        8,
    )
    final = codex(
        task_id="merge",
        prompt="{% for s in fanouts.review.nodes %}{{ s.output }}\n{% endfor %}",
    )
    review >> final
```

### `item` shape (fanout)

Every expanded copy gets an `item` template variable:

| Field | Type | Example |
| --- | --- | --- |
| `item.index` | int | 0, 1, 2, ... |
| `item.number` | int | 1, 2, 3, ... (1-indexed) |
| `item.count` | int | total copies |
| `item.suffix` | str | "0", "01", "001" (zero-padded) |
| `item.node_id` | str | "review_001" |
| `item.value` | Any | the raw iteration value |
| `item.<key>` | Any | dict keys from value are lifted (e.g. `item.target`) |
| `item.<key>` | Any | keys from `derive={}` are added |

### `item` shape (merge)

Reducer nodes get everything above plus:

| Field | Type | Description |
| --- | --- | --- |
| `item.source_group` | str | task_id of the fanout being reduced |
| `item.source_count` | int | total members in the source fanout |
| `item.member_ids` | list | node IDs of members in this group/batch |
| `item.members` | list | full member objects |
| `item.size` | int | members in this group/batch |

With `size=` (batches): `item.start_number`, `item.end_number`, `item.start_index`, `item.end_index`.

With `by=` (groups): the grouping field values are on `item` directly (e.g. `item.target`).

At runtime, `item.scope` provides aggregated results:

| Field | Type | Description |
| --- | --- | --- |
| `item.scope.ids` | list | member node IDs |
| `item.scope.size` | int | count |
| `item.scope.nodes` | list | member objects with status/output |
| `item.scope.outputs` | list | output strings |
| `item.scope.summary` | dict | {total, completed, failed, with_output, ...} |
| `item.scope.with_output` | subset | members with non-empty output |
| `item.scope.without_output` | subset | members with empty output |

### Source types

`fanout(node, source)` dispatches on type:

- `int` -- count: `fanout(node, 128)`
- `list` -- values: `fanout(node, [{"repo": "api"}, {"repo": "billing"}])`
- `dict` -- matrix (cartesian product): `fanout(node, {"repo": [...], "check": [...]})`

Matrix supports `include=` and `exclude=` for curated adjustments.

### Reducer modes

`merge(node, source_node)` requires exactly one of `by=` or `size=`:

- `by=["field", ...]` -- one reducer per unique field combination
- `size=N` -- one reducer per N-item batch

### Derived fields

Add computed fields with `derive=`:

```python
fanout(
    codex(task_id="review", prompt="Work in {{ item.workspace }}"),
    {"repo": [{"name": "api"}, {"name": "billing"}], "check": [{"kind": "security"}]},
    derive={
        "label": "{{ item.name }}/{{ item.kind }}",
        "workspace": "agents/{{ item.name }}_{{ item.kind }}_{{ item.suffix }}",
    },
)
```

### Expansion rules

- A fan-out node expands to `review_0` through `review_7` (zero-padded when needed).
- Dict keys from values are lifted onto `item` (e.g. `item.target`).
- Matrix expands the cartesian product in declaration order.
- `merge` with `by=` creates one reducer per unique field combination.
- `merge` with `size=` partitions into fixed-size batches.
- A downstream `>>` dependency on a fanout node expands to all its members.
- `derive` fields render in declaration order after base expansion.

## Periodic nodes

Use `schedule` when one node should re-run on a fixed interval inside the same pipeline execution.

```python
monitor = codex(
    task_id="monitor",
    schedule={
        "every_seconds": 600,
        "until_fanout_settles_from": "worker",
        "actuation": "output_json",
    },
    prompt=(
        "Tick {{ item.tick_number }}\n"
        "{% for shard in fanouts.worker.nodes %}\n"
        "- {{ shard.id }} stdout={{ shard.artifacts.stdout_log }}\n"
        "{% endfor %}"
    ),
)
```

Periodic nodes are local-only in v1. They stop automatically once the watched fanout group reaches terminal state.

With `actuation: output_json`, the node may emit a JSON envelope with an `analysis` string plus `cancel` / `rerun` actions for members of the watched fanout group.

Runtime numeric settings are validated up front: `concurrency` must be at least `1`, `timeout_seconds` must be greater than `0`, and both `retries` and `retry_backoff_seconds` must be non-negative.

MCP definitions are also validated before launch: `stdio` servers require `command` and reject HTTP-only fields such as `url`, `streamable_http` servers require `url` and reject stdio-only fields such as `command`, and MCP server names must be unique within a node.

`repo_instructions_mode: ignore` is a generic AgentFlow switch with agent-specific implementations. The current adapters use the same high-level pattern: start the agent from an isolated runtime directory, keep the target repo accessible via an explicit allowlist flag such as `--add-dir`, and disable or override repo-local instruction discovery where the underlying CLI supports it. When you enable this mode, write prompts that use absolute paths or explicitly tell the agent to `cd` into the repository before running shell commands.

Built-in provider shorthands:

- `codex`: `openai`
- `claude`: `anthropic`, `kimi`, `minimax`, `minimax-cn`
- `kimi`: `kimi`, `moonshot`, `moonshot-ai`
- `pi`: `minimax`, `minimax-cn`

`provider: kimi` is intentionally rejected on `codex` nodes. Codex requires an OpenAI Responses API backend, and Kimi's public endpoints do not expose `/responses`.

`provider: minimax` uses the global Anthropic-compatible endpoint (`https://api.minimax.io/anthropic`) for `claude` nodes and the global OpenAI-compatible chat-completions endpoint (`https://api.minimax.io/v1`) for `pi` nodes. `provider: minimax-cn` uses the corresponding China endpoints (`https://api.minimaxi.com/anthropic` and `https://api.minimaxi.com/v1`). For Pi, both aliases materialize a scoped `models.json` with declarations for `MiniMax-M3` and `MiniMax-M2.7`, including context windows, supported Pi input types, reasoning support, and token rates. Pi's current model schema cannot represent MiniMax-M3 video input or an unavailable cache-write rate, so the scoped declaration advertises text/image input and uses a zero cache-write rate. Both aliases are rejected on `codex` nodes because MiniMax exposes chat completions rather than the Responses API. Use a full `ProviderConfig` for any other custom endpoint.

When both `provider.env` and `node.env` define the same variable, `node.env` wins. For Claude-compatible Kimi setups, `doctor` and `inspect` also recognize providers that set `ANTHROPIC_BASE_URL=https://api.kimi.com/coding/` in `provider.env` even when `provider.base_url` is omitted.

## Execution targets

### Local

Runs the prepared agent command directly on the host. Set `target.shell` to wrap the command in a specific shell, such as `bash -lc`. If you provide a shell name without an explicit command flag, AgentFlow uses `-c` by default. Opt into startup file loading with `shell_login: true` and `shell_interactive: true`.

`target.cwd` controls the local node working directory. Absolute paths are used as-is; relative paths are resolved from the pipeline `working_dir`. AgentFlow creates that directory right before launch when it does not already exist.

The local bootstrap fields `shell_login`, `shell_interactive`, and `shell_init` require `target.shell`. For the common Kimi helper case, `target.bootstrap: kimi` expands to the same `bash` + login + interactive + `shell_init` setup automatically.

```python
target={"bootstrap": "kimi"}
```

When most local nodes share the same shell bootstrap, move that block to top-level `local_target_defaults` and only override the nodes that differ.

```python
DAG(
    "demo",
    local_target_defaults={"bootstrap": "kimi"},
)
```

If one local node should not inherit the shared bootstrap, set `target={"bootstrap": None}` on that node.
`shell_init` is treated as a bootstrap prerequisite: if it exits non-zero, AgentFlow does not launch the wrapped agent command.

### Docker

The Docker target runs a node in a local Docker container and exposes mounts,
network attachment, privilege, host-daemon access, and Docker-in-Docker as
structured policy. Build the bundled image before using the default target:

```bash
docker build -t agentflow-agents:latest .
```

The image contains AgentFlow, Codex, Claude, Kimi, Pi, the Docker CLI, and a
Docker daemon. It contains the agent programs, not credentials for your model
providers. Configure provider credentials in the node environment or mount
only the required credential files/directories.

```python
from agentflow import Graph, codex

with Graph("docker-demo", working_dir=".") as dag:
    codex(
        task_id="review",
        prompt="Read the repository and summarize its architecture.",
        tools="read_only",
        target={
            "kind": "docker",
            # Optional: this is the default image.
            "image": "agentflow-agents:latest",
            "workdir_read_only": True,
            "mounts": [
                {"source": "./docs", "target": "/reference", "read_only": True},
            ],
            "network_policy": "bridge",
        },
    )

print(dag.to_json())
```

#### Docker target fields

| Field | Default | Description |
| --- | --- | --- |
| `kind` | required | Set to `docker`. |
| `image` | `agentflow-agents:latest` | Image used for the node. |
| `engine` | `docker` | Docker CLI executable. Context auto-discovery follows Docker's config layout; other engines are not yet compatibility-tested. |
| `workdir_read_only` | `false` | Mount the pipeline workspace read-only. |
| `user` | `host` | Run the agent command as the invoking host UID:GID to avoid root-owned bind-mount output. Set a Docker user/group value explicitly, or `null` to use the image default. A custom identity must already be able to access the configured binds and should configure its own writable `HOME`. DinD's daemon always starts as root. |
| `inherit_credentials` | `false` | Permit adapters to expose selected host credential/config files as read-only runtime bind mounts. The Codex adapter uses this for `~/.codex/config.toml` and `auth.json`; see the credential notes below. |
| `mounts` | `[]` | Additional bind mounts with `source`, absolute container `target`, and optional `read_only`. |
| `network_policy` | `bridge` | `none`, `bridge`, `host`, or a custom-network object. |
| `privileged` | `false` | Pass `--privileged` to the container. This substantially weakens host isolation. |
| `mount_docker_daemon` | `false` | Bind-mount a host Docker daemon socket at `/var/run/docker.sock` in the container. |
| `docker_daemon_socket` | auto | Explicit absolute host socket path used with `mount_docker_daemon`; otherwise a Unix `DOCKER_HOST`, the active named Docker context's Unix endpoint, or `/var/run/docker.sock` is used. |
| `dind` | `false` | Start the image's own Docker daemon as root, then run the agent command as `user` (host UID:GID by default). Requires `privileged: true`, is mutually exclusive with `mount_docker_daemon`, and cannot use `--read-only`. |
| `workdir_mount` | `/workspace` | Container path for the pipeline workspace. |
| `runtime_mount` | `/agentflow-runtime` | Container path for the per-node runtime files. |
| `app_mount` | `null` | Optional read-only container path for a local AgentFlow development checkout. The bundled image already has AgentFlow installed. |
| `entrypoint` | image default | Override the image entrypoint. Do not override it for bundled-image DinD runs because the default entrypoint starts the daemon. |
| `extra_args` | `[]` | Allowlisted resource/runtime `docker run` arguments. Isolation-sensitive and positional arguments are rejected. |

`mount_docker_socket` is accepted as a compatibility spelling for
`mount_docker_daemon`. New pipelines should use `mount_docker_daemon`.

#### Automatic and explicit mounts

Every Docker node gets two AgentFlow-managed bind mounts by default, plus an
optional development-code mount:

| Host content | Container path | Access |
| --- | --- | --- |
| Pipeline `working_dir` | `/workspace` (`workdir_mount`) | Read/write, or read-only with `workdir_read_only: true` |
| Per-run, per-node runtime directory | `/agentflow-runtime` (`runtime_mount`) | Read/write so adapters can materialize provider, MCP, and trace-support files |
| Local AgentFlow app (optional) | `app_mount` | Read-only; disabled by default because the bundled image already contains AgentFlow |

By default the agent process runs with the invoking host UID:GID and uses a
writable home below the per-node runtime directory. This prevents ordinary
Docker nodes from leaving root-owned files in bind mounts on Linux. The bundled
entrypoint supplies an NSS identity for numeric host UIDs so tools such as SSH
can resolve the user. Set `user` only when the selected image requires another
identity; an explicit non-root identity that differs from the invoking UID must
already be able to traverse the runtime bind, and it cannot read AgentFlow's
host-owned mode-`0600` generated adapter files. DinD starts its daemon as root, then the bundled entrypoint drops the
agent command back to the host UID:GID; `user: null` keeps the command as the
image-default root user.

The workspace is the container working directory unless an adapter deliberately
uses its runtime directory for isolation. Workspace edits therefore persist on
the host when `workdir_read_only` is false. The runtime directory stays writable
even when the workspace is read-only. Docker nodes share the configured host
workspace; `use_worktree` currently isolates local targets only, so concurrent
writable Docker nodes must coordinate their own file changes.

Additional bind mounts use this shape:

```python
target={
    "kind": "docker",
    "mounts": [
        {"source": "./fixtures", "target": "/inputs", "read_only": True},
        {"source": "/var/tmp/output", "target": "/outputs", "read_only": False},
    ],
}
```

A relative `source` is resolved from the pipeline `working_dir`, not from the
caller's current shell directory. Home-relative and absolute sources are also
accepted. Container targets must be absolute and cannot duplicate or shadow
AgentFlow-managed mount targets. Ancestor/descendant overlap is also rejected
(for example, `/workspace/cache` overlaps the managed `/workspace`); mount
additional content at a separate container root such as `/inputs` instead.
A read-only bind protects that bind mount only; it does not compensate for
`privileged` mode or access to the host Docker daemon. AgentFlow rejects a
writable explicit mount whose canonical host source overlaps a managed
read-only workspace or app mount, so the same files cannot be silently
re-exposed read/write at another container path.

`mount_docker_daemon: false` also prevents an explicit mount from exposing the
active Docker socket (or one of its parent directories) at another path. Other
explicit mounts remain trusted configuration: review every host source, since
the container can read or modify exactly what the configured access permits.

`extra_args` intentionally accepts only resource and process-lifecycle options:
CPU/memory/pid limits, ulimits, labels, hostname, platform/pull policy, logging,
shared-memory size, stop behavior, `--init`, `--read-only`, and
`--oom-kill-disable`. Use the structured fields for mounts, networking,
entrypoint, user identity, privilege, and Docker-daemon access.

#### Credentials and generated runtime files

Docker targets do not inherit the caller's ambient API-key environment or CLI
home directories. Configure model credentials explicitly in `node.env` or
`provider.env`; AgentFlow forwards sensitive values through the Docker CLI's
temporary `--env-file` rather than embedding values in `docker run`'s argument
list or altering the Docker CLI process environment. All values are hidden in
inspection output; the mode-`0600` temporary env file is deleted after the
container exits. Docker environment values must be single-line—mount PEMs and
other multiline credentials/configuration as files. Generated adapter
configuration and other credential-bearing runtime files are also created with
mode `0600` below a mode-`0700` per-node runtime directory.

`inherit_credentials: true` is a separate, explicit trust decision. It permits
an adapter to request read-only bind mounts of selected host files; currently
the Codex adapter uses it to make an existing `~/.codex/config.toml` and
`~/.codex/auth.json` available in its scoped `CODEX_HOME`. Inspection reports a
warning whenever this is enabled. Other CLI homes are not mounted
automatically; use an explicit mount at a non-managed target and set that CLI's
home/config environment deliberately if needed. Credential inheritance does
not make an untrusted image or prompt safe.

#### Network policy

`network_policy` can be a shorthand string:

```python
target={"kind": "docker", "network_policy": "none"}
target={"kind": "docker", "network_policy": "bridge"}
target={"kind": "docker", "network_policy": "host"}
```

| Mode | Container connectivity |
| --- | --- |
| `none` | Only loopback; no model API or internet access. Useful for deterministic local/shell work. |
| `bridge` | Docker's normal bridge connectivity, including outbound access allowed by the host. This is the default. |
| `host` | Shares the host network namespace where supported. This removes Docker's normal network boundary. |
| `custom` | Attaches to one pre-created, operator-managed Docker network. |

For a custom network, provide its name and optional Docker network settings:

```bash
docker network create agentflow-egress
# Connect/configure your egress proxy or firewall on this network separately.
```

```python
target={
    "kind": "docker",
    "network_policy": {
        "mode": "custom",
        "name": "agentflow-egress",
        "aliases": ["review-agent"],
        "dns": ["10.20.0.53"],
        "add_hosts": {"mirror.internal": "10.20.0.10"},
    },
}
```

The network name itself can also be used as shorthand:
`"network_policy": "agentflow-egress"`. `aliases` are valid only for a custom
network. `dns`, `add_hosts`, and aliases change name resolution or addressing;
they are not destination allowlists. A normal custom bridge still permits the
egress allowed by Docker and the host. For domain/IP/port-level restrictions,
attach the custom network to an egress proxy or firewall and enforce the policy
there (or use an internal network when no external access is required).

#### Host daemon versus Docker-in-Docker

To let a node launch sibling containers through the host daemon:

```python
target={
    "kind": "docker",
    "mount_docker_daemon": True,
    # Optional override; must be an absolute host path:
    # "docker_daemon_socket": "/run/user/1000/docker.sock",
}
```

The socket is mounted read/write at `/var/run/docker.sock`, and AgentFlow sets
`DOCKER_HOST` accordingly. Access to a Docker daemon socket is effectively
root-level control of that daemon's host: a process can start privileged
containers and bind-mount arbitrary host paths. A read-only workspace and
`privileged: false` do not mitigate that authority. Use this only with trusted
agents, prompts, images, and inputs.

Docker targets use local host bind mounts, so their Docker CLI must itself use a
local Unix endpoint. AgentFlow resolves `DOCKER_HOST` first, then a named active
context from the Docker config; TCP, SSH, and Windows named-pipe endpoints are
rejected. Set `docker_daemon_socket` explicitly when auto-detection cannot
identify the intended socket. Execution checks that the resolved daemon-mount
path exists and is a Unix socket before launching the container.

Bind sources passed to a sibling `docker run` are interpreted by the host
daemon, not by the agent container. In host-daemon mode AgentFlow therefore
sets `AGENTFLOW_HOST_WORKDIR`, `AGENTFLOW_HOST_RUNTIME_DIR`, and
`AGENTFLOW_DOCKER_CONTAINER_NAME`. Use the first two values when a sibling
container intentionally needs the same host workspace/runtime content, for
example:

```bash
docker run --rm \
  --mount "type=bind,src=${AGENTFLOW_HOST_WORKDIR},dst=/src,readonly" \
  busybox:1.36 ls /src
```

To run an isolated daemon inside the agent container instead:

```python
target={
    "kind": "docker",
    "privileged": True,
    "dind": True,
}
```

DinD uses the bundled image entrypoint to start `dockerd` before AgentFlow's
prepared command. It requires `privileged: true`; validation rejects DinD
without it and rejects combining `dind` with `mount_docker_daemon`. Privileged
mode still exposes host devices and powerful kernel interfaces, so DinD is not
a security boundary for untrusted work. The daemon must write its own runtime
and storage state, so `--read-only` is rejected for DinD.
Custom images used with `dind: true` must preserve a compatible entrypoint that
implements the documented `AGENTFLOW_DIND` and `AGENTFLOW_RUN_UID/GID`
contract; otherwise AgentFlow cannot start or safely drop privileges around the
nested daemon.

The bundled smoke example exercises the three common launch modes without
calling a model provider:

```bash
# Isolated/offline, read-only workspace; checks installed CLIs.
agentflow run examples/docker_target.py --output summary

# Host Docker socket (root-equivalent authority over the host daemon).
AGENTFLOW_DOCKER_MODE=daemon agentflow run examples/docker_target.py --output summary

# Privileged Docker-in-Docker using an independent nested daemon.
AGENTFLOW_DOCKER_MODE=dind agentflow run examples/docker_target.py --output summary
```

#### Legacy `container` compatibility

Existing `kind: "container"` pipelines remain supported and keep their prior
behavior: `image` is required, and AgentFlow wraps the node command in
`docker run` with the workspace, runtime directory, and app mounted. No
migration is required for those pipelines.

Use `kind: "docker"` for new pipelines that need the default all-agent image,
structured mounts or network policy, read-only workspaces, privilege control,
host-daemon mounting, or DinD. The two kinds remain distinct so the stricter
Docker-target validation does not silently change legacy container launches.

### Cloud Hypervisor

`kind: "cloud_hypervisor"` launches one ephemeral KVM virtual machine per node
with [Cloud Hypervisor](https://github.com/cloud-hypervisor/cloud-hypervisor).
The target uses direct kernel boot and a read-only virtio-fs root filesystem,
not a mutable VM disk. The host sends the prepared command over virtio-vsock;
the guest agent mounts the workspace/runtime shares and multiplexes stdout and
stderr back over the same connection. It does not depend on SSH, an IP address,
or a guest login account.

#### Host and guest prerequisites

The execution host must be Linux and provide:

- read/write access to `/dev/kvm`;
- a `cloud-hypervisor` executable;
- a current [Rust `virtiofsd`](https://gitlab.com/virtio-fs/virtiofsd)
  supporting `--readonly`, `--translate-uid`, and `--translate-gid`;
- a Cloud Hypervisor-compatible kernel with built-in virtio-fs and vsock
  support; and
- an exported rootfs containing
  `/usr/local/bin/agentflow-cloud-hypervisor-init` and the desired agent CLIs.

The bundled Docker image contains the guest init, guest agent, Codex, Claude,
Kimi, Pi, Docker CLI, Python, NSS wrapper, and common shell tooling. Build and
export it without preserving container root ownership:

```bash
docker build -t agentflow-agents:latest .
mkdir -p .agentflow/cloud-hypervisor
cloud_hypervisor/export-rootfs.sh \
  agentflow-agents:latest .agentflow/cloud-hypervisor/rootfs
```

For example, download the x86-64 kernel published by the Cloud Hypervisor
project's kernel repository:

```bash
curl -fL \
  https://github.com/cloud-hypervisor/linux/releases/download/ch-release-v6.16.9-20260508/vmlinux-x86_64 \
  -o .agentflow/cloud-hypervisor/vmlinux-x86_64
```

The kernel and rootfs architecture must match the host architecture. The
current target is a Linux/KVM direct-kernel transport; it does not implement
UEFI/disk-image boot or macOS virtualization.

#### Configuration

```python
target={
    "kind": "cloud_hypervisor",
    "kernel": ".agentflow/cloud-hypervisor/vmlinux-x86_64",
    "rootfs": ".agentflow/cloud-hypervisor/rootfs",
    "cpus": 4,
    "memory_mib": 8192,
    "workdir_read_only": True,
    "mounts": [
        {"source": "./fixtures", "target": "/inputs", "read_only": True},
    ],
    "network_policy": "none",
}
```

| Field | Default | Description |
| --- | --- | --- |
| `kind` | required | Set to `cloud_hypervisor`. |
| `kernel` | required | Host path to a direct-boot Cloud Hypervisor kernel. Relative paths resolve from pipeline `working_dir`. |
| `rootfs` | required | Host directory exported as the immutable virtio-fs guest root. Relative paths resolve from pipeline `working_dir`. |
| `binary` | `cloud-hypervisor` | Host Cloud Hypervisor executable. |
| `virtiofsd` | `virtiofsd` | Host Rust virtiofsd executable. |
| `cpus` | `2` | Boot vCPU count. |
| `memory_mib` | `4096` | Guest memory in MiB. Shared memory is always enabled because virtio-fs requires it. |
| `workdir_mount` | `/workspace` | Guest path for the pipeline workspace. |
| `runtime_mount` | `/agentflow-runtime` | Guest path for private per-node runtime files and HOME. |
| `app_mount` | `null` | Optional read-only guest path for the host AgentFlow source tree. The exported rootfs already contains AgentFlow. |
| `workdir_read_only` | `false` | Export the host workspace through read-only virtiofsd and mount it read-only in the guest. |
| `mounts` | `[]` | Additional directory shares with host `source`, absolute guest `target`, and `read_only` (default `true`). |
| `user` | `host` | Run the agent command under the invoking host UID:GID. Also accepts `root`, numeric `UID[:GID]`, or `null` for root. virtiofsd maps that guest identity back to the invoking host identity. |
| `inherit_credentials` | `false` | Permit adapters to copy selected host credential/config files into the private runtime share. Codex uses this for its config/login files. |
| `network_policy` | `none` | No NIC, or a structured TAP attachment described below. |
| `guest_agent_port` | `4050` | AF_VSOCK port used by the preinstalled guest agent. |
| `vsock_cid` | derived | Optional explicit CID. The default is deterministically derived from the per-node runtime path. |
| `boot_timeout_seconds` | `60` | Maximum wait for virtiofsd, VM boot, and guest-agent readiness. The node timeout still bounds the entire operation. |
| `shutdown_timeout_seconds` | `5` | Grace period before force-killing VMM/backend processes. |
| `init_path` | `/usr/local/bin/agentflow-cloud-hypervisor-init` | PID 1 path in the exported rootfs. |
| `nss_wrapper_path` | `/usr/lib/libnss_wrapper.so` | Guest library used to resolve arbitrary numeric host UIDs. Set `null` only when the rootfs already resolves the selected user. |
| `kernel_args` | `[]` | Additional single kernel arguments. Root filesystem, init, and protocol arguments cannot be overridden. |
| `seccomp` | `true` | Cloud Hypervisor seccomp mode: `true`, `log`, or `false`. Disabling it weakens the host-side VMM sandbox. |

#### Filesystem and identity model

AgentFlow starts one independently sandboxed virtiofsd process for each share:

| Host directory | Guest path | Access |
| --- | --- | --- |
| Exported all-agent rootfs | `/` | Always daemon-enforced read-only |
| Pipeline `working_dir` | `workdir_mount` | Read/write unless `workdir_read_only: true` |
| Per-run/per-node runtime | `runtime_mount` | Read/write |
| Local AgentFlow source | `app_mount` | Optional and read-only |
| Each configured `mounts[]` source | Configured target | Read-only by default |

All guest mount points must already exist in the read-only rootfs. The bundled
image creates `/workspace`, `/agentflow-runtime`, `/agentflow-app`, `/inputs`,
`/outputs`, and `/reference`; custom targets should be created while building
the image. Target paths cannot duplicate or be ancestors/descendants of one
another and cannot overlap `/dev`, `/proc`, `/run`, or `/sys`.

Read-only access is enforced twice: virtiofsd receives `--readonly`, and the
guest mount uses `ro`. Canonical host-path checks prevent a read-only workspace,
app, rootfs, or additional share from being reachable through an overlapping
writable export. Likewise, a custom read-only share cannot alias the writable
workspace or runtime under a second path.

If the exported rootfs lives below the pipeline workspace (as in the relative
quick-start path above), the workspace must be read-only. For a writable
workspace, place the rootfs outside it, for example under
`/opt/agentflow-vm/rootfs`. The writable run directory must never overlap the
rootfs. An explicit read-only `app_mount` likewise cannot alias a writable
workspace; omit it when the bundled rootfs copy of AgentFlow is sufficient.

The command defaults to the host numeric UID:GID. virtiofsd translates that
guest identity to the invoking host identity, so writable shares do not retain
guest-root-owned files. A private NSS wrapper passwd/group pair is generated
under the mode-`0700` runtime directory so Python, SSH, and agent CLIs can
resolve the numeric user. Adapter-generated files and copied credentials use
mode `0600`. The PID-1 guest agent invokes the privilege-drop helper by an
absolute trusted-rootfs path with an empty environment, then installs the
prepared command environment only after changing UID/GID; a command-controlled
`PATH` or dynamic-loader variable therefore cannot replace or inject into the
root privilege-drop process.

#### Network policy

The default creates no network device:

```python
target={
    "kind": "cloud_hypervisor",
    "kernel": KERNEL,
    "rootfs": ROOTFS,
    "network_policy": "none",
}
```

The shorthand `"tap"` asks Cloud Hypervisor to create a TAP with host address
`192.168.249.1/24` and configures the guest as `192.168.249.2/24`; creating the
interface requires `CAP_NET_ADMIN`. Any other shorthand string is treated as
the name of a pre-created TAP and uses DHCP in the guest, for example
`"network_policy": "agenttap0"`. DHCP-provided DNS is installed from the
guest's writable `/run` filesystem, so it also works with the immutable rootfs;
an explicit `dns` list replaces those nameservers.

To attach a pre-created TAP and use DHCP inside the guest:

```python
"network_policy": {
    "mode": "tap",
    "tap": "agenttap0",
    "dhcp": True,
    "dns": ["1.1.1.1"],
}
```

For the default single RX/TX queue pair (`num_queues: 2`), create a persistent
TAP with virtio-net headers enabled and grant the AgentFlow host user access to
it. For example:

```bash
sudo ip tuntap add dev agenttap0 mode tap user "$(id -un)" vnet_hdr
sudo ip link set agenttap0 up
```

Do not add `multi_queue` for the default queue count: Cloud Hypervisor opens a
single TAP file descriptor in that configuration, and Linux rejects attaching
it to a TAP that was created as multi-queue. When `num_queues` is greater than
`2`, create the TAP with both `vnet_hdr` and `multi_queue`.

Static guest configuration and an optional Cloud Hypervisor-created TAP use:

```python
"network_policy": {
    "mode": "tap",
    # null asks Cloud Hypervisor to create a TAP and requires CAP_NET_ADMIN.
    "tap": None,
    "host_ip": "192.168.249.1",
    "host_mask": "255.255.255.0",
    "guest_address": "192.168.249.2/24",
    "gateway": "192.168.249.1",
    "dns": ["1.1.1.1"],
    "num_queues": 2,
}
```

`host_ip` and `host_mask` configure Cloud Hypervisor's host side and are IPv4
only. `guest_address` and `gateway` may use IPv4 or IPv6 when the pre-created
TAP supports it, and they must use the same address family. `num_queues`
defaults to `2` and must be an even number.

The target does not create NAT, IP forwarding, DHCP servers, firewall rules, or
an egress allowlist. A TAP connects the guest to networking configured by the
operator. Enforce domain/IP/port policy on the host bridge/firewall or through
a controlled egress gateway. The vsock control path remains available when the
network policy is `none`.

#### Credentials, lifecycle, and failure artifacts

Host environment variables and CLI homes are not inherited implicitly. Values
resolved into `node.env` or `provider.env` are sent in the in-memory vsock
request and never placed in the VMM or virtiofsd argument lists. Inspection
shows only environment keys and redacts all prepared values.

`inherit_credentials: true` is an explicit trust decision. Unlike the Docker
target's nested read-only bind mounts, the VM runner copies adapter-selected
files into its private runtime share so host paths are never exposed through a
symlink. Copies remain mode `0600`.

The configured kernel, rootfs, `cloud-hypervisor`, `virtiofsd`, and mount
sources are trusted host-side inputs. A writable share deliberately grants the
guest command write access to that host directory. `use_worktree` currently
applies only to local targets, so concurrent Cloud Hypervisor nodes share the
same pipeline workspace unless their pipeline configuration points at separate
directories.

On success, failure, cancellation, or timeout AgentFlow requests VMM shutdown,
then terminates and finally kills Cloud Hypervisor and every virtiofsd process
if necessary. Unix API/vsock/backend sockets live in a private host temporary
directory outside the guest-visible runtime share and are removed afterward.
The runtime retains `cloud-hypervisor-vmm.log`,
`cloud-hypervisor-console.log`, and per-share `virtiofsd-NNN.log` files for
diagnosis; they do not contain the command environment or credential values.

Run the credential-free smoke example with:

```bash
AGENTFLOW_CH_KERNEL=.agentflow/cloud-hypervisor/vmlinux-x86_64 \
AGENTFLOW_CH_ROOTFS=.agentflow/cloud-hypervisor/rootfs \
agentflow run examples/cloud_hypervisor_target.py --output summary
```

### Container

The legacy container target wraps the command in `docker run`, mounts the
working directory, runtime directory, and AgentFlow app, then streams stdout
and stderr back into the run trace. See the compatibility note above when
choosing between `container` and `docker`.

## Agent notes

### Codex

- Uses `codex exec --json`
- Maps tools mode to Codex sandboxing
- Keeps model-only Codex nodes on the ambient CLI login path instead of forcing an isolated `CODEX_HOME`
- Writes `CODEX_HOME/config.toml` only when provider or MCP selection requires an isolated home

### Claude

- Uses `claude -p ... --output-format stream-json --verbose`
- Passes `--tools` according to the read-only vs read-write policy
- Writes a per-node MCP JSON config and passes it with `--mcp-config`

### Kimi

- Uses the active Python interpreter via `sys.executable -m agentflow.remote.kimi_bridge`
- Emits a Kimi-style JSON-RPC event stream
- Calls Moonshot's OpenAI-compatible chat completions API
- Provides a small built-in tool layer for read, search, write, and shell actions
