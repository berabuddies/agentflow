# AgentFlow

Orchestrate Codex, Claude, Kimi, and Pi agents in dependency graphs with parallel fanout, iterative cycles, and execution on local Docker, SSH, EC2, or ECS targets.

![AgentFlow Graph](docs/graph.png)
*94-node pipeline: plan → 64 workers → 8 batch merges → 16 reviews → 4 review merges → synthesis*

## Install / Upgrade

One line:

```bash
curl -fsSL https://raw.githubusercontent.com/shouc/agentflow/master/install.sh | bash
```

This installs agentflow, adds it to PATH, and installs the skill for Codex and Claude Code.

Or manually:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
```

## Quick Start

```python
from agentflow import Graph, codex, claude

with Graph("my-pipeline", concurrency=3) as g:
    plan = codex(task_id="plan", prompt="Inspect the repo and plan the work.", tools="read_only")
    impl = claude(task_id="impl", prompt="Implement the plan:\n{{ nodes.plan.output }}", tools="read_write")
    review = codex(task_id="review", prompt="Review:\n{{ nodes.impl.output }}")
    plan >> impl >> review

print(g.to_json())
```

```bash
agentflow run pipeline.py --output summary
```

Or just ask Codex (the agentflow skill is auto-installed):

```bash
codex "Use agentflow to fan out 10 codex agents, each telling a unique joke, then merge their outputs and pick the funniest one. Write the pipeline and run it."
```

## Parallel Fanout

Fan a node into many parallel copies with `fanout()`:

```python
from agentflow import Graph, codex, fanout, merge

with Graph("code-review", concurrency=8) as g:
    scan = codex(task_id="scan", prompt="List the top 5 files to review.")
    review = fanout(
        codex(task_id="review", prompt="Review {{ item.file }}:\n{{ nodes.scan.output }}"),
        [{"file": "api.py"}, {"file": "auth.py"}, {"file": "db.py"}],
    )
    summary = codex(task_id="summary", prompt=(
        "Merge findings:\n{% for r in fanouts.review.nodes %}{{ r.output }}\n{% endfor %}"
    ))
    scan >> review >> summary

print(g.to_json())
```

`fanout(node, source)` dispatches on type:
- `int` -- N identical copies: `fanout(node, 128)`
- `list` -- one per item: `fanout(node, [{"repo": "api"}, ...])`
- `dict` -- cartesian product: `fanout(node, {"axis1": [...], "axis2": [...]})`

Reduce with `merge(node, source, size=N)` (batch) or `merge(node, source, by=["field"])` (group).

## Iterative Cycles

Loop until a stop condition with `on_failure`:

```python
from agentflow import Graph, codex, claude

with Graph("iterative-impl", max_iterations=5) as g:
    write = codex(
        task_id="write",
        prompt="Write a Python email validator.\n{% if nodes.review.output %}Fix: {{ nodes.review.output }}{% endif %}",
        tools="read_write",
    )
    review = claude(
        task_id="review",
        prompt="Review:\n{{ nodes.write.output }}\nIf complete, say LGTM. Otherwise list issues.",
        success_criteria=[{"kind": "output_contains", "value": "LGTM"}],
    )
    write >> review
    review.on_failure >> write  # loop until LGTM or max_iterations

print(g.to_json())
```

## Local & External Models via Pi

Use the `pi` coding agent as a target alongside `codex` and `claude`. Pi routes
to Anthropic, OpenAI, Groq, Cerebras, xAI, DeepSeek, Gemini, OpenRouter, Bedrock,
etc., and to local endpoints (LMStudio, Ollama) via its OpenAI-compatible or
Anthropic-compatible wire protocols.

```python
from agentflow import Graph, codex, pi

with Graph("mixed") as g:
    # External: Claude via Pi
    review = pi(
        task_id="review",
        prompt="Review {{ nodes.impl.output }}",
        model="anthropic/claude-sonnet-4-6:high",
    )

    # Local: LMStudio (add the provider once in ~/.pi/agent/models.json)
    scan = pi(
        task_id="scan",
        prompt="Scan the repo for TODOs.",
        model="lmstudio/qwen/qwen3.6-27b",
        tools="read_only",
    )
```

For one-off inline provider configs (e.g. a remote LMStudio box), pass a full
`ProviderConfig` via `provider={...}` and AgentFlow materializes a scoped
`models.json` for the run. See `examples/pi_local_lmstudio.py`.

## Inference via SkyPilot

Launch a vLLM or SGLang OpenAI-compatible endpoint on SkyPilot-supported clouds:

```bash
agentflow inference Qwen/Qwen2.5-0.5B-Instruct \
  --gpu aws:1xl4@us-east-1
```

The command prints a `base_url` and `api_key` that can be passed to AgentFlow
nodes through a structured `provider` config. Use `--mode batch` for explicit
JSONL batch jobs.

For graph runs, attach the service directly to the pipeline. AgentFlow launches
one shared SkyPilot service before scheduling nodes, then injects the resolved
OpenAI-compatible provider into PI nodes that do not already set `provider`:

```python
from agentflow import Graph, InferenceSetup, pi

with Graph(
    "my-pipeline",
    concurrency=3,
    inference=InferenceSetup(
        gpu="aws:8x8xb200@us-east-2",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        engine="sglang",
    ),
) as g:
    pi(task_id="answer", prompt="Use the shared inference service.")
```

GPU selectors support single-node and multi-node shapes, including
`aws:8xb200@us-east-1` and `aws:8x8xb200@us-east-2`. Spot is enabled by default;
use `--no-spot` to disable it. On AWS B200, AgentFlow resolves the current
Blackwell-capable DLAMI from AWS SSM unless `--image-id` is supplied.

## Execution Targets

### Docker

Build the bundled agent image once. It contains AgentFlow plus the Codex,
Claude, Kimi, and Pi CLIs, and includes both the Docker client and daemon for
nested Docker workloads:

```bash
docker build -t agentflow-agents:latest .
```

`kind: "docker"` uses that image by default:

```python
from agentflow import Graph, codex

with Graph("docker-review", working_dir=".") as g:
    codex(
        task_id="review",
        prompt="Review the repository without changing it.",
        tools="read_only",
        target={
            "kind": "docker",
            "workdir_read_only": True,
            "mounts": [
                {"source": "./docs", "target": "/reference", "read_only": True},
            ],
            "network_policy": "bridge",
        },
    )

print(g.to_json())
```

AgentFlow automatically bind-mounts the pipeline workspace and a writable
per-node runtime directory. The bundled AgentFlow copy stays inside the image;
an optional `app_mount` can expose a development checkout explicitly.
Additional relative mount sources are resolved from the pipeline
`working_dir`. The structured `network_policy` supports `none`, `bridge`,
`host`, or a named custom network; a custom network can be connected to an
operator-managed egress proxy or firewall for narrower access.

Host API-key variables and CLI login homes are not inherited implicitly by a
Docker target. Put provider keys in `node.env`/`provider.env`; Codex login files
can be exposed read-only with the explicit `inherit_credentials: true` opt-in.

Docker access is opt-in. `mount_docker_daemon: true` mounts the host daemon
socket; this is effectively root-level control of the Docker host, even when
the agent container itself is not privileged. `dind: true` instead starts an
isolated daemon inside the container, requires `privileged: true`, and cannot
be combined with a mounted host daemon. Do not enable either mode for
untrusted prompts or workloads.

The executable example has three modes and does not require model API keys:

```bash
# Offline, read-only workspace; verifies all bundled CLIs.
agentflow run examples/docker_target.py --output summary

# Gives the container control of the host Docker daemon. Treat as host root.
AGENTFLOW_DOCKER_MODE=daemon agentflow run examples/docker_target.py --output summary

# Starts a nested daemon. This uses a privileged container.
AGENTFLOW_DOCKER_MODE=dind agentflow run examples/docker_target.py --output summary
```

See [the pipeline reference](docs/pipelines.md#docker) for every field,
custom-network examples, mount behavior, security notes, and compatibility
with the older `kind: "container"` target.

### Cloud Hypervisor

`kind: "cloud_hypervisor"` boots each node in an ephemeral KVM VM. It uses a
Cloud Hypervisor-compatible Linux kernel, a read-only root filesystem exported
from the bundled all-agent Docker image, virtio-fs for workspace/runtime
sharing, and vsock for the command and streaming output channel. Guest SSH and
guest networking are not required for control:

```bash
docker build -t agentflow-agents:latest .
mkdir -p .agentflow/cloud-hypervisor
cloud_hypervisor/export-rootfs.sh \
  agentflow-agents:latest .agentflow/cloud-hypervisor/rootfs

curl -fL \
  https://github.com/cloud-hypervisor/linux/releases/download/ch-release-v6.16.9-20260508/vmlinux-x86_64 \
  -o .agentflow/cloud-hypervisor/vmlinux-x86_64
```

The Linux host must provide `cloud-hypervisor`, a current Rust `virtiofsd`
with UID/GID translation support, and read/write access to `/dev/kvm`.

```python
codex(
    task_id="vm-review",
    prompt="Review the repository in an isolated VM.",
    target={
        "kind": "cloud_hypervisor",
        "kernel": ".agentflow/cloud-hypervisor/vmlinux-x86_64",
        "rootfs": ".agentflow/cloud-hypervisor/rootfs",
        "cpus": 4,
        "memory_mib": 8192,
        "workdir_read_only": True,
        "network_policy": "none",
    },
)
```

The default network policy creates no network device. An explicit TAP policy
can attach a host-managed interface for model/API access; routing, NAT,
firewalling, and destination restrictions remain host responsibilities. Host
credentials are not inherited unless `inherit_credentials: true` is set.
See [the Cloud Hypervisor reference](docs/pipelines.md#cloud-hypervisor) for
image preparation, TAP/static addressing, mount rules, and the guest contract.

### Remote machines

Run agents on remote machines -- zero config needed:

```python
# EC2 (auto-discovers AMI, key pair, VPC)
codex(task_id="remote", prompt="...", target={"kind": "ec2", "region": "us-east-1"})

# ECS Fargate (auto-discovers VPC, builds agent image)
codex(task_id="remote", prompt="...", target={"kind": "ecs", "region": "us-east-1"})

# SSH
codex(task_id="remote", prompt="...", target={"kind": "ssh", "host": "server", "username": "deploy"})
```

Shared instances across nodes:

```python
plan = codex(task_id="plan", prompt="...", target={"kind": "ec2", "shared": "dev-box"})
impl = codex(task_id="impl", prompt="...", target={"kind": "ec2", "shared": "dev-box"})
plan >> impl  # same EC2 instance, files persist
```

## Scratchboard

Shared memory file across all agents:

```python
with Graph("campaign", scratchboard=True) as g:
    shards = fanout(codex(task_id="fuzz", prompt="..."), 128)
```

## Tuned Agent Evolution

Use a completed Codex run as training data to create a reusable tuned agent:

```python
from agentflow import Graph, codex, evolve

with Graph("improve-codex", working_dir=".") as g:
    source = codex(task_id="plan", prompt="Inspect this repo and summarize the main risks.")
    tuned = evolve(source, target="codex", optimizer="codex")

print(g.to_json())
```

Run order:

```bash
agentflow run pipeline.py
agentflow evolve <run_id> -n <node_id> --target codex --profile codex --optimizer codex
agentflow tuned-agents
agentflow tuned-agent codex_tuned --output json
```

Successful evolutions are stored under `.agentflow/tuned_agents/<name>/versions/<version>/` with copied traces, the cloned repo, and version metadata. Tuned agents currently resolve only on local targets.

## Examples

| Example | What it does |
|---|---|
| `airflow_like.py` | Basic pipeline: plan → implement → review → merge |
| `code_review.py` | Fan out code review across files, merge findings |
| `dep_audit.py` | Audit each dependency for security/license issues |
| `test_gap.py` | Find untested modules, suggest tests per module |
| `multi_agent_debate.py` | Codex vs Claude: independent solve + cross-critique |
| `release_check.py` | Parallel release gate: tests + security + changelog |
| `iterative_impl.py` | Write → review → fix cycle until LGTM |
| `airflow_like_fuzz_batched.py` | 128-shard fanout with batch merge + periodic monitor |
| `airflow_like_fuzz_grouped.py` | Matrix fanout with grouped reducers |
| `ec2_remote.py` | Run codex on a remote EC2 instance |
| `ecs_fargate.py` | Run codex on ECS Fargate |
| `docker_target.py` | Exercise isolated, host-daemon, and Docker-in-Docker targets |
| `cloud_hypervisor_target.py` | Boot an all-agent KVM guest through Cloud Hypervisor, virtio-fs, and vsock |

## Graph Optimization Rounds

Run multiple optimization rounds over your graph with top-level `optimizer` and `n_run`. Use this when you want AgentFlow to let the optimizer rewrite the graph between rounds; the validation step only checks that the edited pipeline loads and passes schema validation, not that the edits are semantically better.

Artifacts and logs for each round live under `.agentflow/runs/<run_id>/optimization/round-XXX/`.

```python
from agentflow import Graph, codex

with Graph(
    "optimization-demo",
    optimizer="codex",
    n_run=2,
    concurrency=2,
) as g:
    plan = codex(task_id="plan", prompt="Outline the tasks required to finish the ticket.")
    review = codex(task_id="review", prompt="Review the plan for missing steps or risks.")
    summary = codex(task_id="summary", prompt="Summarize the approved plan and next actions.")
    plan >> review >> summary

print(g.to_json())
```

## CLI

```bash
agentflow run pipeline.py           # run a pipeline
agentflow run pipeline.py --output summary
agentflow evolve <run_id> -n plan   # evolve a tuned agent from prior Codex traces
agentflow tuned-agents              # list locally registered tuned agents
agentflow tuned-agent codex_tuned   # inspect one tuned agent
agentflow inspect pipeline.py       # show expanded graph
agentflow validate pipeline.py      # check without running
agentflow templates                  # list starter templates
agentflow init > pipeline.py        # scaffold a starter
agentflow serve                     # start the local web UI and API on 127.0.0.1:8000
```

## Web UI and API safety

`agentflow serve` binds to `127.0.0.1` by default.

The web API only accepts `application/json` requests for `/api/runs` and `/api/runs/validate`, and `pipeline_path` is disabled on those endpoints by default. This prevents the browser-facing control plane from executing arbitrary local `.py` pipeline files just by referencing a path.

Inline `shell` and `python` utility agents are also disabled for web API requests by default. This prevents browser/API callers from turning a submitted pipeline into direct local command execution.

If you intentionally want the web API to load pipelines from filesystem paths or accept inline executable utility agents in a trusted local environment, opt in explicitly:

```bash
AGENTFLOW_API_ALLOW_PIPELINE_PATH=1 \
AGENTFLOW_API_ALLOW_EXECUTABLE_AGENTS=1 \
agentflow serve
```

Those opt-ins are meant for trusted operator-controlled workflows only.

## Acknowledgements

* [gepa](https://github.com/gepa-ai/gepa)
* [kiss-ai](https://github.com/ksenxx/kiss_ai)
* [claude-code-telegram](https://github.com/RichardAtCT/claude-code-telegram)
* [linux.do](https://linux.do)

## Citation
If you use this tool, please cite our paper:

```bibtex
@misc{liu2026synthesizingmultiagentharnessesvulnerability,
      title={Synthesizing Multi-Agent Harnesses for Vulnerability Discovery}, 
      author={Hanzhi Liu and Chaofan Shou and Xiaonan Liu and Hongbo Wen and Yanju Chen and Ryan Jingyang Fang and Yu Feng},
      year={2026},
      eprint={2604.20801},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.20801}, 
}
```
