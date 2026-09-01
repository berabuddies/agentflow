from __future__ import annotations

from agentflow.runners.base import Runner
from agentflow.runners.cloud_hypervisor import CloudHypervisorRunner
from agentflow.runners.container import ContainerRunner
from agentflow.runners.docker import DockerRunner
from agentflow.runners.ec2 import EC2Runner
from agentflow.runners.ecs import ECSRunner
from agentflow.runners.local import LocalRunner
from agentflow.runners.ssh import SSHRunner


class RunnerRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Runner] = {
            "local": LocalRunner(),
            "container": ContainerRunner(),
            "docker": DockerRunner(),
            "cloud_hypervisor": CloudHypervisorRunner(),
            "ssh": SSHRunner(),
            "ec2": EC2Runner(),
            "ecs": ECSRunner(),
        }

    def register(self, kind: str, runner: Runner) -> None:
        self._registry[kind] = runner

    def get(self, kind: str) -> Runner:
        return self._registry[kind]


default_runner_registry = RunnerRegistry()
