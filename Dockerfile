# syntax=docker/dockerfile:1.7

ARG DOCKER_IMAGE=docker:29-dind
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.5

FROM ${UV_IMAGE} AS uv

FROM ${DOCKER_IMAGE}

ARG CODEX_VERSION=0.148.0
ARG CLAUDE_VERSION=2.1.236
ARG KIMI_CLI_VERSION=1.49.0
ARG PI_VERSION=0.84.2

LABEL org.opencontainers.image.title="AgentFlow agent runtime" \
      org.opencontainers.image.description="AgentFlow with Codex, Claude Code, Kimi CLI, Pi, and Docker-in-Docker"

# Claude Code requires this container marker before it permits the adapter's
# bypassPermissions mode while the DinD-capable image is running as root.
ENV PATH="/opt/agentflow-venv/bin:${PATH}" \
    IS_SANDBOX=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apk add --no-cache \
        bash \
        ca-certificates \
        curl \
        gcompat \
        git \
        iproute2 \
        jq \
        libgcc \
        libstdc++ \
        nodejs \
        nss_wrapper \
        npm \
        openssh-client \
        ripgrep \
        su-exec \
        zsh \
    && update-ca-certificates \
    && uv venv --python 3.13 /opt/agentflow-venv \
    && uv pip install \
        --python /opt/agentflow-venv/bin/python \
        --no-cache \
        "kimi-cli==${KIMI_CLI_VERSION}" \
    && npm install --global --omit=dev --no-audit --no-fund \
        "@openai/codex@${CODEX_VERSION}" \
        "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
        "@earendil-works/pi-coding-agent@${PI_VERSION}" \
    && npm cache clean --force

WORKDIR /opt/agentflow-src
COPY pyproject.toml README.md ./
COPY agentflow ./agentflow

RUN uv pip install \
        --python /opt/agentflow-venv/bin/python \
        --no-cache \
        . \
    && mkdir -p /workspace \
        /agentflow-runtime \
        /agentflow-app \
        /inputs \
        /outputs \
        /reference \
    && command -v agentflow \
    && command -v codex \
    && command -v claude \
    && command -v kimi \
    && command -v pi \
    && command -v docker \
    && command -v dockerd \
    && command -v su-exec \
    && test -r /usr/lib/libnss_wrapper.so

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/agentflow-entrypoint
COPY --chmod=0755 cloud_hypervisor/guest-init.sh /usr/local/bin/agentflow-cloud-hypervisor-init

WORKDIR /workspace

ENTRYPOINT ["agentflow-entrypoint"]
CMD ["agentflow", "--help"]
