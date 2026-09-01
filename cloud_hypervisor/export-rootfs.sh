#!/bin/sh
set -eu

image="${1:-agentflow-agents:latest}"
destination="${2:-}"

if [ -z "$destination" ]; then
    echo "usage: $0 [image] DESTINATION" >&2
    exit 64
fi
if [ -e "$destination" ]; then
    echo "destination already exists: $destination" >&2
    exit 73
fi

mkdir -m 0755 -- "$destination"
container_id=""
completed=false
cleanup() {
    if [ -n "$container_id" ]; then
        docker rm "$container_id" >/dev/null 2>&1 || true
    fi
    if [ "$completed" != true ]; then
        rm -rf -- "$destination"
    fi
}
trap cleanup EXIT HUP INT TERM

container_id="$(docker create "$image")"

docker export "$container_id" | tar --no-same-owner -x -C "$destination"

test -x "$destination/usr/local/bin/agentflow-cloud-hypervisor-init"
test -L "$destination/opt/agentflow-venv/bin/python3"
find "$destination/opt/uv-python" \
    -type f \
    -path '*/bin/python3.13' \
    -perm -0100 \
    -print -quit | grep -q .

completed=true
echo "exported $image to $destination"
