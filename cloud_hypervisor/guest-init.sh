#!/bin/sh
set -u

export PATH="/opt/agentflow-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export IS_SANDBOX=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
mount -t tmpfs -o mode=0755,nosuid,nodev tmpfs /run 2>/dev/null || true
mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs /tmp 2>/dev/null || true

guest_port=4050
for kernel_argument in $(cat /proc/cmdline 2>/dev/null || true); do
    case "$kernel_argument" in
        agentflow.guest_port=*) guest_port="${kernel_argument#agentflow.guest_port=}" ;;
    esac
done

case "$guest_port" in
    '' | *[!0-9]*)
        echo "agentflow-cloud-hypervisor-init: invalid guest port: $guest_port" >&2
        guest_status=64
        ;;
    *)
        python3 -m agentflow.cloud_hypervisor_guest --port "$guest_port"
        guest_status=$?
        ;;
esac

sync || true
poweroff -f 2>/dev/null || reboot -f 2>/dev/null || true
exit "$guest_status"
