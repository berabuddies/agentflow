#!/bin/sh
set -eu

is_enabled() {
    case "${1:-}" in
        1 | true | TRUE | yes | YES | on | ON) return 0 ;;
        *) return 1 ;;
    esac
}

fail() {
    echo "agentflow: $*" >&2
    exit 64
}

validate_id() {
    id_name="$1"
    id_value="$2"
    case "$id_value" in
        '' | *[!0-9]*) fail "$id_name must be a non-negative integer" ;;
    esac
}

prepare_runtime_home() {
    runtime_uid="$1"
    runtime_gid="$2"
    runtime_home="${HOME:-}"
    home_created=0

    # Docker defaults HOME to /root even for a numeric --user. Use a private
    # ephemeral home in that case; AgentFlow normally supplies its writable
    # per-node runtime home explicitly.
    case "$runtime_home" in
        '' | / | /root | *:*)
            if ! runtime_home="$(mktemp -d /tmp/agentflow-home.XXXXXX 2>/dev/null)"; then
                fail "HOME must point to a writable mount when the container root filesystem is read-only"
            fi
            home_created=1
            HOME="$runtime_home"
            export HOME
            ;;
        /*) ;;
        *) fail "HOME must be an absolute path" ;;
    esac

    if [ ! -d "$runtime_home" ]; then
        mkdir -p "$runtime_home" || fail "cannot create HOME at $runtime_home"
        home_created=1
    fi

    if [ "$(id -u)" = 0 ] && [ "$runtime_uid" != 0 ] && [ "$home_created" = 1 ]; then
        chown "$runtime_uid:$runtime_gid" "$runtime_home"
    fi

    if [ "$(id -u)" = 0 ] && [ "$runtime_uid" != 0 ]; then
        if ! su-exec "$runtime_uid:$runtime_gid" test -w "$runtime_home"; then
            fail "HOME must be a writable mount for $runtime_uid:$runtime_gid: $runtime_home"
        fi
    elif [ ! -w "$runtime_home" ]; then
        fail "HOME must point to a writable mount: $runtime_home"
    fi
}

prepare_nss_identity() {
    runtime_uid="$1"
    runtime_gid="$2"

    prepare_runtime_home "$runtime_uid" "$runtime_gid"

    passwd_entry="$(getent passwd "$runtime_uid" 2>/dev/null || true)"
    group_entry="$(getent group "$runtime_gid" 2>/dev/null || true)"

    if [ -n "$passwd_entry" ]; then
        runtime_user="$(printf '%s\n' "$passwd_entry" | cut -d: -f1)"
    else
        runtime_user=agentflow
        if getent passwd "$runtime_user" >/dev/null 2>&1; then
            runtime_user="agentflow$runtime_uid"
        fi
    fi

    if [ -n "$group_entry" ]; then
        runtime_group="$(printf '%s\n' "$group_entry" | cut -d: -f1)"
    else
        runtime_group=agentflow
        if getent group "$runtime_group" >/dev/null 2>&1; then
            runtime_group="agentflow$runtime_gid"
        fi
    fi

    if ! nss_dir="$(mktemp -d "$HOME/.agentflow-nss.XXXXXX" 2>/dev/null)"; then
        fail "cannot create the private NSS directory below writable HOME: $HOME"
    fi
    chmod 0700 "$nss_dir"
    # Replace (rather than duplicate) a coincidentally existing numeric UID so
    # getpwuid reports the container's actual primary GID and writable HOME.
    awk -F: -v runtime_uid="$runtime_uid" '$3 != runtime_uid' /etc/passwd \
        >"$nss_dir/passwd"
    cp /etc/group "$nss_dir/group"

    printf '%s:x:%s:%s:AgentFlow runtime:%s:/bin/sh\n' \
        "$runtime_user" "$runtime_uid" "$runtime_gid" "$HOME" \
        >>"$nss_dir/passwd"
    if [ -z "$group_entry" ]; then
        printf '%s:x:%s:\n' "$runtime_group" "$runtime_gid" >>"$nss_dir/group"
    fi

    # Docker --group-add can introduce supplementary numeric groups that also
    # lack names. Map those when setup runs as the final non-root identity.
    if [ "$(id -u)" = "$runtime_uid" ]; then
        for supplementary_gid in $(id -G); do
            validate_id "supplementary group ID" "$supplementary_gid"
            if [ "$supplementary_gid" = "$runtime_gid" ]; then
                continue
            fi
            if ! getent group "$supplementary_gid" >/dev/null 2>&1; then
                printf 'agentflow%s:x:%s:\n' "$supplementary_gid" "$supplementary_gid" \
                    >>"$nss_dir/group"
            fi
        done
    fi

    chmod 0600 "$nss_dir/passwd" "$nss_dir/group"
    if [ "$(id -u)" = 0 ] && [ "$runtime_uid" != 0 ]; then
        chown "$runtime_uid:$runtime_gid" "$nss_dir" "$nss_dir/passwd" "$nss_dir/group"
    fi

    NSS_WRAPPER_PASSWD="$nss_dir/passwd"
    NSS_WRAPPER_GROUP="$nss_dir/group"
    LD_PRELOAD="/usr/lib/libnss_wrapper.so${LD_PRELOAD:+:$LD_PRELOAD}"
    USER="$runtime_user"
    LOGNAME="$runtime_user"
    export NSS_WRAPPER_PASSWD NSS_WRAPPER_GROUP LD_PRELOAD USER LOGNAME

    getent passwd "$runtime_uid" >/dev/null 2>&1 \
        || fail "NSS setup could not resolve uid $runtime_uid"
    getent group "$runtime_gid" >/dev/null 2>&1 \
        || fail "NSS setup could not resolve gid $runtime_gid"
}

current_uid="$(id -u)"
current_gid="$(id -g)"
run_uid="${AGENTFLOW_RUN_UID:-$current_uid}"
run_gid="${AGENTFLOW_RUN_GID:-$current_gid}"

if { [ -n "${AGENTFLOW_RUN_UID:-}" ] && [ -z "${AGENTFLOW_RUN_GID:-}" ]; } \
    || { [ -z "${AGENTFLOW_RUN_UID:-}" ] && [ -n "${AGENTFLOW_RUN_GID:-}" ]; }; then
    fail "AGENTFLOW_RUN_UID and AGENTFLOW_RUN_GID must be set together"
fi
validate_id AGENTFLOW_RUN_UID "$run_uid"
validate_id AGENTFLOW_RUN_GID "$run_gid"

if [ "$current_uid" != 0 ] \
    && { [ "$run_uid" != "$current_uid" ] || [ "$run_gid" != "$current_gid" ]; }; then
    fail "cannot switch from $current_uid:$current_gid to $run_uid:$run_gid without root"
fi

show_dockerd_log() {
    if [ -r "$dockerd_log" ]; then
        echo "agentflow: dockerd log follows:" >&2
        tail -n 200 "$dockerd_log" >&2 || true
    fi
}

if ! is_enabled "${AGENTFLOW_DIND:-0}"; then
    if [ "$current_uid" = 0 ] && [ "$run_uid" != 0 ]; then
        prepare_nss_identity "$run_uid" "$run_gid"
        exec su-exec "$run_uid:$run_gid" "$@"
    fi
    if [ "$current_uid" != 0 ]; then
        prepare_nss_identity "$current_uid" "$current_gid"
    fi
    exec "$@"
fi

if [ "$current_uid" != 0 ]; then
    fail "DinD must start as root; omit Docker --user and use AGENTFLOW_RUN_UID/GID"
fi

case "${AGENTFLOW_DIND_TIMEOUT:-30}" in
    '' | *[!0-9]*)
        echo "agentflow: AGENTFLOW_DIND_TIMEOUT must be a non-negative integer" >&2
        exit 64
        ;;
esac

# DinD always uses the daemon started in this container. Socket-mounted and
# remote-daemon modes leave AGENTFLOW_DIND unset and preserve their DOCKER_HOST.
export DOCKER_HOST="unix:///var/run/docker.sock"

dockerd_log="${AGENTFLOW_DIND_LOG:-/tmp/agentflow-dockerd.log}"
# Supplying an explicit host keeps the upstream DinD entrypoint from adding
# any TCP listener. The nested daemon is intentionally reachable only through
# its in-container Unix socket.
/usr/local/bin/dockerd-entrypoint.sh \
    dockerd \
    --host=unix:///var/run/docker.sock \
    >"$dockerd_log" 2>&1 &
dockerd_pid=$!

elapsed=0
while ! docker info >/dev/null 2>&1; do
    if ! kill -0 "$dockerd_pid" 2>/dev/null; then
        status=1
        wait "$dockerd_pid" || status=$?
        echo "agentflow: dockerd exited before becoming ready (status $status)" >&2
        show_dockerd_log
        exit "$status"
    fi

    if [ "$elapsed" -ge "${AGENTFLOW_DIND_TIMEOUT:-30}" ]; then
        echo "agentflow: dockerd did not become ready within ${AGENTFLOW_DIND_TIMEOUT:-30}s" >&2
        show_dockerd_log
        kill "$dockerd_pid" 2>/dev/null || true
        wait "$dockerd_pid" 2>/dev/null || true
        exit 70
    fi

    sleep 1
    elapsed=$((elapsed + 1))
done

if [ "$run_uid" != 0 ]; then
    # Let the post-start, host-identity agent process use the daemon without a
    # supplementary in-image docker group. The socket remains owner/group-only.
    chown "0:$run_gid" /var/run/docker.sock
    chmod 0660 /var/run/docker.sock
    prepare_nss_identity "$run_uid" "$run_gid"
    exec su-exec "$run_uid:$run_gid" "$@"
fi

exec "$@"
