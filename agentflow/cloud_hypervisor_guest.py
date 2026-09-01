"""Single-request AgentFlow guest agent for Cloud Hypervisor VMs.

The host connects through Cloud Hypervisor's Unix-to-vsock proxy. The guest
mounts the advertised virtio-fs shares, configures the optional TAP interface,
executes one command, and streams newline-delimited JSON events back to the
host. This module intentionally depends only on Python's standard library so it
can run as PID 1 in the exported AgentFlow image root filesystem.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import ctypes
import json
import os
import posixpath
import re
import selectors
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_STREAM_CHUNK_BYTES = 8 * 1024
_STREAM_DRAIN_SECONDS = 3.0
_ALLOWED_TAG = re.compile(r"^agentflow-[A-Za-z0-9_.-]+$")
_RESERVED_TARGETS = ("/dev", "/proc", "/run", "/sys")
# CPython only exposes these names when it was built against libc headers that
# define AF_VSOCK.  The bundled musl Python can still use Linux vsock sockets by
# passing the stable UAPI numeric values directly.
_LINUX_AF_VSOCK = 40
_LINUX_VMADDR_CID_ANY = 0xFFFFFFFF
_SU_EXEC_PATH = "/sbin/su-exec"
_ENV_PATH = "/usr/bin/env"
_SHELL_PATH = "/bin/sh"


class GuestRequestError(ValueError):
    """The host sent an invalid or unsupported guest request."""


def _send_event(stream: BinaryIO, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    stream.write(encoded)
    stream.flush()


def _run_checked(command: list[str], *, description: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip()
    suffix = f": {detail}" if detail else ""
    raise GuestRequestError(
        f"{description} failed with status {result.returncode}{suffix}"
    )


def _paths_overlap(left: str, right: str) -> bool:
    left = posixpath.normpath(left)
    right = posixpath.normpath(right)
    return (
        left == right
        or left.startswith(right.rstrip("/") + "/")
        or right.startswith(left.rstrip("/") + "/")
    )


def _validate_mounts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise GuestRequestError("`mounts` must be a list")
    mounts: list[dict[str, Any]] = []
    targets: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise GuestRequestError("each mount must be an object")
        tag = item.get("tag")
        target = item.get("target")
        read_only = item.get("read_only")
        if not isinstance(tag, str) or not _ALLOWED_TAG.fullmatch(tag):
            raise GuestRequestError(f"invalid virtio-fs tag: {tag!r}")
        if (
            not isinstance(target, str)
            or not target.startswith("/")
            or target.startswith("//")
        ):
            raise GuestRequestError(
                f"mount target must be an absolute guest path: {target!r}"
            )
        target = posixpath.normpath(target)
        if target == "/" or any(
            _paths_overlap(target, reserved) for reserved in _RESERVED_TARGETS
        ):
            raise GuestRequestError(
                f"mount target overlaps a reserved guest path: {target}"
            )
        if not isinstance(read_only, bool):
            raise GuestRequestError(f"mount `read_only` must be boolean for {target}")
        if any(_paths_overlap(target, existing) for existing in targets):
            raise GuestRequestError(
                f"mount target overlaps another guest share: {target}"
            )
        targets.append(target)
        mounts.append({"tag": tag, "target": target, "read_only": read_only})
    return mounts


def _mount_shares(mounts: list[dict[str, Any]]) -> None:
    for mount in mounts:
        target = Path(mount["target"])
        if not target.is_dir():
            raise GuestRequestError(
                f"virtio-fs mount point must already exist in the read-only guest rootfs: {target}"
            )
        options = "ro" if mount["read_only"] else "rw"
        _run_checked(
            ["mount", "-t", "virtiofs", "-o", options, mount["tag"], str(target)],
            description=f"mounting virtio-fs tag {mount['tag']} at {target}",
        )


_RUNTIME_RESOLV_CONF = Path("/run/agentflow-resolv.conf")
_RUNTIME_DHCP_CONFIG = Path("/run/agentflow-udhcpc.conf")


def _bind_resolv_conf(path: Path) -> None:
    _run_checked(
        ["mount", "--bind", str(path), "/etc/resolv.conf"],
        description="installing guest DNS config",
    )


def _write_resolv_conf(dns_servers: list[str], *, already_bound: bool) -> bool:
    if not dns_servers:
        return already_bound
    path = _RUNTIME_RESOLV_CONF
    path.write_text(
        "".join(f"nameserver {server}\n" for server in dns_servers), encoding="utf-8"
    )
    if not already_bound:
        _bind_resolv_conf(path)
    return True


def _configure_dhcp() -> bool:
    # Alpine's default udhcpc script writes a sibling temporary file before
    # replacing resolv.conf. The guest rootfs is intentionally read-only, so
    # point that script at /run through its existing configuration hook.
    _RUNTIME_DHCP_CONFIG.write_text(
        f'RESOLV_CONF="{_RUNTIME_RESOLV_CONF}"\n', encoding="utf-8"
    )
    _run_checked(
        [
            "mount",
            "--bind",
            str(_RUNTIME_DHCP_CONFIG),
            "/etc/udhcpc/udhcpc.conf",
        ],
        description="installing read-only-rootfs DHCP config",
    )
    _run_checked(
        ["udhcpc", "-q", "-n", "-i", "eth0"],
        description="guest DHCP configuration",
    )
    if _RUNTIME_RESOLV_CONF.is_file():
        _bind_resolv_conf(_RUNTIME_RESOLV_CONF)
        return True
    return False


def _configure_network(value: Any) -> None:
    if not isinstance(value, dict):
        raise GuestRequestError("`network` must be an object")
    mode = value.get("mode")
    if mode not in {"none", "tap"}:
        raise GuestRequestError(f"unsupported network mode: {mode!r}")

    with open(os.devnull, "wb") as devnull:
        subprocess.run(
            ["ip", "link", "set", "lo", "up"],
            stdout=devnull,
            stderr=devnull,
            check=False,
        )
    if mode == "none":
        return

    _run_checked(
        ["ip", "link", "set", "eth0", "up"], description="bringing guest eth0 up"
    )
    guest_address = value.get("guest_address")
    gateway = value.get("gateway")
    dhcp = value.get("dhcp")
    dns = value.get("dns")
    if (
        not isinstance(dhcp, bool)
        or not isinstance(dns, list)
        or not all(isinstance(item, str) for item in dns)
    ):
        raise GuestRequestError("invalid guest network configuration")
    resolv_conf_bound = False
    if dhcp:
        resolv_conf_bound = _configure_dhcp()
    elif guest_address is not None:
        if not isinstance(guest_address, str):
            raise GuestRequestError("`network.guest_address` must be a string")
        _run_checked(
            ["ip", "address", "add", guest_address, "dev", "eth0"],
            description="setting guest address",
        )
        if gateway is not None:
            if not isinstance(gateway, str):
                raise GuestRequestError("`network.gateway` must be a string")
            _run_checked(
                ["ip", "route", "replace", "default", "via", gateway],
                description="setting guest gateway",
            )
    _write_resolv_conf(dns, already_bound=resolv_conf_bound)


def _validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuestRequestError("request must be an object")
    if value.get("protocol") != _PROTOCOL_VERSION:
        raise GuestRequestError(
            f"unsupported protocol version: {value.get('protocol')!r}"
        )
    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(
            isinstance(item, str) and item and "\x00" not in item for item in command
        )
    ):
        raise GuestRequestError(
            "`command` must be a non-empty string list without NUL bytes"
        )
    env = value.get("env")
    if not isinstance(env, dict) or not all(
        isinstance(key, str)
        and key
        and "=" not in key
        and "\x00" not in key
        and isinstance(item, str)
        and "\x00" not in item
        for key, item in env.items()
    ):
        raise GuestRequestError("`env` must be a NUL-free environment mapping")
    cwd = value.get("cwd")
    if (
        not isinstance(cwd, str)
        or not cwd.startswith("/")
        or cwd.startswith("//")
        or "\x00" in cwd
    ):
        raise GuestRequestError("`cwd` must be an absolute guest path")
    stdin = value.get("stdin")
    if stdin is not None and not isinstance(stdin, str):
        raise GuestRequestError("`stdin` must be a string or null")
    uid = value.get("uid")
    gid = value.get("gid")
    if (
        not isinstance(uid, int)
        or isinstance(uid, bool)
        or not 0 <= uid <= 4_294_967_294
        or not isinstance(gid, int)
        or isinstance(gid, bool)
        or not 0 <= gid <= 4_294_967_294
    ):
        raise GuestRequestError("`uid` and `gid` must be Linux 32-bit identifiers")
    return {
        "command": command,
        "env": env,
        "cwd": posixpath.normpath(cwd),
        "stdin": stdin,
        "uid": uid,
        "gid": gid,
        "mounts": _validate_mounts(value.get("mounts")),
        "network": value.get("network"),
    }


def _stdin_writer(pipe: BinaryIO, content: str | None) -> None:
    try:
        if content is not None:
            pipe.write(content.encode("utf-8"))
            pipe.flush()
    except BrokenPipeError:
        pass
    finally:
        pipe.close()


def _send_stream_text(
    protocol_stream: BinaryIO,
    stream_name: str,
    content: str,
    *,
    line_end: bool,
) -> None:
    chunks = [
        content[offset : offset + _MAX_STREAM_CHUNK_BYTES]
        for offset in range(0, len(content), _MAX_STREAM_CHUNK_BYTES)
    ] or [""]
    for index, chunk in enumerate(chunks):
        _send_event(
            protocol_stream,
            {
                "event": "stream",
                "stream": stream_name,
                "text": chunk,
                "line_end": line_end and index == len(chunks) - 1,
            },
        )


def _stream_process(process: subprocess.Popen[bytes], protocol_stream: BinaryIO) -> int:
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending = {"stdout": b"", "stderr": b""}
    decoders = {
        "stdout": codecs.getincrementaldecoder("utf-8")("replace"),
        "stderr": codecs.getincrementaldecoder("utf-8")("replace"),
    }
    line_fragment_sent = {"stdout": False, "stderr": False}

    def send_decoded(stream_name: str, content: bytes, *, final: bool) -> None:
        decoder = decoders[stream_name]
        text = decoder.decode(content, final=final)
        if text:
            _send_stream_text(protocol_stream, stream_name, text, line_end=final)
            line_fragment_sent[stream_name] = not final
        elif final:
            _send_stream_text(protocol_stream, stream_name, "", line_end=True)
            line_fragment_sent[stream_name] = False
        if final:
            decoder.reset()

    def finish_stream(file_object: BinaryIO, stream_name: str) -> None:
        with contextlib.suppress(KeyError):
            selector.unregister(file_object)
        remainder = pending[stream_name]
        buffered_input = bool(decoders[stream_name].getstate()[0])
        if remainder or buffered_input or line_fragment_sent[stream_name]:
            send_decoded(stream_name, remainder, final=True)
        pending[stream_name] = b""

    exit_seen_at: float | None = None
    while selector.get_map():
        events = selector.select(timeout=0.2)
        for key, _ in events:
            chunk = os.read(key.fileobj.fileno(), 64 * 1024)
            stream_name = key.data
            if not chunk:
                finish_stream(key.fileobj, stream_name)
                continue
            buffered = pending[stream_name] + chunk
            lines = buffered.split(b"\n")
            pending[stream_name] = lines.pop()
            for line in lines:
                send_decoded(stream_name, line, final=True)
            while len(pending[stream_name]) > _MAX_STREAM_CHUNK_BYTES:
                ready_chunk = pending[stream_name][:_MAX_STREAM_CHUNK_BYTES]
                pending[stream_name] = pending[stream_name][_MAX_STREAM_CHUNK_BYTES:]
                send_decoded(stream_name, ready_chunk, final=False)

        if process.poll() is not None:
            if exit_seen_at is None:
                exit_seen_at = time.monotonic()
            if time.monotonic() - exit_seen_at >= _STREAM_DRAIN_SECONDS:
                for key in list(selector.get_map().values()):
                    finish_stream(key.fileobj, key.data)
                break
    return process.wait()


def _command_for_identity(
    command: list[str], env: dict[str, str], uid: int, gid: int
) -> tuple[list[str], dict[str, str]]:
    if (uid, gid) == (0, 0):
        return list(command), env

    # Do not let request-controlled PATH or dynamic-loader variables affect the
    # root process responsible for dropping privileges. The minimal launcher
    # has an empty environment and passes the requested environment as argv to
    # `env` only after su-exec has changed UID/GID. The small shell wrapper
    # preserves command names that themselves contain `=`.
    assignments = [f"{key}={value}" for key, value in env.items()]
    return (
        [
            _SU_EXEC_PATH,
            f"{uid}:{gid}",
            _ENV_PATH,
            "-i",
            "--",
            *assignments,
            _SHELL_PATH,
            "-c",
            'exec -- "$@"',
            "agentflow-command",
            *command,
        ],
        {},
    )


def _execute_request(request: dict[str, Any], protocol_stream: BinaryIO) -> int:
    _mount_shares(request["mounts"])
    _configure_network(request["network"])
    cwd = Path(request["cwd"])
    if not cwd.is_dir():
        raise GuestRequestError(f"guest working directory does not exist: {cwd}")

    env = os.environ.copy()
    env.update(request["env"])
    home = env.get("HOME")
    if not home or not Path(home).is_dir():
        raise GuestRequestError(
            f"guest HOME must name a pre-created runtime directory: {home!r}"
        )

    uid = request["uid"]
    gid = request["gid"]
    command, launch_env = _command_for_identity(list(request["command"]), env, uid, gid)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=request["cwd"],
        env=launch_env,
    )
    assert process.stdin is not None
    stdin_thread = threading.Thread(
        target=_stdin_writer, args=(process.stdin, request["stdin"]), daemon=True
    )
    stdin_thread.start()
    _send_event(protocol_stream, {"event": "started", "pid": process.pid})
    exit_code = _stream_process(process, protocol_stream)
    stdin_thread.join(timeout=1.0)
    return exit_code


class _SockAddrVm(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort),
        ("port", ctypes.c_uint32),
        ("cid", ctypes.c_uint32),
        ("zero", ctypes.c_ubyte * 4),
    ]


def _accept_vsock_without_python_address_support(port: int) -> int:
    """Accept one Linux vsock connection when CPython lacks sockaddr_vm support."""

    libc = ctypes.CDLL(None, use_errno=True)
    libc.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libc.socket.restype = ctypes.c_int
    libc.bind.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_SockAddrVm),
        ctypes.c_uint32,
    ]
    libc.bind.restype = ctypes.c_int
    libc.listen.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.listen.restype = ctypes.c_int
    libc.accept.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.accept.restype = ctypes.c_int

    def checked(result: int, operation: str) -> int:
        if result >= 0:
            return result
        error_number = ctypes.get_errno()
        raise OSError(error_number, f"vsock {operation}: {os.strerror(error_number)}")

    listener_fd = checked(libc.socket(_LINUX_AF_VSOCK, socket.SOCK_STREAM, 0), "socket")
    try:
        address = _SockAddrVm(
            family=_LINUX_AF_VSOCK,
            reserved=0,
            port=port,
            cid=_LINUX_VMADDR_CID_ANY,
        )
        checked(
            libc.bind(listener_fd, ctypes.byref(address), ctypes.sizeof(address)),
            "bind",
        )
        checked(libc.listen(listener_fd, 1), "listen")
        return checked(libc.accept(listener_fd, None, None), "accept")
    finally:
        os.close(listener_fd)


def _handle_connection(protocol_stream: BinaryIO) -> int:
    _send_event(protocol_stream, {"event": "hello", "protocol": _PROTOCOL_VERSION})
    line = protocol_stream.readline(_MAX_REQUEST_BYTES + 1)
    if not line:
        raise GuestRequestError("host disconnected before sending a request")
    if len(line) > _MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        raise GuestRequestError(
            "guest request exceeds the 16 MiB single-line protocol limit"
        )
    try:
        request = _validate_request(json.loads(line))
        exit_code = _execute_request(request, protocol_stream)
    except (
        GuestRequestError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        _send_event(protocol_stream, {"event": "error", "message": str(exc)})
        return 1
    _send_event(protocol_stream, {"event": "result", "exit_code": exit_code})
    return 0


def _serve_one(port: int) -> int:
    if hasattr(socket, "AF_VSOCK") and hasattr(socket, "VMADDR_CID_ANY"):
        listener = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        listener.bind((socket.VMADDR_CID_ANY, port))
        listener.listen(1)
        connection, _ = listener.accept()
        listener.close()
        with connection, connection.makefile("rwb", buffering=0) as protocol_stream:
            return _handle_connection(protocol_stream)

    connection_fd = _accept_vsock_without_python_address_support(port)
    with os.fdopen(connection_fd, "r+b", buffering=0) as protocol_stream:
        return _handle_connection(protocol_stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentFlow Cloud Hypervisor guest agent"
    )
    parser.add_argument("--port", type=int, default=4050)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        return _serve_one(args.port)
    except (
        GuestRequestError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"agentflow-cloud-hypervisor-guest: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
