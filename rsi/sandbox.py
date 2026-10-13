"""Normal Python policies, opaque object snapshots, read-only isolated queries.

Landlock and seccomp enforce operational boundaries; no Python syntax or
function whitelist. This worker is not an audited general-purpose sandbox.
"""
from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import functools
import hashlib
import importlib.util
import inspect
import io
import platform
import sysconfig
import types
import ctypes
import ctypes.util
import errno
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class SandboxError(RuntimeError):
    """Candidate rejected, failed, or exceeded its execution bounds."""
    def __init__(self, message, category="runtime"):
        super().__init__(message)
        self.category = category


class ContractError(SandboxError):
    def __init__(self, message): super().__init__(message, "contract")


class IsolationError(SandboxError):
    def __init__(self, message): super().__init__(message, "isolation")


def validate_source(source: str) -> ast.Module:
    """Validate Python syntax only; implementation style is unrestricted."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ContractError("candidate source exceeds size limit")
    try:
        tree = ast.parse(source, mode="exec")
        compile(tree, "<candidate-policy>", "exec")
        return tree
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ContractError("candidate source is not valid Python") from exc


def _landlock_abi() -> int:
    if not sys.platform.startswith("linux") or platform.machine() not in {"x86_64", "aarch64", "riscv64"}:
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    return max(0, int(libc.syscall(444, 0, 0, 1)))


@functools.lru_cache(maxsize=1)
def isolation_available() -> bool:
    return bool(_landlock_abi() and ctypes.util.find_library("seccomp")
                and importlib.util.find_spec("cloudpickle"))


@functools.lru_cache(maxsize=1)
def _runtime_paths():
    # Only interpreter/library trees; never the project root, datasets, or runs.
    system_libraries = {str(Path(p).resolve()) for p in
                        ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/local/lib") if Path(p).exists()}
    roots = ["/usr/share/zoneinfo", "/etc/ld.so.cache", "/etc/localtime",
             "/dev/null", "/dev/urandom"]
    libraries = {p for p in sys.path if "site-packages" in p}
    libraries.update(sysconfig.get_paths().get(k, "") for k in ("stdlib", "platstdlib", "purelib", "platlib"))
    directory_only = set(system_libraries)
    for root in system_libraries:
        roots.extend(str(p) for p in Path(root).rglob("*.so*") if p.is_file())
    for root in libraries:
        if not root or not Path(root).is_dir(): continue
        directory_only.add(str(Path(root).resolve()))
        for child in Path(root).iterdir():
            if child.name.startswith(".") or child.name == "site-packages": continue
            if child.is_file() and child.suffix in {".py", ".pyc", ".so", ".zip"}:
                roots.append(str(child))
            elif child.is_dir() and (child.name == "lib-dynload" or child.name.isidentifier() or
                                      child.name.endswith((".libs", ".dist-info", ".egg-info"))):
                roots.append(str(child))
    allowed = {str(Path(p).resolve()): False for p in roots if p and Path(p).exists()}
    for path in directory_only: allowed.setdefault(path, True)
    return tuple(sorted(allowed.items()))


def _install_landlock(paths):
    """Restrict filesystem access to read-only runtime paths before any code."""
    abi = _landlock_abi()
    if not abi:
        raise IsolationError("Landlock isolation unavailable; fail-closed")
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    handled = (1 << 13) - 1
    if abi >= 2: handled |= 1 << 13  # REFER
    if abi >= 3: handled |= 1 << 14  # TRUNCATE
    libc = ctypes.CDLL(None, use_errno=True)
    attr = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise IsolationError("Landlock ruleset creation failed")
    try:
        for path, directory_only in paths:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = (1 << 3) if directory_only else (1 << 2) | 1
                if Path(path).is_dir(): allowed |= 1 << 3  # READ_DIR
                rule = PathRule(allowed, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise IsolationError("Landlock path rule failed")
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) < 0:
            raise IsolationError("Landlock restriction failed")
    finally:
        os.close(fd)


def _snapshot(state, source, cloudpickle, module):
    # State and definitions are one graph: restored instances must satisfy
    # isinstance(state, Memory) against the restored policy's own Memory class.
    blob = cloudpickle.dumps({"state": state, "policy": module}, protocol=5)
    return {"serializer": "cloudpickle-base64-v1", "data": base64.b64encode(blob).decode("ascii"),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _restore_snapshot(snapshot, source, cloudpickle):
    if not isinstance(snapshot, dict) or snapshot.get("serializer") != "cloudpickle-base64-v1":
        raise ContractError("invalid opaque state snapshot")
    blob = base64.b64decode(snapshot["data"], validate=True)
    if hashlib.sha256(blob).hexdigest() != snapshot.get("sha256"):
        raise ContractError("state snapshot checksum mismatch")
    if hashlib.sha256(source.encode()).hexdigest() != snapshot.get("source_sha256"):
        raise ContractError("state snapshot belongs to a different policy version")
    # This is ONLY called inside the isolated child, never by the host.
    return cloudpickle.loads(blob)


class _OutputCapture(io.TextIOBase):
    def __init__(self): self.parts = []; self.remaining = 8192
    def write(self, text):
        clipped = str(text)[:self.remaining]
        self.parts.append(clipped); self.remaining -= len(clipped)
        return len(text)


def _child(payload: dict[str, Any]) -> dict[str, Any]:
    source, operation = payload.get("source"), payload.get("operation")
    request = payload.get("payload", {})
    if operation not in {"initialize", "ingest", "search"} or not isinstance(request, dict):
        raise SandboxError("invalid sandbox operation")
    tree = validate_source(source)
    import cloudpickle  # Trusted dependency loaded before filesystem isolation.
    # Produced by the trusted parent, never accepted from policy state/config.
    paths = payload.get("runtime_paths") or _runtime_paths()
    _install_seccomp()
    _install_landlock(paths)
    module = types.ModuleType("candidate_policy")
    module.__dict__["__builtins__"] = builtins.__dict__
    sys.modules[module.__name__] = module
    capture = _OutputCapture()
    with contextlib.redirect_stdout(capture):
        exec(compile(tree, "<candidate-policy>", "exec"), module.__dict__, module.__dict__)
        cloudpickle.register_pickle_by_value(module)
        fn = module.__dict__.get(operation)
        if not callable(fn): raise ContractError("candidate missing callable entrypoint: " + operation)
        if operation == "initialize":
            for name, arity in (("initialize", 1), ("ingest", 2), ("search", 3)):
                entry = module.__dict__.get(name)
                if not callable(entry): raise ContractError("candidate missing callable entrypoint: " + name)
                try: inspect.signature(entry).bind(*([None] * arity))
                except (ValueError, TypeError) as exc: raise ContractError("candidate entrypoint signature mismatch: " + name) from exc
            value = fn(request.get("config", {}))
            result = {"state": _snapshot(value, source, cloudpickle, module), "serializable": True}
        else:
            bundle = _restore_snapshot(request.get("state"), source, cloudpickle)
            state, module = bundle["state"], bundle["policy"]
            sys.modules[module.__name__] = module
            cloudpickle.register_pickle_by_value(module)
            fn = module.__dict__.get(operation)
            if not callable(fn): raise ContractError("candidate missing callable entrypoint: " + operation)
            if operation == "ingest":
                value = fn(state, request.get("session", {}))
                result = {"state": _snapshot(value, source, cloudpickle, module), "serializable": True}
            else:
                # Never serialize or publish mutated query state.
                result = {"ids": fn(state, request.get("query", ""), request.get("filters", {}))}
    if capture.parts: result["stdout_excerpt"] = "".join(capture.parts)
    return result


def _install_seccomp() -> None:
    """Require Linux libseccomp and deny filesystem/network/process escape APIs."""
    if not sys.platform.startswith("linux"):
        raise IsolationError("seccomp worker is unavailable on this platform; candidate rejected")
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise IsolationError("libseccomp is unavailable; candidate rejected (no AST-only fallback)")
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    class ArgCompare(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                           ctypes.c_uint, ctypes.POINTER(ArgCompare)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None

    # SCMP_ACT_ALLOW is 0x7fff0000. Every listed syscall fails with EPERM.
    ctx = lib.seccomp_init(ctypes.c_uint32(0x7FFF0000))
    if not ctx:
        raise IsolationError("libseccomp could not initialize; candidate rejected")
    denied = {
        "openat2", "creat", "open_by_handle_at",
        "socket", "socketcall", "connect", "bind", "listen", "accept", "accept4",
        "sendto", "sendmsg", "recvfrom", "recvmsg",
        "execve", "execveat", "clone", "clone3", "fork", "vfork", "ptrace",
        "process_vm_readv", "process_vm_writev", "kill", "tkill", "tgkill", "pidfd_send_signal",
        "pidfd_open", "pidfd_getfd",
        "bpf", "userfaultfd", "unshare", "setns", "reboot", "kexec_load",
        "unlink", "unlinkat", "rename", "renameat", "renameat2", "mkdir", "mkdirat",
        "rmdir", "chmod", "fchmod", "fchmodat", "chown", "fchown", "fchownat",
        "symlink", "symlinkat", "link", "linkat", "mknod", "mknodat", "truncate",
        "ftruncate", "mount", "umount2", "pivot_root", "io_uring_setup", "memfd_create",
    }
    action_errno = 0x00050000 | (errno.EPERM & 0xFFFF)  # SCMP_ACT_ERRNO(EPERM)
    try:
        for name in sorted(denied):
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                # A syscall absent on this architecture is not callable here.
                continue
            if lib.seccomp_rule_add(ctx, action_errno, number, 0) != 0:
                raise IsolationError("libseccomp rule installation failed; candidate rejected")
        # ABI-1 Landlock does not mediate all truncate cases. Deny every
        # writing open flag while allowing ordinary read-only library imports.
        for name, arg in (("open", 1), ("openat", 2)):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0: continue
            for bit in (1, 2, os.O_CREAT, os.O_TRUNC, os.O_APPEND, 0x400000):
                compare = ArgCompare(arg, 7, bit, bit)  # SCMP_CMP_MASKED_EQ
                if lib.seccomp_rule_add_array(ctx, action_errno, number, 1, ctypes.byref(compare)) != 0:
                    raise IsolationError("read-only open filter installation failed")
        # Set no_new_privs before loading the filter, independent of privilege.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise IsolationError("PR_SET_NO_NEW_PRIVS failed; candidate rejected")
        if lib.seccomp_load(ctx) != 0:
            raise IsolationError("libseccomp filter load failed; candidate rejected")
    finally:
        lib.seccomp_release(ctx)


def _trusted_seccomp_test_probe() -> int:
    """Trusted test-only branch: confirm file and socket syscalls get EPERM."""
    paths = _runtime_paths()
    _install_seccomp()
    _install_landlock(paths)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.open.restype = ctypes.c_int
    libc.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libc.socket.restype = ctypes.c_int
    opened = libc.open(b"/etc/passwd", 0)
    open_errno = ctypes.get_errno()
    sock = libc.socket(2, 1, 0)  # AF_INET, SOCK_STREAM
    socket_errno = ctypes.get_errno()
    if opened != -1 or open_errno != errno.EPERM or sock != -1 or socket_errno != errno.EPERM:
        return 1
    return 0


def _child_main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        sys.stdout.write(json.dumps({"error": "sandbox input exceeds size limit"}))
        return 2
    try:
        payload = json.loads(raw)
        result = _child(payload)
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise ContractError("candidate output exceeds size limit")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:
        msg = str(exc)[:500] or type(exc).__name__
        category = getattr(exc, "category", "environment" if isinstance(exc, ImportError) else "runtime")
        encoded = json.dumps({"error": msg, "category": category,
                              "error_type": type(exc).__name__}, ensure_ascii=False).encode("utf-8")
        sys.stdout.buffer.write(encoded[:MAX_OUTPUT_BYTES])
        sys.stdout.buffer.flush()
        return 1


def _limits(memory_bytes: int, timeout_s: float):
    """Apply required Linux process limits; failure prevents worker startup."""
    import resource

    cpu_seconds = max(1, math.ceil(timeout_s))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    if not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("RLIMIT_AS is unavailable")
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def run_candidate(source: str, operation: str, payload: dict[str, Any], *,
                  timeout_s: float = 30.0, memory_bytes: int = 2 * 1024 * 1024 * 1024) -> dict[str, Any]:
    if not isolation_available():
        raise IsolationError("required Linux seccomp/Landlock/cloudpickle isolation is unavailable (fail-closed)")
    request = {"source": source, "operation": operation, "payload": payload,
               "runtime_paths": _runtime_paths()}
    try:
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("sandbox input must be JSON serializable") from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise ContractError("sandbox input exceeds size limit")
    package_root = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-m", "rsi.sandbox", "--child"]
    try:
        proc = subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=str(package_root), env={"OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}, timeout=max(0.05, timeout_s),
                              preexec_fn=(lambda: _limits(memory_bytes, timeout_s)) if os.name == "posix" else None,
                              check=False)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError("candidate exceeded wall-clock limit") from exc
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise IsolationError("required resource-limited worker could not start") from exc
    if len(proc.stdout) > MAX_OUTPUT_BYTES:
        raise ContractError("candidate output exceeds size limit")
    try:
        result = json.loads(proc.stdout or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("candidate returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ContractError("candidate result must be an object")
    if proc.returncode != 0 or "error" in result:
        raise SandboxError(str(result.get("error", "candidate process failed")), result.get("category", "runtime"))
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        raise SystemExit(_child_main())
    if sys.argv[1:] == ["--trusted-seccomp-test-probe"]:
        raise SystemExit(_trusted_seccomp_test_probe())
    raise SystemExit("This module is intended for GeneratedPolicy or --child execution")
