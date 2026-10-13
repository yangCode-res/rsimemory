"""Subprocess runner for the deliberately small generated-policy dialect.

The AST checks and resource limits reduce accidental and obvious hostile
behavior. Python is not a security boundary; use an OS container/VM for truly
untrusted code.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

MAX_SOURCE_BYTES = 128 * 1024
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024


class SandboxError(RuntimeError):
    """Candidate rejected, failed, or exceeded its execution bounds."""


SAFE_BUILTINS = {
    "sorted": sorted, "len": len, "min": min, "max": max, "sum": sum,
    "range": range, "set": set, "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "bool": bool, "enumerate": enumerate,
    "zip": zip, "abs": abs, "round": round,
}

SAFE_METHODS = {
    # string methods
    "lower", "upper", "strip", "lstrip", "rstrip", "split", "rsplit",
    "join", "replace", "startswith", "endswith", "find", "count",
    "title", "capitalize", "casefold",
    # list methods
    "append", "extend", "insert", "pop", "remove", "reverse", "sort",
    "copy", "index",
    # dict methods
    "get", "keys", "values", "items", "update", "setdefault",
}

SAFE_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.Expr, ast.For, ast.If, ast.Pass, ast.Break, ast.Continue,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict,
    ast.Set, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.BinOp,
    ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


def validate_source(source: str) -> ast.Module:
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SandboxError("candidate source exceeds size limit")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise SandboxError("candidate source is not valid restricted Python") from exc
    functions: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef):
            raise SandboxError("only top-level function definitions are permitted")
        if stmt.name.startswith("_") or stmt.name in functions:
            raise SandboxError("private or duplicate function names are forbidden")
        if stmt.name in SAFE_BUILTINS:
            raise SandboxError("candidate function names cannot shadow safe builtins")
        if stmt.decorator_list or stmt.returns is not None or stmt.type_comment:
            raise SandboxError("decorators and annotations are forbidden")
        args = stmt.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults:
            raise SandboxError("variadic and default arguments are forbidden")
        if args.posonlyargs:
            raise SandboxError("positional-only arguments are forbidden")
        if any(arg.arg.startswith("_") for arg in args.args):
            raise SandboxError("private argument names are forbidden")
        functions.add(stmt.name)
    if not {"initialize", "ingest", "search"}.issubset(functions):
        raise SandboxError("candidate must define initialize, ingest, and search")
    expected_arity = {"initialize": 1, "ingest": 2, "search": 3}
    for stmt in tree.body:
        if stmt.name in expected_arity and len(stmt.args.args) != expected_arity[stmt.name]:
            raise SandboxError(f"candidate {stmt.name} has the wrong function signature")

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODES):
            raise SandboxError("candidate uses syntax outside the restricted dialect")
        if isinstance(node, ast.FunctionDef) and node not in tree.body:
            raise SandboxError("nested function definitions are forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise SandboxError("dunder and private names are forbidden")
        if isinstance(node, ast.Attribute):
            # No object attributes, module paths, or dunders. Only these narrow
            # data methods can be called on JSON strings/lists/dictionaries.
            if node.attr.startswith("_") or node.attr not in SAFE_METHODS:
                raise SandboxError("candidate attribute access is forbidden")
            parent = parents.get(node)
            if not isinstance(parent, ast.Call) or parent.func is not node:
                raise SandboxError("candidate cannot retrieve method objects")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in SAFE_BUILTINS and name not in functions:
                    raise SandboxError("candidate call is not allowlisted")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_METHODS:
                    raise SandboxError("candidate method call is not allowlisted")
            else:
                raise SandboxError("indirect calls are forbidden")
    return tree


def _child(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    operation = payload.get("operation")
    request = payload.get("payload", {})
    if operation not in {"initialize", "ingest", "search"} or not isinstance(request, dict):
        raise SandboxError("invalid sandbox operation")
    tree = validate_source(source)
    # Imports and JSON parsing are complete. Install the OS filter before
    # compiling or running any candidate statements. No AST-only fallback.
    _install_seccomp()
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    # Candidate source is compiled only after the restricted AST has passed.
    exec(compile(tree, "<policy-candidate>", "exec"), namespace, namespace)
    if operation == "initialize":
        value = namespace["initialize"](request.get("config", {}))
        return {"state": value}
    state = request.get("state")
    if operation == "ingest":
        value = namespace["ingest"](state, request.get("session", {}))
        return {"state": value}
    value = namespace["search"](state, request.get("query", ""), request.get("filters", {}))
    return {"ids": value}


def _install_seccomp() -> None:
    """Require Linux libseccomp and deny filesystem/network/process escape APIs."""
    if not sys.platform.startswith("linux"):
        raise SandboxError("seccomp worker is unavailable on this platform; candidate rejected")
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise SandboxError("libseccomp is unavailable; candidate rejected (no AST-only fallback)")
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None

    # SCMP_ACT_ALLOW is 0x7fff0000. Every listed syscall fails with EPERM.
    ctx = lib.seccomp_init(ctypes.c_uint32(0x7FFF0000))
    if not ctx:
        raise SandboxError("libseccomp could not initialize; candidate rejected")
    denied = {
        "open", "openat", "openat2", "creat", "open_by_handle_at",
        "socket", "socketcall", "connect", "bind", "listen", "accept", "accept4",
        "sendto", "sendmsg", "recvfrom", "recvmsg",
        "execve", "execveat", "clone", "clone3", "fork", "vfork", "ptrace",
        "process_vm_readv", "process_vm_writev",
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
                raise SandboxError("libseccomp rule installation failed; candidate rejected")
        # Set no_new_privs before loading the filter, independent of privilege.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise SandboxError("PR_SET_NO_NEW_PRIVS failed; candidate rejected")
        if lib.seccomp_load(ctx) != 0:
            raise SandboxError("libseccomp filter load failed; candidate rejected")
    finally:
        lib.seccomp_release(ctx)


def _trusted_seccomp_test_probe() -> int:
    """Trusted test-only branch: confirm file and socket syscalls get EPERM."""
    _install_seccomp()
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
            raise SandboxError("candidate output exceeds size limit")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:
        msg = str(exc)[:500] or type(exc).__name__
        encoded = json.dumps({"error": msg}, ensure_ascii=False).encode("utf-8")
        sys.stdout.buffer.write(encoded[:MAX_OUTPUT_BYTES])
        sys.stdout.buffer.flush()
        return 1


def _limits(memory_bytes: int):
    """Apply required Linux process limits; failure prevents worker startup."""
    import resource

    cpu_seconds = 2
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    if not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("RLIMIT_AS is unavailable")
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def run_candidate(source: str, operation: str, payload: dict[str, Any], *,
                  timeout_s: float = 2.0, memory_bytes: int = 128 * 1024 * 1024) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or not ctypes.util.find_library("seccomp"):
        raise SandboxError("required Linux libseccomp sandbox is unavailable; candidate rejected (fail-closed)")
    request = {"source": source, "operation": operation, "payload": payload}
    try:
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SandboxError("sandbox input must be JSON serializable") from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise SandboxError("sandbox input exceeds size limit")
    package_root = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-m", "rsi.sandbox", "--child"]
    try:
        proc = subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=str(package_root), env={}, timeout=max(0.05, timeout_s),
                              preexec_fn=(lambda: _limits(memory_bytes)) if os.name == "posix" else None,
                              check=False)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError("candidate exceeded wall-clock limit") from exc
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise SandboxError("required resource-limited worker could not start") from exc
    if len(proc.stdout) > MAX_OUTPUT_BYTES:
        raise SandboxError("candidate output exceeds size limit")
    try:
        result = json.loads(proc.stdout or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SandboxError("candidate returned invalid JSON") from exc
    if proc.returncode != 0 or "error" in result:
        raise SandboxError(str(result.get("error", "candidate process failed")))
    if not isinstance(result, dict):
        raise SandboxError("candidate result must be an object")
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        raise SystemExit(_child_main())
    if sys.argv[1:] == ["--trusted-seccomp-test-probe"]:
        raise SystemExit(_trusted_seccomp_test_probe())
    raise SystemExit("This module is intended for GeneratedPolicy or --child execution")
