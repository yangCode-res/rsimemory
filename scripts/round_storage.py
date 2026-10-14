#!/usr/bin/env python3
"""Node-local scratch selection and resilient archival for full-round runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

NETWORK_FS = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "sshfs", "ceph", "cephfs",
    "fuse.sshfs", "glusterfs", "lustre", "beegfs", "panfs", "gpfs", "9p", "afs",
}
EXCLUDED_DIRS = {"__pycache__", ".cache", "cache", "caches", "embedding_cache"}
MANIFEST = "_archive-manifest.json"


class SourceChangedError(RuntimeError):
    """The source file changed during a bounded archive attempt."""


def filesystem_type(path: Path) -> str:
    """Return the mount type containing path, using findmnt or /proc/mounts."""
    try:
        result = subprocess.run(["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"],
                                check=True, capture_output=True, text=True)
        value = result.stdout.strip().splitlines()
        if value:
            return value[0]
    except (OSError, subprocess.CalledProcessError):
        pass
    mounts = Path("/proc/mounts")
    if mounts.exists():
        best = (0, "")
        for line in mounts.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mountpoint = fields[1].replace("\\040", " ")
            try:
                if os.path.commonpath((str(path.resolve()), mountpoint)) == mountpoint and len(mountpoint) > best[0]:
                    best = (len(mountpoint), fields[2])
            except (OSError, ValueError):
                continue
        if best[1]:
            return best[1]
    raise OSError(f"cannot determine filesystem type for {path}")


def _free_bytes(path: Path) -> int:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize


def create_scratch(job_id: str, min_free_bytes: int = 10 * 1024**3,
                   candidates: list[str] | None = None) -> tuple[Path, int]:
    """Create an owner-only scratch directory on a non-network filesystem."""
    candidates = candidates or [os.environ.get("SLURM_TMPDIR", ""),
                                os.environ.get("TMPDIR", ""), "/tmp"]
    seen: set[str] = set()
    problems: list[str] = []
    for raw in candidates:
        if not raw:
            continue
        root = Path(raw).expanduser()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not root.is_dir():
                raise OSError("not a directory")
            fs_type = filesystem_type(root)
            if fs_type.lower() in NETWORK_FS:
                raise OSError(f"network filesystem {fs_type}")
            free = _free_bytes(root)
            if free < min_free_bytes:
                raise OSError(f"only {free} free bytes; need {min_free_bytes}")
            path = Path(tempfile.mkdtemp(prefix=f"rsi-{job_id}-", dir=root))
            os.chmod(path, 0o700)
            probe = path / ".storage-probe"
            with probe.open("wb") as stream:
                stream.write(b"rsi-local-scratch\n")
                stream.flush()
                os.fsync(stream.fileno())
            if probe.read_bytes() != b"rsi-local-scratch\n":
                raise OSError("scratch write/read probe failed")
            probe.unlink()
            return path, free
        except (OSError, PermissionError) as exc:
            problems.append(f"{root}: {exc}")
    raise OSError("no suitable local scratch directory; " + "; ".join(problems))


def _files(root: Path):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and
                         not (Path(base) / d).is_symlink())
        for name in sorted(names):
            if name.startswith(".rsi-atomic-") and name.endswith(".tmp"):
                continue
            path = Path(base) / name
            if name == MANIFEST or path.is_symlink() or not path.is_file():
                continue
            yield path


def _copy_hash(source: Path, temp: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, temp.open("wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)
            digest.update(block)
            size += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    return digest.hexdigest(), size


def _hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _atomic_copy(source: Path, target: Path, retries: int, retry_delay: float,
                 require_stable: bool) -> dict:
    last_error = None
    for attempt in range(retries):
        temp = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                             dir=target.parent)
            os.close(fd)
            temp = Path(temp_name)
            digest, size = _copy_hash(source, temp)
            after = source.stat()
            stable = (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
                      and before.st_ino == after.st_ino)
            if require_stable and not stable:
                raise SourceChangedError(f"source changed while copying: {source}")
            os.replace(temp, target)
            return {"sha256": digest, "size": size, "stable": stable}
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
            if attempt + 1 < retries:
                time.sleep(retry_delay * (attempt + 1))
    if isinstance(last_error, SourceChangedError):
        raise last_error
    raise OSError(f"copy failed after {retries} attempts for {source}: {last_error}") from last_error


def archive(source: Path, destination: Path, *, final: bool,
            retries: int = 5, retry_delay: float = 0.25) -> dict:
    """Archive configured results; call final mode only after all writers stop."""
    if retries < 1:
        raise ValueError("retries must be at least 1")
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise OSError(f"source result directory does not exist: {source}")
    # Prevent recursive copies or reading outside the explicitly configured source.
    if destination == source or source in destination.parents:
        raise ValueError("destination must not be inside source")
    # Any later archive mutation invalidates an earlier verified snapshot.
    manifest_path = destination / MANIFEST
    manifest_path.unlink(missing_ok=True)
    copied, skipped, failed = {}, [], []
    for src in _files(source):
        relative = src.relative_to(source).as_posix()
        try:
            item = _atomic_copy(src, destination / relative, retries, retry_delay,
                                require_stable=True)
            if not item["stable"]:
                raise SourceChangedError(f"source changed while copying: {src}")
            copied[relative] = {"sha256": item["sha256"], "size": item["size"]}
        except SourceChangedError as exc:
            if final:
                failed.append(f"{relative}: {exc}")
            else:
                skipped.append(relative)
        except OSError as exc:
            failed.append(f"{relative}: {exc}")
    if final and not failed:
        for relative, expected in copied.items():
            try:
                actual = _hash(destination / relative)
                if actual != (expected["sha256"], expected["size"]):
                    failed.append(f"{relative}: destination checksum/size mismatch")
            except OSError as exc:
                failed.append(f"{relative}: destination verification failed: {exc}")
    result = {"consistent": bool(final and not failed), "copied_files": len(copied),
              "skipped_changed": skipped, "failures": failed}
    if final and not failed:
        manifest = {"format": 1, "verified": True, "files": copied}
        _publish_manifest(manifest_path, manifest, retries, retry_delay)
    if failed:
        raise OSError(json.dumps(result, sort_keys=True))
    return result


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _publish_manifest(path: Path, value: dict, retries: int, retry_delay: float) -> None:
    last_error = None
    for attempt in range(retries):
        try:
            _write_json_atomic(path, value)
            # Verify the published manifest itself is readable and unchanged.
            if json.loads(path.read_text()) != value:
                raise OSError("archive manifest verification failed")
            return
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(retry_delay * (attempt + 1))
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    raise OSError(f"manifest publication/verification failed after {retries} attempts: {last_error}") from last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create node-local scratch space")
    create.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    create.add_argument("--min-free-gib", type=float, default=10)
    sync = sub.add_parser("sync", help="best-effort atomic snapshot while the run is active")
    finish = sub.add_parser("finalize", help="verify and archive after the result writer stops")
    for command in (sync, finish):
        command.add_argument("--source", required=True)
        command.add_argument("--destination", required=True)
        command.add_argument("--retries", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            path, free = create_scratch(args.job_id, int(args.min_free_gib * 1024**3))
            result = {"path": str(path), "free_bytes_at_creation": free, "mode": "0700"}
        else:
            result = archive(Path(args.source), Path(args.destination),
                             final=args.command == "finalize", retries=args.retries)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"round_storage: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
