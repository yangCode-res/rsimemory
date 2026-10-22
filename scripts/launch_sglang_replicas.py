#!/usr/bin/env python3
"""Launch and supervise three independent TP=2 SGLang replicas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


PORTS = (7862, 7863, 7864)


def allocated_devices(value: str | None = None) -> list[str]:
    """Return six Slurm-assigned device identifiers without assuming GPU ordinals."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    devices = [item.strip() for item in (value or "").split(",") if item.strip()]
    if len(devices) != 6:
        raise ValueError(
            "Expected exactly six allocated CUDA_VISIBLE_DEVICES entries; "
            f"received {len(devices)}"
        )
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device identifiers")
    return devices


def replica_specs(root: Path, local_root: Path, meta_dir: Path,
                  devices: list[str]) -> list[dict]:
    specs = []
    for index, port in enumerate(PORTS, start=1):
        replica_devices = devices[(index - 1) * 2:index * 2]
        cache = local_root / "cache" / f"replica-{index}"
        tmp = local_root / "tmp" / f"replica-{index}"
        replica_meta = meta_dir / "services" / f"replica-{index}"
        for path in (cache, tmp, replica_meta):
            path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(replica_devices),
            "PORT": str(port), "HOST": "127.0.0.1", "TP_SIZE": "2",
            "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(cache),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
            "SGLANG_CACHE_DIR": str(cache / "sglang"),
            "SGLANG_DG_CACHE_DIR": str(cache / "deep_gemm"),
            "CUTE_DSL_CACHE_DIR": str(cache / "cute"),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        })
        specs.append({
            "replica": index,
            "cuda_visible_devices": replica_devices,
            "port": port,
            "health_url": f"http://127.0.0.1:{port}/health",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "cache_dir": str(cache),
            "log": str(local_root / "artifacts" / f"server-{index}.log"),
            "meta_dir": str(replica_meta),
            "command": ["bash", str(root / "scripts" / "launch_sglang_local.sh")],
            "env": env,
        })
    return specs


def probe_health(url: str, timeout_s: float = 3) -> bool:
    try:
        with urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def write_metadata(meta_dir: Path, specs: list[dict], procs: list,
                   devices: list[str]) -> None:
    records = []
    for spec, proc in zip(specs, procs):
        records.append({key: spec[key] for key in (
            "replica", "cuda_visible_devices", "port", "health_url", "base_url",
            "cache_dir", "log", "meta_dir") } | {"pid": proc.pid})
    payload = {"allocated_cuda_visible_devices": devices, "replicas": records}
    meta_dir.mkdir(parents=True, exist_ok=True)
    tmp = meta_dir / ".replicas.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(meta_dir / "replicas.json")


def stop_processes(procs: list, *, signal_group=os.killpg, sleep=time.sleep) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                signal_group(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    sleep(2)
    for proc in procs:
        if proc.poll() is None:
            try:
                signal_group(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass


def start_replicas(specs: list[dict], *, popen=subprocess.Popen) -> list:
    procs = []
    try:
        for spec in specs:
            log_path = Path(spec["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("ab", buffering=0)
            try:
                proc = popen(spec["command"], cwd=str(Path(spec["command"][1]).parent.parent),
                             env=spec["env"], stdout=log_file, stderr=subprocess.STDOUT,
                             start_new_session=True)
            except BaseException:
                log_file.close()
                raise
            # The parent retains file handles so each log remains available until shutdown.
            proc._rsi_log_file = log_file
            procs.append(proc)
        return procs
    except BaseException:
        stop_processes(procs)
        raise


def wait_until_ready(specs: list[dict], procs: list, *, timeout_s: float,
                     health_probe=probe_health, clock=time.monotonic,
                     sleep=time.sleep, interval_s=5) -> None:
    pending = {i for i in range(len(specs))}
    deadline = clock() + timeout_s
    while pending and clock() < deadline:
        for index, proc in enumerate(procs):
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang replica {index + 1} exited during startup")
        for index in tuple(pending):
            if health_probe(specs[index]["health_url"]):
                pending.remove(index)
        if pending:
            sleep(min(interval_s, max(0, deadline - clock())))
    if pending:
        replicas = ", ".join(str(i + 1) for i in sorted(pending))
        raise TimeoutError(f"SGLang replicas did not become healthy before timeout: {replicas}")
    for index, proc in enumerate(procs):
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang replica {index + 1} exited before readiness was recorded")


def supervise(root: Path, local_root: Path, meta_dir: Path, *, startup_timeout_s=1800,
              popen=subprocess.Popen, health_probe=probe_health, clock=time.monotonic,
              sleep=time.sleep, signal_group=os.killpg) -> int:
    devices = allocated_devices()
    specs = replica_specs(root, local_root, meta_dir, devices)
    procs = []
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        procs = start_replicas(specs, popen=popen)
        write_metadata(meta_dir, specs, procs, devices)
        wait_until_ready(specs, procs, timeout_s=startup_timeout_s,
                         health_probe=health_probe, clock=clock, sleep=sleep)
        ready = meta_dir / "replicas-ready.json"
        ready.write_text(json.dumps({"healthy": True, "replicas": [
            {"replica": spec["replica"], "health_url": spec["health_url"]}
            for spec in specs]}, indent=2) + "\n")
        print("ALL_REPLICAS_READY", flush=True)
        while True:
            for index, proc in enumerate(procs, start=1):
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(f"SGLang replica {index} exited with status {code}")
            sleep(5)
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        print(f"REPLICA_SUPERVISOR_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_processes(procs, signal_group=signal_group, sleep=sleep)
        for proc in procs:
            log_file = getattr(proc, "_rsi_log_file", None)
            if log_file:
                log_file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    args = parser.parse_args()
    return supervise(args.root, args.local_root, args.meta_dir,
                     startup_timeout_s=args.startup_timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
