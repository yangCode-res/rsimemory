"""Supervise the model endpoint without treating one slow probe as a crash."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


@dataclass
class Probe:
    curl_exit_code: int
    http_status: int | None
    duration_s: float
    error: str = ""

    @property
    def healthy(self):
        return self.curl_exit_code == 0 and self.http_status == 200


def probe_health(url, timeout_s=30):
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--output", "/dev/null",
             "--write-out", "%{http_code}", "--connect-timeout", "3",
             "--max-time", str(timeout_s), url],
            capture_output=True, text=True, timeout=timeout_s + 2,
        )
        status = int(result.stdout.strip()) if result.stdout.strip().isdigit() else None
        return Probe(result.returncode, status, time.monotonic() - started,
                     result.stderr.strip()[:500])
    except subprocess.TimeoutExpired:
        return Probe(124, None, time.monotonic() - started, "probe subprocess timed out")
    except OSError as exc:
        return Probe(127, None, time.monotonic() - started, type(exc).__name__)


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # kill(pid, 0) also succeeds for an unreaped Linux zombie.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
        return state not in ("Z", "X")
    except (OSError, IndexError):
        return True


@dataclass
class FailureWindow:
    threshold: int = 3
    grace_s: float = 120
    failures: int = 0
    first_failure_at: float | None = None

    def observe(self, healthy, started, now):
        if healthy:
            self.failures = 0
            self.first_failure_at = None
            return False
        if self.first_failure_at is None:
            self.first_failure_at = started
        self.failures += 1
        return (self.failures >= self.threshold
                and now - self.first_failure_at >= self.grace_s)


def monitor(server_pid, client_pid, meta_dir, *, url="http://127.0.0.1:7862/health",
            timeout_s=30, interval_s=10, threshold=3, grace_s=120,
            probe=probe_health, alive=process_alive, clock=time.monotonic,
            sleep=time.sleep, terminate=None):
    if server_pid <= 1 or client_pid <= 1 or server_pid == client_pid:
        raise ValueError("Expected distinct server and replay process IDs greater than 1")
    if min(timeout_s, interval_s, threshold, grace_s) <= 0:
        raise ValueError("Monitor budgets must be positive")
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    window = FailureWindow(threshold, grace_s)

    def emit(event, **values):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
                  "server_pid": server_pid, "replay_pid": client_pid, **values}
        with (meta_dir / "service-health.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")

    def status(value):
        temporary = meta_dir / ".service-status.tmp"
        temporary.write_text(value + "\n")
        temporary.replace(meta_dir / "service-status")

    def stop(reason):
        status("unavailable")
        emit("replay_stop_requested", reason=reason)
        if terminate is not None:
            terminate(client_pid)
        else:
            try:
                os.killpg(client_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return 1

    emit("monitor_started", timeout_s=timeout_s, interval_s=interval_s,
         failure_threshold=threshold, failure_grace_s=grace_s, endpoint=url)
    while alive(client_pid):
        if not alive(server_pid):
            return stop("server_process_exited")
        started = clock()
        result = probe(url, timeout_s)
        # A probe completing after replay exits must not change its outcome.
        if not alive(client_pid):
            break
        if not alive(server_pid):
            emit("probe_result", **asdict(result), server_alive=False)
            return stop("server_process_exited")
        should_stop = window.observe(result.healthy, started, clock())
        emit("probe_result", **asdict(result), healthy=result.healthy,
             consecutive_failures=window.failures,
             failure_elapsed_s=(clock() - window.first_failure_at
                                if window.first_failure_at is not None else 0),
             server_alive=True)
        status("healthy" if result.healthy else "degraded")
        if should_stop:
            return stop("persistent_health_failures")
        sleep(interval_s)
    emit("replay_exited")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--client-pid", type=int, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7862/health")
    parser.add_argument("--timeout-s", type=float, default=30)
    parser.add_argument("--interval-s", type=float, default=10)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--grace-s", type=float, default=120)
    args = parser.parse_args()
    return monitor(args.server_pid, args.client_pid, args.meta_dir, url=args.url,
                   timeout_s=args.timeout_s, interval_s=args.interval_s,
                   threshold=args.threshold, grace_s=args.grace_s)


if __name__ == "__main__":
    raise SystemExit(main())
