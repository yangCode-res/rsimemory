import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_sglang_replicas as launcher


class FakeProcess:
    next_pid = 4000

    def __init__(self, command, **kwargs):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.returncode or 0
        return self.returncode


def _fake_popen_factory(created):
    def fake_popen(command, **kwargs):
        proc = FakeProcess(command, **kwargs)
        created.append(proc)
        return proc
    return fake_popen


class FakeClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class ParallelLauncherTests(unittest.TestCase):
    def test_three_tp2_replicas_use_disjoint_allocated_devices_ports_and_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            devices = ["GPU-a", "GPU-b", "MIG-c", "MIG-d", "7", "9"]
            specs = launcher.replica_specs(tmp_path / "code", tmp_path / "scratch",
                                           tmp_path / "meta", devices)
            self.assertEqual([spec["port"] for spec in specs], [7862, 7863, 7864])
            self.assertEqual([spec["cuda_visible_devices"] for spec in specs], [
                ["GPU-a", "GPU-b"], ["MIG-c", "MIG-d"], ["7", "9"]])
            self.assertEqual(len({device for spec in specs
                                  for device in spec["cuda_visible_devices"]}), 6)
            self.assertEqual(len({spec["cache_dir"] for spec in specs}), 3)

            created = []
            procs = launcher.start_replicas(specs, popen=_fake_popen_factory(created))
            try:
                self.assertEqual(len(procs), 3)
                self.assertEqual([proc.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for proc in created], [
                    "GPU-a,GPU-b", "MIG-c,MIG-d", "7,9"])
                self.assertEqual([proc.kwargs["env"]["PORT"] for proc in created],
                                 ["7862", "7863", "7864"])
                self.assertTrue(all(proc.kwargs["env"]["TP_SIZE"] == "2" for proc in created))
                self.assertEqual([Path(proc.kwargs["stdout"].name).name for proc in created], [
                    "server-1.log", "server-2.log", "server-3.log"])
            finally:
                for proc in procs:
                    proc._rsi_log_file.close()

    def test_supervisor_startup_failure_cleans_every_launched_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            created = []
            signaled = []

            def signal_group(pid, sig):
                signaled.append((pid, sig))
                for proc in created:
                    if proc.pid == pid and sig == signal.SIGTERM:
                        proc.returncode = 0

            clock = FakeClock()
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}):
                result = launcher.supervise(
                    tmp_path / "code", tmp_path / "scratch", tmp_path / "meta",
                    startup_timeout_s=10, popen=_fake_popen_factory(created),
                    health_probe=lambda url: url.endswith(":7862/health"),
                    clock=clock, sleep=clock.sleep, signal_group=signal_group)

            self.assertEqual(result, 1)
            self.assertEqual(len(created), 3)
            self.assertTrue((tmp_path / "meta" / "replicas.json").exists())
            records = json.loads((tmp_path / "meta" / "replicas.json").read_text())["replicas"]
            self.assertEqual([record["pid"] for record in records],
                             [proc.pid for proc in created])
            self.assertTrue(all(proc.returncode is not None for proc in created))
            self.assertTrue(all((proc.pid, signal.SIGTERM) in signaled for proc in created))

    def test_ready_replica_exit_is_detected_while_peer_is_still_starting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            specs = launcher.replica_specs(tmp_path / "code", tmp_path / "scratch",
                                           tmp_path / "meta", list("abcdef"))
            procs = [FakeProcess([], pid=0) for _ in specs]
            procs[0].returncode = None
            died = False

            def probe(url):
                nonlocal died
                if url.endswith(":7863/health") and not died:
                    procs[0].returncode = 9
                    died = True
                return url.endswith(":7862/health")

            clock = FakeClock()
            with self.assertRaisesRegex(RuntimeError, "replica 1 exited during startup"):
                launcher.wait_until_ready(specs, procs, timeout_s=20,
                                          health_probe=probe, clock=clock,
                                          sleep=clock.sleep)

    def test_child_exit_immediately_after_readiness_fails_supervisor_and_cleans_peers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            created = []
            signaled = []

            def signal_group(pid, sig):
                signaled.append((pid, sig))
                for proc in created:
                    if proc.pid == pid and sig == signal.SIGTERM:
                        proc.returncode = 0

            original_write = Path.write_text

            def write_then_crash_ready(path, data, *args, **kwargs):
                result = original_write(path, data, *args, **kwargs)
                if path.name == "replicas-ready.json":
                    created[0].returncode = 9
                return result

            clock = FakeClock()
            with (patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}),
                  patch.object(Path, "write_text", autospec=True,
                               side_effect=write_then_crash_ready)):
                result = launcher.supervise(
                    tmp_path / "code", tmp_path / "scratch", tmp_path / "meta",
                    startup_timeout_s=10, popen=_fake_popen_factory(created),
                    health_probe=lambda _url: True, clock=clock, sleep=clock.sleep,
                    signal_group=signal_group)

            self.assertEqual(result, 1)
            self.assertTrue((tmp_path / "meta" / "replicas-ready.json").exists())
            self.assertTrue(all(proc.returncode is not None for proc in created))
            self.assertTrue(all((proc.pid, signal.SIGTERM) in signaled for proc in created[1:]))

    def test_allocated_device_validation_rejects_non_six_masks(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            launcher.allocated_devices("0,1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            launcher.allocated_devices("0,1,2,3,4,4")


if __name__ == "__main__":
    unittest.main()
