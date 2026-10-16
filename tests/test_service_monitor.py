import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from scripts.service_monitor import FailureWindow, Probe, monitor, probe_health


class ServiceMonitorTests(unittest.TestCase):
    def test_recovery_resets_both_failure_count_and_grace_period(self):
        window = FailureWindow(threshold=3, grace_s=120)
        for started in (0, 40, 80):
            self.assertFalse(window.observe(False, started, started + 30))
        self.assertFalse(window.observe(True, 115, 116))
        self.assertEqual(window.failures, 0)
        self.assertFalse(window.observe(False, 200, 230))
        self.assertEqual(window.first_failure_at, 200)

    def test_persistent_outage_requires_count_and_elapsed_time(self):
        window = FailureWindow(threshold=3, grace_s=120)
        self.assertFalse(window.observe(False, 0, 1))
        self.assertFalse(window.observe(False, 121, 122))
        self.assertTrue(window.observe(False, 123, 124))

    def test_probe_preserves_status_exit_code_and_error_without_response_body(self):
        with mock.patch("scripts.service_monitor.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=28, stdout="000", stderr="timed out")
            result = probe_health("http://127.0.0.1/health")
        self.assertFalse(result.healthy)
        self.assertEqual(result.curl_exit_code, 28)
        self.assertEqual(result.error, "timed out")
        command = run.call_args.args[0]
        self.assertIn("30", command)
        self.assertEqual(command[command.index("--output") + 1], "/dev/null")

    def test_real_http_response_slower_than_old_five_second_budget_is_healthy(self):
        class SlowHealth(BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(5.2)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), SlowHealth) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = probe_health(f"http://127.0.0.1:{server.server_port}/health")
            finally:
                server.shutdown()
                thread.join()
        self.assertTrue(result.healthy)
        self.assertGreater(result.duration_s, 5)

    def run_sequence(self, outcomes, *, server_alive=True, grace_s=120):
        stopped = []
        active = True
        now = 0
        outcomes = iter(outcomes)

        def probe(url, timeout):
            nonlocal now
            item = next(outcomes)
            now += item.duration_s
            return item

        def sleep(seconds):
            nonlocal now, active
            now += seconds
            # Stop naturally once a successful response arrives.
            if (Path(temp) / "service-status").read_text().strip() == "healthy":
                active = False

        with tempfile.TemporaryDirectory() as temp:
            result = monitor(1001, 1002, temp, probe=probe,
                             alive=lambda pid: active if pid == 1002 else server_alive,
                             clock=lambda: now, sleep=sleep, terminate=stopped.append,
                             grace_s=grace_s)
            status = (Path(temp) / "service-status").read_text().strip()
            events = [json.loads(line) for line in
                      (Path(temp) / "service-health.jsonl").read_text().splitlines()]
        return result, status, stopped, events

    def test_transient_timeout_and_503_do_not_stop_replay(self):
        result, status, stopped, events = self.run_sequence([
            Probe(28, 0, 30, "timeout"), Probe(0, 503, 20), Probe(0, 200, 8)])
        self.assertEqual((result, status, stopped), (0, "healthy", []))
        self.assertEqual([e["consecutive_failures"] for e in events
                          if e["event"] == "probe_result"], [1, 2, 0])

    def test_repeated_timeouts_eventually_terminate_replay(self):
        result, status, stopped, events = self.run_sequence([Probe(28, 0, 30)] * 4)
        self.assertEqual((result, status, stopped), (1, "unavailable", [1002]))
        self.assertEqual(events[-1]["reason"], "persistent_health_failures")
        self.assertGreaterEqual(events[-2]["failure_elapsed_s"], 120)

    def test_dead_server_stops_immediately_without_waiting_for_grace(self):
        result, status, stopped, events = self.run_sequence([], server_alive=False)
        self.assertEqual((result, status, stopped), (1, "unavailable", [1002]))
        self.assertEqual(events[-1]["reason"], "server_process_exited")

    def test_replay_exit_during_probe_does_not_trigger_termination(self):
        stopped = []
        active = True

        def probe(url, timeout):
            nonlocal active
            active = False
            return Probe(28, 0, 30)

        with tempfile.TemporaryDirectory() as temp:
            result = monitor(1001, 1002, temp, probe=probe,
                             alive=lambda pid: active if pid == 1002 else True,
                             terminate=stopped.append)
            events = (Path(temp) / "service-health.jsonl").read_text()
        self.assertEqual(result, 0)
        self.assertEqual(stopped, [])
        self.assertNotIn("replay_stop_requested", events)


if __name__ == "__main__":
    unittest.main()
