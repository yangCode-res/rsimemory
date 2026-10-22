from contextlib import ExitStack, redirect_stdout
import io
import json
import tempfile
import threading
from pathlib import Path
import unittest
from unittest import mock

from rsi import cli
from rsi.model import ModelServiceUnavailable
from rsi.policy import PlainPolicy


def report(score):
    return {
        "AVG": score,
        "E2E_AVG": score,
        "abnormal_rate": 0.0,
        "counts": {},
        "cost": {"policy": [], "search_seconds": 0.0},
    }


class EndpointClient:
    def __init__(self, endpoint):
        self.base_url = endpoint
        self.calls = []


class ParallelCandidateTests(unittest.TestCase):
    def _run_fixture(self, output, *, replay_impl, generated_policy=None,
                     candidate_clients=None, candidate_workers=3):
        proposals = [{"id": f"p{i}", "basis": "fixture", "rationale": "fixture"}
                     for i in range(1, 4)]
        endpoints = ["http://replica-a/v1", "http://replica-b/v1", "http://replica-c/v1"]
        argv = ["rsi", "--fixture", "--evolve", "--no-diagnosis",
                "--max-interventions", "0", "--output", str(output),
                "--candidate-workers", str(candidate_workers), "--candidate-base-urls", *endpoints]
        patches = [
            mock.patch.object(cli, "reflect", return_value={"proposals": proposals}),
            mock.patch.object(cli, "generate", side_effect=lambda client, proposal, *args, **kwargs:
                              (f"source-{proposal['id']}", {"calls": []})),
            mock.patch.object(cli, "replay", side_effect=replay_impl),
            mock.patch.object(cli, "commit_policy", side_effect=lambda repo, source, label, parent=None:
                              f"commit-{label}"),
            mock.patch.object(cli.signal, "signal"),
            mock.patch("sys.argv", argv),
        ]
        if generated_policy is not None:
            patches.append(mock.patch.object(cli, "GeneratedPolicy", side_effect=generated_policy))
        if candidate_clients is not None:
            patches.append(mock.patch.object(cli, "_candidate_clients", side_effect=candidate_clients))
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            with redirect_stdout(io.StringIO()):
                cli.main()
        return json.loads((output / "summary.json").read_text()), json.loads((output / "tree.json").read_text())

    def test_three_replicas_overlap_and_merge_in_proposal_order_with_three_visits(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            barrier = threading.Barrier(3)
            pairs = {}
            calls_seen = []
            def clients(args, config, endpoint, actor_client, judge_client, cancelled):
                pair = (EndpointClient(endpoint), EndpointClient(endpoint))
                pairs[endpoint] = pair
                return pair

            def replay_impl(sessions, queries, ids, factory, actor, judge, config, output_path):
                path = Path(output_path)
                if path.parts[-2] == "plain-bm25":
                    return report(.5), []
                candidate_index = int(path.parent.name.split("-")[-1])
                self.assertEqual(actor.base_url, f"http://replica-{'abc'[candidate_index - 1]}/v1")
                self.assertEqual(judge.base_url, actor.base_url)
                self.assertIsNot(actor, judge)
                calls_seen.append((path.parent.name, path.name, id(actor), id(judge)))
                actor.calls.append({"role": "actor", "latency_s": .01})
                judge.calls.append({"role": "judge", "latency_s": .02})
                if path.name == "train":
                    barrier.wait(timeout=10)
                    return report(.2 + candidate_index * .1), []
                return report(.6), []

            summary, tree = self._run_fixture(
                output, replay_impl=replay_impl, generated_policy=lambda *a, **kw: PlainPolicy(),
                candidate_clients=clients)

            self.assertEqual(summary["status"], "completed")
            branches = summary["branches"]
            self.assertEqual([branch["id"] for branch in branches],
                             ["candidate-1", "candidate-2", "candidate-3"])
            self.assertTrue(all(branch["status"] == "evaluated" for branch in branches))
            self.assertEqual(tree["nodes"]["plain-bm25"]["visits"], 3)
            self.assertAlmostEqual(tree["nodes"]["plain-bm25"]["Q"], .4)
            self.assertEqual([tree["nodes"][name]["inference_endpoint"] for name in
                              ("candidate-1", "candidate-2", "candidate-3")],
                             ["http://replica-a/v1", "http://replica-b/v1", "http://replica-c/v1"])
            self.assertEqual(len(calls_seen), 6)
            self.assertEqual(len({(actor_id, judge_id) for _, _, actor_id, judge_id in calls_seen}), 3)
            self.assertTrue(all(actor_id != judge_id for _, _, actor_id, judge_id in calls_seen))
            self.assertTrue(all(len(actor.calls) == 2 and len(judge.calls) == 2
                                for actor, judge in pairs.values()))

    def test_candidate_contract_failure_does_not_cancel_other_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            barrier = threading.Barrier(2)
            def generated(source, config, **kwargs):
                if source == "source-p2":
                    exc = ValueError("candidate contract rejected")
                    exc.category = "contract"
                    raise exc
                return PlainPolicy()

            def replay_impl(sessions, queries, ids, factory, actor, judge, config, output_path):
                path = Path(output_path)
                if path.parts[-2] == "plain-bm25":
                    return report(.5), []
                if path.name == "train":
                    barrier.wait(timeout=10)
                return report(.7), []

            summary, tree = self._run_fixture(output, replay_impl=replay_impl,
                                              generated_policy=generated)
            self.assertEqual(summary["status"], "completed")
            by_id = {branch["id"]: branch for branch in summary["branches"]}
            self.assertEqual([by_id[f"candidate-{i}"]["status"] for i in (1, 2, 3)],
                             ["evaluated", "invalid", "evaluated"])
            self.assertEqual(by_id["candidate-2"]["failure_category"], "contract")
            self.assertEqual(tree["nodes"]["plain-bm25"]["visits"], 2)
            self.assertEqual([branch["id"] for branch in summary["branches"]],
                             ["candidate-1", "candidate-2", "candidate-3"])

    def test_transport_failure_cancels_peer_replays_and_marks_round_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            barrier = threading.Barrier(3)
            def replay_impl(sessions, queries, ids, factory, actor, judge, config, output_path):
                path = Path(output_path)
                if path.parts[-2] == "plain-bm25":
                    return report(.5), []
                if path.name == "train":
                    barrier.wait(timeout=10)
                    candidate = path.parent.name
                    if candidate == "candidate-1":
                        raise ModelServiceUnavailable("replica unavailable")
                    if not actor.cancelled.wait(timeout=5):
                        self.fail("peer failure did not cancel the replay")
                    raise ModelServiceUnavailable("peer replay cancelled")
                return report(.6), []

            with self.assertRaises(ModelServiceUnavailable):
                self._run_fixture(output, replay_impl=replay_impl,
                                  generated_policy=lambda *a, **kw: PlainPolicy())
            status = json.loads((output / "run-status.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["exception_type"], "ModelServiceUnavailable")
            self.assertTrue(status["stage"].startswith("candidate-"))
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
