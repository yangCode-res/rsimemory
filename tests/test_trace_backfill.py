import json
from pathlib import Path
import tempfile
import unittest

from rsi.trace_backfill import build_backfill, write_backfill


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def answer(i: int, score: int):
    return {
        "persona_id": 100 + i,
        "query_id": f"p{100+i}-s{i+1}-q0",
        "session_id": i + 1,
        "query_type": "synthetic",
        "question": f"synthetic question {i}",
        "answer": f"synthetic answer {i}",
        "actor_status": "completed",
        "trajectory": [{"step": 1, "text": f"trajectory {i}"}],
        "evidence": [{"id": f"e{i}", "text": f"evidence {i}"}],
        "tool_audit": [{"action": "search", "ids": [f"e{i}"]}],
        "score": score,
        "status": "scored",
        "reason": "synthetic reason",
        "call_log": [{"role": "actor", "usage": {"prompt_tokens": 9}}],
    }


class TraceBackfillTests(unittest.TestCase):
    def make_run(self, root: Path):
        run = root / "full-v03-local-146133" / "result"
        dump(run / "config.json", {"model": "recorded-model"})
        dump(run / "config-hash.json", {"sha256": "a" * 64})
        dump(run / "summary.json", {"test_evaluated": False, "comparison": {"plain-bm25": {"AVG": 0.4}}})
        (run / "policy-git").mkdir()
        (run / "policy-git" / "plain-bm25.py").write_text("# recorded historical parent\n", encoding="utf-8")
        answers = [answer(i, 0) for i in range(5)] + [answer(i + 5, 1) for i in range(3)]
        path = run / "plain-bm25" / "train" / "answers.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
        validation_path = run / "plain-bm25" / "validation" / "answers.jsonl"
        validation_path.parent.mkdir(parents=True)
        validation_path.write_text(json.dumps(answer(20, 1)) + "\n", encoding="utf-8")
        dump(run / "reflexion.json", {
            "clusters": [{"label": "model hypothesis", "case_ids": [answers[0]["query_id"]],
                          "attribution": "unverified", "uncertainty": "high"}],
            "proposals": [{"direction": f"d{i}", "hypothesis": f"h{i}",
                           "prediction": f"p{i}", "risk": f"r{i}"} for i in range(3)],
            "diagnostic_example_count": 5,
            "call": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "latency_s": 3.2},
        })
        dump(run / "interventions.json", [{"query_id": answers[0]["query_id"], "diagnosis": "evidence sensitivity only"}])
        for i in (1, 3):
            branch = run / "branches" / f"candidate-{i}"
            dump(branch / "proposal.json", {"direction": f"d{i-1}", "hypothesis": f"h{i-1}"})
            (branch / "policy.py").write_text(f"# candidate {i}\n", encoding="utf-8")
        dump(run / "branches" / "candidate-2" / "proposal.json", {"direction": "d1"})
        dump(run / "branches" / "candidate-2" / "failure.json", {
            "reason": "finish_reason=length; codegen output was truncated",
            "category": "contract", "stage": "codegen", "policy_cost": None,
        })
        dump(run / "plain-bm25" / "train" / "report.json", {"AVG": 0.4, "cost": {"policy": []}})
        dump(run / "plain-bm25" / "validation" / "report.json", {"AVG": 0.5, "cost": {"policy": []}})
        dump(run / "branches" / "candidate-3" / "train" / "report.json", {"AVG": 0.3, "cost": {"policy": []}})
        branch_train = run / "branches" / "candidate-1" / "train" / "answers.jsonl"
        branch_train.parent.mkdir(parents=True)
        branch_train.write_text(json.dumps(answer(21, 1)) + "\n", encoding="utf-8")
        branch_validation = run / "branches" / "candidate-3" / "validation" / "answers.jsonl"
        branch_validation.parent.mkdir(parents=True)
        branch_validation.write_text(json.dumps(answer(22, 1)) + "\n", encoding="utf-8")
        test_path = run / "plain-bm25" / "test" / "answers.jsonl"
        test_path.parent.mkdir(parents=True)
        test_path.write_text(json.dumps({"question": "must never ingest test"}) + "\n", encoding="utf-8")
        return run, answers

    def test_six_diagnostic_cases_use_failure_then_success_order(self):
        with tempfile.TemporaryDirectory() as temp:
            run, answers = self.make_run(Path(temp))
            traces, index = build_backfill(run, imported_at_ns=123)
            cases = [row for row in traces if row["name"] == "rsi.historical.case"]
            self.assertEqual(index["selected_diagnostic_case_count"], 6)
            self.assertEqual(len(cases), 6)
            self.assertEqual([row["output"]["query_id"] for row in cases],
                             [row["query_id"] for row in answers[:4] + answers[5:7]])
            self.assertIn("trajectory", cases[0]["output"]["recorded_result"])
            self.assertIn("call_log", cases[0]["output"]["recorded_result"])
            self.assertIn("query_id", cases[0]["input"]["recorded_query"])
            self.assertTrue(all(row["start_time_ns"] == row["end_time_ns"] == 123 for row in traces))
            self.assertTrue(all(row["metadata"]["timing"] == "import_time_not_original" for row in traces))
            self.assertEqual(sum(row["metadata"]["diagnostic_input_status"] ==
                                 "prefix_reconstructed_from_recorded_count" for row in cases), 5)
            self.assertEqual(cases[-1]["metadata"]["diagnostic_input_status"],
                             "candidate_selected_input_unconfirmed")

    def test_no_proposal_case_links_and_truncation_is_an_unavailable_source(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.make_run(Path(temp))
            traces, _ = build_backfill(run, imported_at_ns=123)
            candidates = [row for row in traces if row["name"] == "rsi.historical.codegen"]
            self.assertEqual(len(candidates), 3)
            self.assertTrue(all(row["metadata"]["proposal_case_links"] == "not_recorded" for row in candidates))
            failed = candidates[1]
            self.assertEqual(failed["status"], "error")
            self.assertIn("truncated", failed["output"]["failure"]["reason"])
            self.assertIsNone(failed["output"]["source"])
            self.assertEqual(failed["provenance"], "historical_recorded")
            summary = next(row for row in traces
                           if row["name"] == "rsi.historical.validation_and_cost_summary")
            self.assertIn("branches/candidate-3/train", summary["output"]["reports"])
            diagnostic = next(row for row in traces
                              if row["name"] == "rsi.historical.diagnostic_bundle")
            parent = diagnostic["output"]["recorded_parent_policy"]
            self.assertEqual(parent["provenance"], "historical_recorded")
            self.assertTrue(parent["sha256"])
            case = next(row for row in traces if row["name"] == "rsi.historical.case")
            self.assertEqual(case["links"][0]["span_id"],
                             next(row["span_id"] for row in traces
                                  if row["name"] == "rsi.historical.diagnostic_bundle"))

    def test_ids_are_repeatable_and_test_artifacts_are_never_read(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.make_run(Path(temp))
            first, first_index = build_backfill(run, imported_at_ns=123)
            second, second_index = build_backfill(run, imported_at_ns=456)
            self.assertEqual([row["trace_id"] for row in first], [row["trace_id"] for row in second])
            self.assertEqual(first_index["decision_trace_id"], second_index["decision_trace_id"])
            self.assertFalse(first_index["test_data_ingested"])
            self.assertNotIn("must never ingest test", json.dumps(first))
            all_traces, all_index = build_backfill(run, scope="all", imported_at_ns=123)
            self.assertEqual(all_index["case_trace_count"], 11)
            case_rows = [row for row in all_traces if row["name"] == "rsi.historical.case"]
            self.assertTrue(any(row["metadata"]["fold"] == "validation" for row in case_rows))
            self.assertNotIn("must never ingest test", json.dumps(all_traces))
            self.assertTrue(all_index["trace_refs"])

    def test_write_creates_new_spool_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, _ = self.make_run(root)
            output = root / "spool"
            index = write_backfill(run, output, imported_at_ns=123)
            self.assertTrue((output / "traces.jsonl").is_file())
            self.assertTrue((output / "index.json").is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertEqual((output / "traces.jsonl").stat().st_mode & 0o777, 0o600)
            self.assertEqual((output / "index.json").stat().st_mode & 0o777, 0o600)
            saved_index = json.loads((output / "index.json").read_text())
            self.assertEqual(saved_index["decision_trace_id"], index["decision_trace_id"])
            with self.assertRaises(FileExistsError):
                write_backfill(run, output, imported_at_ns=123)

    def test_malformed_existing_artifacts_fail_closed_without_partial_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.make_run(Path(temp))
            answers_path = run / "plain-bm25" / "train" / "answers.jsonl"
            with answers_path.open("a", encoding="utf-8") as stream:
                stream.write('{broken\n')
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                build_backfill(run, imported_at_ns=123)

    def test_root_navigation_metadata_stays_bounded_for_many_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.make_run(Path(temp))
            answers_path = run / "plain-bm25" / "train" / "answers.jsonl"
            answers_path.write_text("".join(
                json.dumps(answer(i, 0 if i < 500 else 1)) + "\n" for i in range(1000)),
                encoding="utf-8")
            traces, index = build_backfill(run, scope="all", imported_at_ns=123)
            root = next(row for row in traces if row["name"] == "rsi.historical.decision")
            self.assertEqual(index["case_trace_count"], 1003)
            self.assertGreater(root["metadata"]["trace_ref_count"], 1000)
            self.assertLessEqual(len(root["metadata"]["trace_refs"]), 16)
            self.assertLess(len(json.dumps(root, ensure_ascii=False).encode("utf-8")), 64 * 1024)
            self.assertIn("comparison", root["output"]["historical_summary"])


if __name__ == "__main__":
    unittest.main()
