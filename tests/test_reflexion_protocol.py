import json
from pathlib import Path
import tempfile
import unittest

from rsi.model import ModelOutputTruncated, ModelServiceUnavailable
from rsi.reflexion_protocol import (
    DIAGNOSIS_SCHEMA, PROPOSAL_SCHEMA, ProtocolValidationError,
    ReflexionStageFailure, call_validated_stage, validate_diagnosis,
    validate_proposals,
)


def diagnosis(*, status="hypothesized", mechanism="retrieval omitted a relevant update"):
    return {"clusters": [{
        "label": "missed-update", "observed_failure": "The answer omitted the dated status update.",
        "supporting_case_ids": ["case-1"],
        "cause_hypotheses": ([{"mechanism": mechanism,
                               "supporting_observations": ["The update was absent from retrieved evidence."],
                               "counter_evidence": ["The trace does not show the full candidate set."],
                               "confidence": "medium"}] if status == "hypothesized" else []),
        "attribution_status": status, "discriminating_test": "Compare retrieval with and without the update index.",
    }]}


def proposals():
    fields = {"direction": "revise retrieval", "hypothesis": "a broader index improves recall",
              "prediction": "the missing update is retrieved", "risk": "more irrelevant matches",
              "basis": "training_cases", "rationale": "case-1 shows an omitted update",
              "mechanism": "include update metadata in candidate ranking",
              "falsification_test": "measure whether the same update remains absent",
              "supporting_case_ids": ["case-1"], "supporting_cluster_labels": ["missed-update"],
              "uncertainty": "medium"}
    return {"proposals": [dict(fields, direction=f"{fields['direction']} {i}") for i in range(3)]}


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((json.loads(json.dumps(messages)), kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def response(obj, usage=None):
    return {"text": json.dumps(obj), "usage": usage or {"prompt_tokens": 2, "completion_tokens": 3},
            "latency_s": 0.25, "role": "reflexion"}


class ReflexionProtocolTests(unittest.TestCase):
    def test_three_confidence_only_explanations_rejected_then_corrected(self):
        invalid = diagnosis(mechanism="medium")
        invalid["clusters"] = [dict(invalid["clusters"][0], label=f"failure-{i}") for i in range(3)]
        corrected = diagnosis()
        client = QueueClient([response(invalid), response(corrected)])
        original_messages = [{"role": "user", "content": "synthetic diagnostic input"}]

        obj, record = call_validated_stage(
            client, "diagnosis", original_messages, DIAGNOSIS_SCHEMA,
            lambda value: validate_diagnosis(value, ["case-1"]))

        self.assertEqual(obj, corrected)
        self.assertEqual(record["status"], "validated")
        first, second = record["attempts"]
        self.assertEqual(first["status"], "invalid")
        self.assertEqual(len(first["validation_errors"]), 3)
        self.assertTrue(all("mechanism" in error for error in first["validation_errors"]))
        self.assertEqual(first["response"]["text"], json.dumps(invalid))
        self.assertEqual(first["cost"], {"usage": {"prompt_tokens": 2, "completion_tokens": 3},
                                          "latency_s": 0.25, "role": "reflexion"})
        self.assertEqual(second["input_messages"][0], original_messages[0])
        self.assertEqual(second["input_messages"][1]["content"], json.dumps(invalid))
        correction_text = second["input_messages"][2]["content"]
        self.assertIn("Do not invent observations", correction_text)
        self.assertNotIn(json.dumps(invalid), correction_text)
        self.assertEqual(client.calls[0][1], {"role": "reflexion", "json_mode": True,
                                               "json_schema": DIAGNOSIS_SCHEMA})

    def test_unresolved_with_no_hypotheses_is_valid_and_does_not_add_cause(self):
        obj = diagnosis(status="unresolved")
        validate_diagnosis(obj, ["case-1"])
        self.assertEqual(obj["clusters"][0]["cause_hypotheses"], [])

    def test_case_and_cluster_references_must_be_from_supplied_input(self):
        bad_diagnosis = diagnosis()
        bad_diagnosis["clusters"][0]["supporting_case_ids"] = ["case-absent"]
        with self.assertRaises(ProtocolValidationError) as caught:
            validate_diagnosis(bad_diagnosis, ["case-1"])
        self.assertTrue(any("absent from input" in issue for issue in caught.exception.issues))

        bad_proposals = proposals()
        bad_proposals["proposals"][0]["supporting_cluster_labels"] = ["unknown-cluster"]
        with self.assertRaises(ProtocolValidationError) as caught:
            validate_proposals(bad_proposals, ["case-1"], ["missed-update"])
        self.assertTrue(any("cluster label absent" in issue for issue in caught.exception.issues))

    def test_retry_exhaustion_records_each_attempt_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            invalid = {"clusters": [{"label": "x"}]}
            client = QueueClient([response(invalid), response(invalid)])
            with self.assertRaises(ReflexionStageFailure) as caught:
                call_validated_stage(client, "diagnosis", [{"role": "user", "content": "synthetic"}],
                                     DIAGNOSIS_SCHEMA,
                                     lambda value: validate_diagnosis(value, ["case-1"]),
                                     artifact_dir=temp)
            failure = caught.exception
            self.assertEqual(failure.stage, "diagnosis")
            self.assertEqual(failure.category, "validation_error")
            self.assertEqual(len(failure.attempts["attempts"]), 2)
            out = Path(temp) / "diagnosis"
            self.assertTrue((out / "attempt-001.json").exists())
            self.assertTrue((out / "attempt-002.json").exists())
            self.assertEqual((out / "attempt-001.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((out / "attempt-002.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((out / "result.json").stat().st_mode & 0o777, 0o600)
            stored = json.loads((out / "result.json").read_text())
            self.assertEqual(stored["status"], "failed")
            self.assertEqual(stored["attempts"][0]["response"]["text"], json.dumps(invalid))

    def test_large_input_and_arrays_are_never_trimmed(self):
        huge_text = "x" * 25000
        huge_cases = [f"case-{i}" for i in range(12000)]
        messages = [{"role": "user", "content": huge_text},
                    {"role": "user", "content": json.dumps(huge_cases)}]
        original = json.loads(json.dumps(messages))
        client = QueueClient([response({"clusters": []})])
        obj, record = call_validated_stage(client, "diagnosis", messages, DIAGNOSIS_SCHEMA,
                                           lambda value: validate_diagnosis(value, huge_cases))
        self.assertEqual(obj, {"clusters": []})
        self.assertEqual(client.calls[0][0], original)
        self.assertEqual(record["attempts"][0]["input_messages"], original)
        self.assertEqual(len(record["attempts"][0]["input_messages"][0]["content"]), 25000)
        self.assertEqual(len(json.loads(record["attempts"][0]["input_messages"][1]["content"])), 12000)

    def test_service_unavailable_is_not_retried_and_attempt_is_retained(self):
        client = QueueClient([ModelServiceUnavailable("offline")])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ModelServiceUnavailable) as caught:
                call_validated_stage(client, "proposals", [{"role": "user", "content": "synthetic"}],
                                     PROPOSAL_SCHEMA, lambda value: validate_proposals(value, [], []),
                                     max_corrections=4, artifact_dir=temp)
            self.assertEqual(caught.exception.stage, "proposals")
            self.assertEqual(caught.exception.category, "service_unavailable")
            self.assertEqual(len(client.calls), 1)
            stored = json.loads((Path(temp) / "proposals" / "result.json").read_text())
            self.assertEqual(stored["status"], "service_unavailable")
            self.assertEqual(stored["attempts"][0]["status"], "service_unavailable")

    def test_failed_call_uses_only_its_own_model_call_cost_and_category(self):
        class TruncatedClient:
            def __init__(self):
                self.calls = [{"role": "reflexion", "usage": {"prompt_tokens": 900},
                               "latency_s": 99.0, "error": "an earlier call"}]

            def complete(self, *args, **kwargs):
                self.calls.append({"role": "reflexion", "usage": {"prompt_tokens": 7,
                               "completion_tokens": 11}, "latency_s": 1.75,
                               "error": "truncated generation"})
                raise ModelOutputTruncated("truncated generation")

        client = TruncatedClient()
        with self.assertRaises(ReflexionStageFailure) as caught:
            call_validated_stage(client, "diagnosis", [{"role": "user", "content": "synthetic"}],
                                 DIAGNOSIS_SCHEMA, lambda value: validate_diagnosis(value, []),
                                 max_corrections=0)
        failure = caught.exception
        self.assertEqual(failure.category, "generation_truncated")
        attempt = failure.attempts["attempts"][0]
        self.assertEqual(attempt["cost"], {"usage": {"prompt_tokens": 7, "completion_tokens": 11},
                                           "latency_s": 1.75, "role": "reflexion"})
        self.assertEqual(attempt["usage"], {"prompt_tokens": 7, "completion_tokens": 11})


if __name__ == "__main__":
    unittest.main()
