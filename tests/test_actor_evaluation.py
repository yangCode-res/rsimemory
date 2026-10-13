import io
import json
import unittest
from unittest.mock import patch

from rsi.actor import Actor
from rsi.evaluation import TYPES, aggregate, evaluate
from rsi.model import ModelClient


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.roles = []
        self.message_calls = []

    def complete(self, messages, role="actor", json_mode=False):
        self.roles.append((role, json_mode))
        self.message_calls.append(list(messages))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False)
        else:
            text = item
        return {"text": text, "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                "latency_s": 0.01, "role": role}


class ReadOnlyTools:
    def __init__(self):
        self.calls = []
        self.snapshot_calls = 0

    def hostsnapshot(self):
        self.snapshot_calls += 1
        return self

    def search(self, query, filters):
        self.calls.append(("search", query, filters))
        return [{"id": "m1", "text": "Patient takes metformin.", "session_id": 3,
                 "private_annotation": "must not leak"}]

    def read(self, ids):
        self.calls.append(("read", ids))
        return [{"id": "m1", "text": "Patient takes metformin.", "session_id": 3}]


class ActorEvaluationTests(unittest.TestCase):
    def test_actor_search_read_final_is_bounded_and_read_only(self):
        client = FakeClient([
            {"action": "search", "query": "current medication", "filters": {}},
            {"action": "read", "ids": ["m1"]},
            {"action": "final", "answer": "The record says metformin."},
        ])
        tools = ReadOnlyTools()
        result = Actor(client, max_steps=4).answer("Which medication?", tools)
        self.assertEqual(result["answer"], "The record says metformin.")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call[0] for call in tools.calls], ["search", "read"])
        self.assertEqual(tools.snapshot_calls, 1)
        self.assertEqual(result["cost"]["calls"], 3)
        self.assertNotIn("private_annotation", json.dumps(result["evidence"]))

    def test_crud_action_is_protocol_error_and_never_dispatched(self):
        client = FakeClient([
            {"action": "delete", "ids": ["m1"]},
            {"action": "final", "answer": "No change was made."},
        ])
        tools = ReadOnlyTools()
        result = Actor(client).answer("Question", tools)
        self.assertEqual(result["answer"], "No change was made.")
        self.assertEqual(tools.calls, [])
        self.assertTrue(any(item["action"] == "invalid_protocol" for item in result["trajectory"]))

    def test_step_exhaustion_uses_marked_forced_final(self):
        client = FakeClient([
            {"action": "search", "query": "medication", "filters": {}},
            {"action": "final", "answer": "The evidence is insufficient."},
        ])
        result = Actor(client, max_steps=1).answer("Question", ReadOnlyTools())
        self.assertEqual(result["status"], "forced_final")
        self.assertEqual(result["trajectory"][-1]["action"], "forced_final")
        final_messages = client.message_calls[-1]
        system_positions = [i for i, message in enumerate(final_messages)
                            if message["role"] == "system"]
        self.assertEqual(system_positions, [0])

    def test_invalid_judge_output_is_null_and_cost_is_logged(self):
        client = FakeClient(['{"score":true,"reason":"invalid boolean"}'])
        result = evaluate({"query_type": "state_update", "question": "state",
                           "positive_answers": ["stable"]}, "stable", client)
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "judge_failed")
        self.assertEqual(result["call_log"][0]["role"], "judge")
        self.assertEqual(result["cost"]["calls"], 1)

    def test_float_judge_score_is_invalid(self):
        client = FakeClient(['{"score":1.0,"reason":"not an integer"}'])
        result = evaluate({"query_type": "state_update", "question": "state",
                           "positive_answers": ["stable"]}, "stable", client)
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "judge_failed")

    def test_aggregate_macro_requires_all_six_complete_types(self):
        rows = [{"query_type": kind, "status": "scored", "score": 1} for kind in TYPES]
        result = aggregate(rows)
        self.assertEqual(result["AVG"], 1.0)
        rows.append({"query_type": "state_update", "status": "judge_failed", "score": None})
        result = aggregate(rows)
        self.assertIsNone(result["AVG"])
        self.assertEqual(result["counts"]["abnormal"], 1)
        self.assertIsNone(result["type_accuracy"]["state_update"])
        self.assertEqual(result["counts"]["by_status"]["judge_failed"], 1)

    def test_entity_match_uses_normalized_full_entity(self):
        query = {"query_type": "entity_exact_match", "positive_answers": ["Aspirin"]}
        self.assertEqual(evaluate(query, "The patient uses ASPIRIN.", None)["score"], 1)
        self.assertEqual(evaluate(query, "The patient uses aspirational therapy.", None)["score"], 0)

    def test_entity_match_rejects_explicit_negation(self):
        query = {"query_type": "entity_exact_match", "positive_answers": ["diabetes"]}
        self.assertEqual(evaluate(query, "No diabetes.", None)["score"], 0)
        self.assertEqual(evaluate(query, "The patient has diabetes.", None)["score"], 1)
        chinese = {"query_type": "entity_exact_match", "positive_answers": ["糖尿病"]}
        self.assertEqual(evaluate(chinese, "没有糖尿病。", None)["score"], 0)
        self.assertEqual(evaluate(chinese, "患者患有糖尿病。", None)["score"], 1)

    def test_multiple_choice_requires_exact_option_set(self):
        query = {"query_type": "multiple_choice", "options": ["one", "two", "three"],
                 "correct_options": ["A", "C"]}
        self.assertEqual(evaluate(query, "A, C", None)["score"], 1)
        self.assertEqual(evaluate(query, "A and C, though B may also apply", None)["score"], 0)

    def test_unknown_type_prevents_macro_avg(self):
        rows = [{"query_type": kind, "status": "scored", "score": 1} for kind in TYPES]
        rows.append({"query_type": "unexpected", "status": "scored", "score": 1})
        result = aggregate(rows)
        self.assertIsNone(result["AVG"])
        self.assertEqual(result["counts"]["abnormal"], 1)
        self.assertEqual(result["counts"]["unknown_type"], {"unexpected": 1})

    def test_model_client_records_truncation_and_sets_non_thinking_param(self):
        payload = {"choices": [{"finish_reason": "length",
                                "message": {"content": '{"action":"final"}'}}],
                   "usage": {"prompt_tokens": 8}}
        client = ModelClient(model="test-model")
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch("rsi.model.urllib.request.urlopen", return_value=response) as urlopen:
            with self.assertRaisesRegex(ValueError, "truncated"):
                client.complete([{"role": "user", "content": "q"}], role="actor", json_mode=True)
        body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(client.calls[0]["role"], "actor")
        self.assertEqual(client.calls[0]["usage"], {"prompt_tokens": 8})
        self.assertIn("error", client.calls[0])

    def test_model_client_records_null_content_and_transport_error(self):
        client = ModelClient(model="test-model")
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}
        with patch("rsi.model.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(payload).encode("utf-8"))):
            with self.assertRaisesRegex(ValueError, "null content"):
                client.complete([], role="judge")
        with patch("rsi.model.urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                client.complete([], role="actor")
        self.assertEqual([call["role"] for call in client.calls], ["judge", "actor"])
        self.assertTrue(all("error" in call for call in client.calls))


if __name__ == "__main__":
    unittest.main()
