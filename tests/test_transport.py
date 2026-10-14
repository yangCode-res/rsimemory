import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from rsi.actor import Actor
from rsi.evaluation import evaluate
from rsi.model import ModelClient, ModelServiceUnavailable
from rsi.policy import PlainPolicy
from rsi.runtime import replay


class TransportFailureTests(unittest.TestCase):
    def test_network_failure_aborts_replay_without_scoring_questions(self):
        client = ModelClient()
        request_count = 0

        def unavailable(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            raise URLError("private-host-and-token")

        queries = [
            {"query_id": f"q{i}", "persona_id": "p1", "session_id": 1,
             "query_type": "entity_exact_match", "question": "Which medication?",
             "positive_answers": ["metformin"]}
            for i in range(2)
        ]
        sessions = [{"persona_id": "p1", "session_id": 1,
                     "text": "The patient takes metformin."}]
        config = {"chunk_chars": 100, "chunk_overlap": 10, "max_steps": 3,
                  "context_chars": 1000, "top_k": 5}

        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", unavailable):
            output = Path(temp) / "replay"
            with self.assertRaises(ModelServiceUnavailable):
                replay(sessions, queries, ["p1"], PlainPolicy, client, client,
                       config, output)

            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(request_count, 1)
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed"], 0)
            self.assertEqual(progress["scored"], 0)
            self.assertEqual(progress["abnormal"], 1)
            self.assertEqual(len(client.calls), 1)
            self.assertNotIn("private-host-and-token", client.calls[0]["error"])

    def test_malformed_model_json_remains_an_ordinary_protocol_failure(self):
        class MalformedThenFinal:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, role="actor", json_mode=False):
                self.calls += 1
                text = "not json" if self.calls == 1 else json.dumps(
                    {"action": "final", "answer": "The evidence is insufficient."})
                return {"text": text, "usage": {}, "latency_s": 0.0, "role": role}

        client = MalformedThenFinal()
        result = Actor(client).answer("Which medication?", object())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(client.calls, 2)
        self.assertTrue(any(row["action"] == "invalid_protocol" for row in result["trajectory"]))

    def test_judge_transport_failure_propagates_but_invalid_json_stays_null(self):
        query = {"query_type": "state_update", "question": "state",
                 "positive_answers": ["stable"]}

        class BrokenJudge:
            def complete(self, *args, **kwargs):
                raise ModelServiceUnavailable("Model service unavailable (ConnectionRefusedError)")

        with self.assertRaises(ModelServiceUnavailable):
            evaluate(query, "stable", BrokenJudge())

        class InvalidJudge:
            def complete(self, *args, **kwargs):
                return {"text": "not json", "usage": {}, "latency_s": 0.0}

        self.assertIsNone(evaluate(query, "stable", InvalidJudge())["score"])


if __name__ == "__main__":
    unittest.main()
