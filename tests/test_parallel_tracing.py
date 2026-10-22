import contextvars
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest

from rsi import tracing


class ParallelTracingTests(unittest.TestCase):
    def test_copied_context_workers_keep_nested_trace_state_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parallel.jsonl"
            barrier = threading.Barrier(3)
            contexts = []
            with tracing.trace_run(path, "parallel-run", {"experiment": "parallel"}) as root:
                root_stack = tracing._ACTIVE.get().stack
                contexts = [contextvars.copy_context() for _ in range(3)]

                def replay_candidate(index):
                    lane = f"candidate-{index}"
                    policy = f"policy-{index}"
                    huge_unicode = (f"病例{index}🌱汉字-é-" * 12000)
                    with tracing.trace_span(
                            "candidate", metadata={"lane": lane, "policy": policy}) as candidate:
                        with tracing.trace_span("replay", metadata={"phase": "replay"}) as replay:
                            with tracing.trace_span(
                                    "question", new_trace=True,
                                    metadata={"question_lane": lane}) as question:
                                with tracing.trace_span(
                                        "llm.answer", kind="generation",
                                        metadata={"role": "answer"}) as llm:
                                    barrier.wait(timeout=10)
                                    llm.update(input={"prompt": huge_unicode},
                                               output={"answer": huge_unicode})
                    return {
                        "candidate": candidate.span_id,
                        "replay": replay.span_id,
                        "question": question.span_id,
                        "llm": llm.span_id,
                        "root_trace_id": root._state.trace_id,
                    }

                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(ctx.run, replay_candidate, i)
                               for i, ctx in enumerate(contexts)]
                    expected = [future.result(timeout=20) for future in futures]

                self.assertEqual(tracing._ACTIVE.get().stack, root_stack)
                self.assertEqual(len(root_stack), 1)

            lines = path.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(len(records), 13)
            by_id = {record["span_id"]: record for record in records}
            root_record = next(r for r in records if r["name"] == "rsi-evolution-round")

            for index, ids in enumerate(expected):
                lane = f"candidate-{index}"
                policy = f"policy-{index}"
                candidate = by_id[ids["candidate"]]
                replay = by_id[ids["replay"]]
                question = by_id[ids["question"]]
                llm = by_id[ids["llm"]]

                self.assertEqual(candidate["parent_span_id"], root_record["span_id"])
                self.assertEqual(replay["parent_span_id"], candidate["span_id"])
                self.assertEqual(question["links"], [{
                    "trace_id": replay["trace_id"], "span_id": replay["span_id"]}])
                self.assertEqual(llm["parent_span_id"], question["span_id"])
                self.assertNotEqual(question["trace_id"], root_record["trace_id"])
                self.assertEqual(candidate["metadata"]["lane"], lane)
                self.assertEqual(replay["metadata"]["policy"], policy)
                self.assertEqual(question["metadata"]["policy"], policy)
                self.assertEqual(llm["metadata"]["policy"], policy)
                self.assertEqual(llm["metadata"]["role"], "answer")
                self.assertEqual(llm["metadata"]["question_lane"], lane)
                self.assertEqual(llm["trace_id"], question["trace_id"])

            self.assertEqual(root_record["trace_id"], expected[0]["root_trace_id"])
            self.assertEqual(len({by_id[item["llm"]]["trace_id"] for item in expected}), 3)
            self.assertEqual(len({by_id[item["llm"]]["span_id"] for item in expected}), 3)


if __name__ == "__main__":
    unittest.main()
