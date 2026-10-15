import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from urllib.request import Request
from unittest import mock

from rsi import tracing
from scripts import export_langfuse


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class TracingTests(unittest.TestCase):
    def test_parent_inheritance_and_independent_linked_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with tracing.trace_run(path, "run-7", {"experiment": "e1"}) as root:
                with tracing.trace_span("model", kind="generation", input={"q": "x"}) as span:
                    span.update(output="answer", usage={"input": 4}, model="m")
                with tracing.trace_span("question", new_trace=True) as question:
                    question.update(output="done")
                with tracing.trace_span("after-question"):
                    pass
            spans = rows(path)
            self.assertEqual([s["name"] for s in spans], ["model", "question", "after-question", "rsi-evolution-round"])
            self.assertEqual(spans[0]["parent_span_id"], spans[3]["span_id"])
            self.assertEqual(spans[2]["parent_span_id"], spans[3]["span_id"])
            self.assertNotEqual(spans[1]["trace_id"], spans[3]["trace_id"])
            self.assertEqual(spans[1]["links"], [{"trace_id": spans[3]["trace_id"], "span_id": spans[3]["span_id"]}])
            for span in spans:
                self.assertEqual(span["metadata"]["run_id"], "run-7")
                expected_name = "question" if span["name"] == "question" else "rsi-evolution-round"
                self.assertEqual(span["trace_name"], expected_name)

    def test_scope_metadata_inherits_to_descendants_and_linked_question_then_restores(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with tracing.trace_run(path, "run-7") as root:
                with tracing.trace_span("candidate", metadata={"fold": "train", "candidate": "c1"}) as candidate:
                    candidate.update(metadata={"policy": "p1"})
                    with tracing.trace_span("generation"):
                        pass
                    with tracing.trace_span("question", new_trace=True):
                        pass
                with tracing.trace_span("next-candidate"):
                    pass
            spans = {s["name"]: s for s in rows(path)}
            for name in ("generation", "question"):
                self.assertEqual(spans[name]["metadata"]["fold"], "train")
                self.assertEqual(spans[name]["metadata"]["candidate"], "c1")
                self.assertEqual(spans[name]["metadata"]["policy"], "p1")
            self.assertNotIn("candidate", spans["next-candidate"]["metadata"])
            self.assertEqual(spans["question"]["metadata"]["run_id"], "run-7")
            self.assertEqual(spans["question"]["trace_name"], "question")

    def test_unconfigured_is_noop_and_file_errors_fail_open(self):
        with tracing.trace_span("ignored") as observation:
            observation.update(output="ignored")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with mock.patch.object(tracing.os, "open", side_effect=OSError("private text")):
                with self.assertWarnsRegex(RuntimeWarning, "count=1"):
                    with tracing.trace_run(path, "r"):
                        pass

    def test_warning_filter_errors_cannot_mask_application_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with mock.patch.object(tracing.os, "open", side_effect=OSError("private disk error")):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaisesRegex(ValueError, "application failure"):
                        with tracing.trace_run(path, "r"):
                            raise ValueError("application failure")

    def test_thrown_exception_is_preserved_and_only_type_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with self.assertRaisesRegex(ValueError, "original detail"):
                with tracing.trace_run(path, "r"):
                    with tracing.trace_span("failure"):
                        raise ValueError("original detail")
            span = rows(path)[0]
            self.assertEqual(span["status"], "error")
            self.assertEqual(span["error_type"], "ValueError")
            self.assertNotIn("original detail", Path(path).read_text())

    def test_spool_permissions_and_no_auth_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with tracing.trace_run(path, "r"):
                pass
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            text = path.read_text()
            self.assertNotIn("LANGFUSE", text)
            self.assertNotIn("pk-lf", text)


class LangfuseExporterTests(unittest.TestCase):
    def _record(self):
        return {"schema_version": 1, "trace_id": "a" * 32, "span_id": "b" * 16,
                "parent_span_id": "c" * 16, "name": "generation", "kind": "generation",
                "start_time_ns": 100, "end_time_ns": 200, "input": {"q": "x"},
                "output": {"a": 1}, "metadata": {"run_id": "run-1", "phase": "evo"},
                "model": "model-x", "usage": {"input": 2, "output": 3},
                "status": "ok", "provenance": "live"}

    def test_dry_run_needs_no_credentials_or_network_and_has_no_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")
            result = export_langfuse.export(source, receipt, dry_run=True, env={})
            self.assertEqual(result["span_ids"], ["b" * 16])
            self.assertNotIn('"q":"x"', receipt.read_text())
            self.assertTrue(result["dry_run"])

    def test_missing_credentials_fails_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")
            opener = mock.Mock(side_effect=AssertionError("network must not be touched"))
            with self.assertRaisesRegex(export_langfuse.ExportError, "KEY"):
                export_langfuse.export(source, receipt, env={
                    "LANGFUSE_BASE_URL": "http://localhost:3000"}, opener=opener)
            opener.assert_not_called()

    def test_http_request_shape_and_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")
            captured = {}

            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b"{}"

            def opener(request, timeout):
                captured["request"] = request
                captured["body"] = json.loads(request.data)
                return Response()

            export_langfuse.export(source, receipt, env={
                "LANGFUSE_BASE_URL": "http://127.0.0.1:3000",
                "LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
                opener=opener)
            req = captured["request"]
            self.assertEqual(req.full_url, "http://127.0.0.1:3000/api/public/otel/v1/traces")
            self.assertEqual(req.get_header("Content-type"), "application/json")
            self.assertEqual(req.get_header("X-langfuse-ingestion-version"), "4")
            self.assertNotIn("sk-test", req.data.decode())
            span = captured["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
            self.assertEqual(span["parentSpanId"], "c" * 16)
            self.assertEqual(span["startTimeUnixNano"], "100")
            attrs = {a["key"]: a["value"] for a in span["attributes"]}
            self.assertEqual(attrs["langfuse.session.id"]["stringValue"], "run-1")
            self.assertEqual(attrs["langfuse.observation.model.name"]["stringValue"], "model-x")
            self.assertEqual(json.loads(attrs["langfuse.observation.usage_details"]["stringValue"]), {"input": 2, "output": 3})

    def test_provenance_and_root_trace_name_are_propagated(self):
        child = self._record()
        child.update(provenance="historical_reconstructed")
        root = {"trace_id": child["trace_id"], "span_id": child["parent_span_id"],
                "name": "rsi-evolution-round", "kind": "span",
                "start_time_ns": 1, "end_time_ns": 2, "metadata": {}, "status": "ok"}
        records = [child, root]  # Spool order may place children before roots.
        spans = [span for batch in export_langfuse._batches(
            records, 100000, export_langfuse._trace_names(records)) for span in batch]
        attrs_by_span = [{item["key"]: item["value"] for item in span["attributes"]}
                         for span in spans]
        self.assertEqual([attrs["langfuse.trace.name"]["stringValue"]
                          for attrs in attrs_by_span], ["rsi-evolution-round"] * 2)
        child_attrs = attrs_by_span[0]
        self.assertEqual(child_attrs["langfuse.observation.metadata.provenance"]["stringValue"],
                         "historical_reconstructed")

    def test_backfill_observation_types_chain_and_agent_are_preserved(self):
        for kind in ("chain", "agent"):
            with self.subTest(kind=kind):
                span = export_langfuse.record_to_span({**self._record(), "kind": kind})
                attrs = {item["key"]: item["value"] for item in span["attributes"]}
                self.assertEqual(attrs["langfuse.observation.type"]["stringValue"], kind)

    def test_openai_usage_aliases_flatten_and_emit_standard_token_attributes(self):
        raw_usage = {
            "prompt_tokens": 23,
            "completion_tokens": 8,
            "total_tokens": 31,
            "prompt_tokens_details": {"cached_tokens": 12},
            "completion_tokens_details": {"reasoning_tokens": 3},
            "invalid_negative": -1,
            "invalid_float": 2.5,
            "bool_value": True,
        }
        span = export_langfuse.record_to_span({**self._record(), "usage": raw_usage})
        attrs = {item["key"]: item["value"] for item in span["attributes"]}
        details = json.loads(attrs["langfuse.observation.usage_details"]["stringValue"])
        self.assertEqual(details, {"input": 23, "output": 8, "total": 31})
        self.assertEqual(attrs["gen_ai.usage.input_tokens"]["intValue"], "23")
        self.assertEqual(attrs["gen_ai.usage.output_tokens"]["intValue"], "8")

    def test_partial_rejection_is_not_marked_delivered(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")

            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b'{"partialSuccess":{"rejectedSpans":1}}'

            result = export_langfuse.export(source, receipt, env={
                "LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk"}, opener=lambda *args, **kwargs: Response())
            self.assertEqual(result["accepted_ids"], [])
            self.assertEqual(result["pending_ids"], [])
            self.assertEqual(result["blocked_ids"], ["b" * 16])
            self.assertEqual(result["delivery_status"], "blocked_partial_success")
            self.assertEqual(result["rejected_spans"], 1)
            retried = mock.Mock(side_effect=AssertionError("partial-success batches must not auto-retry"))
            again = export_langfuse.export(source, receipt, env={
                "LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk"}, opener=retried)
            retried.assert_not_called()
            self.assertEqual(again["blocked_ids"], ["b" * 16])
            self.assertEqual(again["pending_ids"], [])

    def test_invalid_success_response_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")

            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b"<html>login required</html>"

            with self.assertRaisesRegex(export_langfuse.ExportError, "invalid OTLP response"):
                export_langfuse.export(source, receipt, env={
                    "LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk",
                    "LANGFUSE_SECRET_KEY": "sk"}, opener=lambda *args, **kwargs: Response())
            self.assertFalse(receipt.exists())

    def test_rejected_span_counts_must_be_valid_nonnegative_integers(self):
        for body in (b'{"partialSuccess":{"rejectedSpans":-1}}',
                     b'{"partialSuccess":{"rejectedSpans":"invalid"}}',
                     b'{"partialSuccess":[]}'):
            with self.subTest(body=body):
                with self.assertRaises(export_langfuse.ExportError):
                    export_langfuse._rejected_spans(body)
        self.assertEqual(export_langfuse._rejected_spans(
            b'{"partialSuccess":{"rejectedSpans":"1"}}'), 1)
        self.assertEqual(export_langfuse._rejected_spans(
            b'{"partialSuccess":{"rejectedSpans":1}}'), 1)

    def test_partial_success_is_checkpointed_before_later_http_error(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            first = self._record()
            second = {**self._record(), "trace_id": "d" * 32, "span_id": "e" * 16}
            source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            span_size = len(json.dumps(export_langfuse._payload([
                export_langfuse.record_to_span(first)]), separators=(",", ":")).encode())
            calls = 0

            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b'{"partialSuccess":{"rejectedSpans":"1"}}'

            def opener(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return Response()
                from urllib.error import HTTPError
                raise HTTPError("http://localhost", 503, "unavailable", {}, None)

            with self.assertRaisesRegex(export_langfuse.ExportError, "HTTP 503"):
                export_langfuse.export(source, receipt, env={
                    "LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk",
                    "LANGFUSE_SECRET_KEY": "sk"}, opener=opener,
                    max_batch_bytes=span_size)
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved["blocked_ids"], [first["span_id"]])
            self.assertEqual(saved["pending_ids"], [second["span_id"]])

    def test_redirect_handler_refuses_all_redirects(self):
        handler = export_langfuse._NoRedirectHandler()
        request = Request("https://langfuse.example/api/public/otel/v1/traces",
                          headers={"Authorization": "Basic secret"})
        self.assertIsNone(handler.redirect_request(
            request, None, 302, "Found", {}, "https://other.example/collect"))

    def test_retry_receipt_is_bound_to_input_and_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            record = self._record()
            source.write_text(json.dumps(record) + "\n")
            env = {"LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk",
                   "LANGFUSE_SECRET_KEY": "sk"}

            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b"{}"

            export_langfuse.export(source, receipt, env=env,
                                   opener=lambda *args, **kwargs: Response())
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved["delivery_semantics"],
                             "accepted_by_langfuse_ingestion_not_ui_visibility")
            with self.assertRaisesRegex(export_langfuse.ExportError, "different Langfuse project"):
                export_langfuse.export(source, receipt,
                    env={**env, "LANGFUSE_PUBLIC_KEY": "another-pk"},
                    opener=lambda *args, **kwargs: Response())
            source.write_text(json.dumps({**record, "output": {"changed": True}}) + "\n")
            with self.assertRaisesRegex(export_langfuse.ExportError, "different input contents"):
                export_langfuse.export(source, receipt, env=env,
                                       opener=lambda *args, **kwargs: Response())

    def test_receipt_write_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / "receipt.json"
            receipt.write_text('{"keep":"old"}\n')
            with mock.patch.object(export_langfuse.os, "fsync", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    export_langfuse._write_receipt(receipt, {"keep": "new"})
            self.assertEqual(json.loads(receipt.read_text()), {"keep": "old"})

    def test_external_destination_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            source, receipt = Path(temp) / "traces.jsonl", Path(temp) / "receipt.json"
            source.write_text(json.dumps(self._record()) + "\n")
            with self.assertRaisesRegex(export_langfuse.ExportError, "--allow-remote"):
                export_langfuse.export(source, receipt, env={
                    "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
                    "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})


if __name__ == "__main__":
    unittest.main()
