import sys
from pathlib import Path
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import unittest
from unittest.mock import patch

from rsi.policy import GeneratedPolicy, PlainPolicy
from rsi.sandbox import SandboxError, run_candidate, validate_source
from rsi import sandbox


MINIMAL_SOURCE = '''
def initialize(config):
    return []

def ingest(state, session):
    state.append(session)
    return state

def search(state, query, filters):
    ids = []
    for record in state:
        sid = record["session_id"]
        checkpoint = filters.get("checkpoint", filters.get("max_session_id"))
        if checkpoint is None or sid <= checkpoint:
            ids.append(record["id"])
    return ids
'''


def worker_available():
    """The real worker requires the complete Linux seccomp/Landlock boundary."""
    probe = getattr(sandbox, "isolation_available", None)
    return bool(probe()) if callable(probe) else False


class PlainPolicyBoundaries(unittest.TestCase):
    def setUp(self):
        self.policy = PlainPolicy({"top_k": 10})
        self.policy.ingest({"id": "old", "persona_id": 3, "session_id": 1, "text": "患者需要控制血糖"})
        self.policy.ingest({"id": "future", "persona_id": 3, "session_id": 9, "text": "患者血糖控制计划"})

    def test_checkpoint_and_unknown_read_id(self):
        found = self.policy.search("血糖控制", {"persona_id": 3, "checkpoint": 1})
        self.assertEqual([row["id"] for row in found], ["old"])
        self.assertEqual(self.policy.read(["old"])[0],
                         {"id": "old", "text": "患者需要控制血糖", "session_id": 1})
        with self.assertRaises(PermissionError):
            self.policy.read(["future"])
        with self.assertRaises(PermissionError):
            self.policy.read(["unknown"])

    def test_question_snapshot_is_deeply_isolated(self):
        snap = self.policy.snapshot()
        snap["chunks"][0]["text"] = "改写副本"
        snap["chunks"].append({"id": "injected", "text": "假数据"})
        snap["cost"]["ingest_calls"] = -1
        result = self.policy.search("血糖控制", {"checkpoint": 1})
        self.assertEqual(result[0]["text"], "患者需要控制血糖")
        self.assertEqual(self.policy.cost["ingest_calls"], 2)

    def test_cost_counters_cover_ingest_search_and_storage(self):
        self.policy.search("血糖")
        self.assertEqual(self.policy.cost["ingest_calls"], 2)
        self.assertEqual(self.policy.cost["search_calls"], 1)
        self.assertGreater(self.policy.cost["storage_bytes"], 0)
        self.assertEqual(self.policy.cost["peak_storage_bytes"], self.policy.cost["storage_bytes"])

    def test_bm25_weight_can_be_zero_for_semantic_only_ranking(self):
        class Vector:
            def __init__(self, values):
                self.values = values
            def __iter__(self):
                return iter(self.values)

        class LocalEncoder:
            def encode(self, texts, batch_size=32, normalize_embeddings=True):
                # Deliberately prefer the second text, which has no query bigrams.
                return [Vector([1, 0]), Vector([0, 1]), Vector([1, 0])]

        policy = PlainPolicy({"top_k": 2, "bm25_weight": 0, "semantic_weight": 0})
        policy._model = LocalEncoder()
        policy.semantic_weight = 1
        policy.ingest({"id": "lexical", "session_id": 1, "text": "血糖控制"})
        policy.ingest({"id": "semantic", "session_id": 2, "text": "other"})
        self.assertEqual(policy.search("血糖控制")[0]["id"], "semantic")

    def test_policy_deepcopy_shares_only_model_and_embedding_cache(self):
        class CountingEncoder:
            def __init__(self):
                self.calls = []
            def encode(self, texts, batch_size=32, normalize_embeddings=True):
                self.calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

        policy = PlainPolicy({"top_k": 5})
        policy.semantic_weight = 1.0
        policy._model = CountingEncoder()
        policy.ingest({"id": "a", "persona_id": 1, "session_id": 1, "text": "first"})
        policy.ingest({"id": "b", "persona_id": 1, "session_id": 2, "text": "second"})
        first = copy.deepcopy(policy)
        self.assertIs(first._model, policy._model)
        self.assertIs(first._embedding_cache, policy._embedding_cache)
        self.assertIsNot(first._chunks, policy._chunks)
        self.assertIsNot(first._by_id["a"], policy._by_id["a"])
        self.assertIsNot(first.cost, policy.cost)
        first.search("question")
        second = copy.deepcopy(policy)
        second.search("another question")
        # The second call encodes only its query; document vectors came from cache.
        self.assertEqual([len(call) for call in policy._model.calls], [3, 1])
        self.assertGreater(first.cost["embedding_cache_bytes"], 0)
        self.assertEqual(policy.cost["embedding_cache_bytes"], 0)

    def test_missing_semantic_model_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "semantic_model_path"):
            PlainPolicy({"semantic_weight": 0.5})
        with self.assertRaises(FileNotFoundError):
            PlainPolicy({"semantic_weight": 0.5, "semantic_model_path": "/no/such/local/model"})

    def test_restore_isolated_question_snapshot(self):
        snapshot = self.policy.snapshot()
        self.policy.ingest({"id": "later", "persona_id": 3, "session_id": 10, "text": "later"})
        self.policy.restore(snapshot)
        self.assertEqual([c["id"] for c in self.policy.snapshot()["chunks"]], ["old", "future"])
        with self.assertRaises(PermissionError):
            self.policy.read(["old"])

    def test_failed_semantic_search_clears_previous_read_visibility(self):
        class FailedEncoder:
            def encode(self, *args, **kwargs):
                raise RuntimeError("model unavailable")

        self.policy.search("血糖", {"checkpoint": 1})
        self.policy._model = FailedEncoder()
        self.policy.semantic_weight = 1.0
        with self.assertRaisesRegex(RuntimeError, "semantic encoder failed"):
            self.policy.search("血糖", {"checkpoint": 1})
        with self.assertRaises(PermissionError):
            self.policy.read(["old"])


class CandidateSyntaxBoundaries(unittest.TestCase):
    def test_validation_checks_syntax_without_restricting_python_dialect(self):
        source = '''
import math
import datetime
class Box:
    def __init__(self, value=0):
        self.value = value
def initialize(config=None):
    transform = lambda x: x + 1
    try:
        values = [transform(x) for x in range(3)]
        return {"values": values, "stamp": datetime.datetime.now(), "box": Box()}
    except Exception as err:
        raise ValueError(str(err))
def ingest(state, session):
    return state
def search(state, query, filters):
    return [math.floor(x) for x in state["values"]]
'''
        validate_source(source)

    def test_validation_rejects_only_nontext_oversized_or_invalid_syntax(self):
        validate_source("import os\n")
        with self.assertRaises(SandboxError):
            validate_source("def broken(:\n    pass\n")
        with self.assertRaises(SandboxError):
            validate_source(None)
        with self.assertRaises(SandboxError):
            validate_source("#" * (sandbox.MAX_SOURCE_BYTES + 1))

    def test_fail_closed_without_complete_isolation(self):
        if worker_available():
            self.skipTest("complete Linux sandbox is available")
        with self.assertRaisesRegex(SandboxError, "fail-closed|isolation"):
            run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}})


@unittest.skipUnless(worker_available(), "real candidate worker requires Linux seccomp and Landlock")
class GeneratedPolicyWorker(unittest.TestCase):
    def test_instance_state_keeps_module_class_identity_through_ingest_search_restore(self):
        source = '''
import datetime
class Memory:
    def __init__(self):
        self.ids = []
        self.seen = set()
        self.stamp = datetime.datetime(2020, 1, 1)
        self.selector = lambda value: value + 1
        self.value = 0
    def remember(self, session):
        self.ids.append(session["id"])
        self.seen.add(session["id"])
        self.stamp += datetime.timedelta(days=1)
        self.value = self.selector(self.value)
    def results(self):
        if self.value == len(self.seen) and self.stamp.year >= 2020:
            return list(self.ids)
        return []
def initialize(config):
    return Memory()
def ingest(state, session):
    if not isinstance(state, Memory):
        raise TypeError("state lost Memory class identity")
    state.remember(session)
    return state
def search(state, query, filters):
    if not isinstance(state, Memory):
        raise TypeError("state lost Memory class identity")
    return state.results()
'''
        policy = GeneratedPolicy(source)
        policy.ingest({"id": "one", "persona_id": 1, "session_id": 1, "text": "one"})
        snapshot = policy.snapshot()
        policy.ingest({"id": "two", "persona_id": 1, "session_id": 2, "text": "two"})
        self.assertEqual([row["id"] for row in policy.search("q")], ["one", "two"])
        policy.restore(snapshot)
        self.assertEqual([row["id"] for row in policy.search("q")], ["one"])

    def test_normal_python_and_arbitrary_object_state_survive_operations_and_restore(self):
        source = '''
import datetime
class Box:
    def __init__(self, value=0):
        self.value = value
def initialize(config):
    return {"ids": [], "seen": set(), "stamp": datetime.datetime(2020, 1, 1),
            "box": Box(), "select": lambda value: value}
def ingest(state, session):
    state["ids"].append(session["id"])
    state["seen"].add(session["id"])
    state["stamp"] += datetime.timedelta(days=1)
    state["box"].value += 1
    return state
def search(state, query, filters):
    if (isinstance(state["stamp"], datetime.datetime) and
            state["select"](state["box"].value) == len(state["seen"])):
        return list(state["ids"])
    return []
'''
        policy = GeneratedPolicy(source)
        policy.ingest({"id": "one", "persona_id": 1, "session_id": 1, "text": "one"})
        snapshot = policy.snapshot()
        policy.ingest({"id": "two", "persona_id": 1, "session_id": 2, "text": "two"})
        self.assertEqual([item["id"] for item in policy.search("q")], ["one", "two"])
        policy.restore(snapshot)
        self.assertEqual([item["id"] for item in policy.search("q")], ["one"])

    def test_candidate_language_can_import_stdlib_and_use_classes_lambda_and_exceptions(self):
        source = '''
import math
def initialize(config):
    return {"values": [3.2, 2.1], "selector": lambda value: math.floor(value), "ids": []}
def ingest(state, session):
    try:
        state["values"].append(float(session["value"]))
        state["ids"].append(session["id"])
    except (TypeError, ValueError):
        raise ValueError("bad numeric value")
    return state
def search(state, query, filters):
    try:
        if isinstance(state["values"], list) and state["selector"](state["values"][0]) > 2:
            return [value for value in state["ids"]]
    except (TypeError, ValueError):
        raise RuntimeError("candidate computation failed")
    return []
'''
        policy = GeneratedPolicy(source)
        policy.ingest({"id": "normal", "persona_id": 1, "session_id": 1,
                       "text": "normal", "value": "4.2"})
        self.assertEqual([row["id"] for row in policy.search("q")], ["normal"])

    def test_search_mutation_isolation_and_host_id_scope(self):
        source = '''
def initialize(config):
    return []
def ingest(state, session):
    state.append(session)
    return state
def search(state, query, filters):
    state.reverse()
    return [row["id"] for row in state]
'''
        policy = GeneratedPolicy(source)
        policy.ingest({"id": "one", "persona_id": 1, "session_id": 1, "text": "one"})
        policy.ingest({"id": "two", "persona_id": 1, "session_id": 2, "text": "two"})
        self.assertEqual([x["id"] for x in policy.search("q")], ["two", "one"])
        self.assertEqual([x["id"] for x in policy.search("q")], ["two", "one"])

        future_source = source.replace('return [row["id"] for row in state]', 'return ["future"]')
        future = GeneratedPolicy(future_source)
        future.ingest({"id": "past", "persona_id": 1, "session_id": 1, "text": "past"})
        future.ingest({"id": "future", "persona_id": 1, "session_id": 9, "text": "future"})
        with self.assertRaisesRegex(SandboxError, "unknown or out-of-scope"):
            future.search("q", {"checkpoint": 1})

        unknown_source = source.replace('return [row["id"] for row in state]', 'return ["unregistered"]')
        unknown = GeneratedPolicy(unknown_source)
        unknown.ingest({"id": "past", "persona_id": 1, "session_id": 1, "text": "past"})
        with self.assertRaisesRegex(SandboxError, "unknown or out-of-scope"):
            unknown.search("q")

    def test_missing_entrypoint_fails_at_policy_initialization(self):
        source = '''
def initialize(config): return {}
def ingest(state, session): return state
'''
        with self.assertRaises(SandboxError):
            GeneratedPolicy(source)

    def test_snapshot_cannot_be_restored_by_different_candidate_source(self):
        policy = GeneratedPolicy(MINIMAL_SOURCE)
        policy.ingest({"id": "one", "persona_id": 1, "session_id": 1, "text": "one"})
        snapshot = policy.snapshot()
        changed = GeneratedPolicy(MINIMAL_SOURCE.replace("return []", "return {\"ids\": []}", 1))
        with self.assertRaises((SandboxError, ValueError)):
            changed.restore(snapshot)

    def test_file_network_and_serialization_side_effects_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "escaped.txt"
            file_source = f'''\ndef initialize(config):\n    try:\n        open({str(marker)!r}, "w").write("escaped")\n    except OSError:\n        return {{}}\n    raise RuntimeError("write unexpectedly succeeded")\ndef ingest(state, session): return state\ndef search(state, query, filters): return []\n'''
            GeneratedPolicy(file_source)
            self.assertFalse(marker.exists())

            socket_source = '''
import socket
def initialize(config):
    try:
        socket.socket()
    except OSError:
        return {}
    raise RuntimeError("socket unexpectedly succeeded")
def ingest(state, session): return state
def search(state, query, filters): return []
'''
            GeneratedPolicy(socket_source)

            reduce_source = f'''\nclass Evil:\n    def __reduce__(self):\n        open({str(marker)!r}, "w").write("escaped")\n        return (int, (1,))\ndef initialize(config): return Evil()\ndef ingest(state, session): return state\ndef search(state, query, filters): return []\n'''
            with self.assertRaises(SandboxError):
                GeneratedPolicy(reduce_source)
            self.assertFalse(marker.exists())

    def test_deserialization_reduce_side_effect_runs_only_inside_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "unpickle-escaped.txt"
            source = f'''\nclass Evil:\n    def __reduce__(self):\n        return (exec, ("open({str(marker)!r}, 'w').write('escaped')",))\ndef initialize(config): return Evil()\ndef ingest(state, session): return state\ndef search(state, query, filters): return []\n'''
            policy = GeneratedPolicy(source)
            # State is held by the host as an opaque capsule; only the worker
            # deserializes it, under filesystem isolation.
            self.assertFalse(marker.exists())
            with self.assertRaises(SandboxError):
                policy.search("q")
            self.assertFalse(marker.exists())

    def test_candidate_cannot_read_private_files_outside_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            private_file = Path(temp_dir) / "private.txt"
            private_file.write_text("private test content")
            source = f'''\ndef initialize(config):\n    try:\n        open({str(private_file)!r}, "r").read()\n    except PermissionError:\n        return {{}}\n    raise RuntimeError("private file read unexpectedly succeeded")\ndef ingest(state, session): return state\ndef search(state, query, filters): return []\n'''
            # Successful construction means the candidate caught the expected
            # denial. If Landlock permits the read, initialization raises.
            GeneratedPolicy(source)

    def test_wall_timeout(self):
        source = '''
def initialize(config): return {}
def ingest(state, session): return state
def search(state, query, filters):
    while True:
        pass
'''
        policy = GeneratedPolicy(source, timeout_s=0.25)
        with self.assertRaises(SandboxError):
            policy.search("q")


@unittest.skipUnless(worker_available(), "seccomp/Landlock probe requires complete Linux isolation")
class IsolationOSBoundary(unittest.TestCase):
    def test_child_bootstrap_sets_both_limits_before_loading_worker_without_preexec(self):
        cpu_limit = sandbox._CHILD_BOOTSTRAP.index("resource.setrlimit(resource.RLIMIT_CPU")
        memory_limit = sandbox._CHILD_BOOTSTRAP.index("resource.setrlimit(resource.RLIMIT_AS")
        worker_load = sandbox._CHILD_BOOTSTRAP.index("runpy.run_module('rsi.sandbox'")
        self.assertLess(cpu_limit, worker_load)
        self.assertLess(memory_limit, worker_load)
        self.assertIn("cpu_seconds + 1", sandbox._CHILD_BOOTSTRAP)

        def check_command(command, **kwargs):
            self.assertEqual(command[1], "-c")
            self.assertEqual(command[2], sandbox._CHILD_BOOTSTRAP)
            self.assertNotIn("preexec_fn", kwargs)
            self.assertEqual(command[3:], [str(2 * 1024 * 1024 * 1024), "30.0"])
            return sandbox.subprocess.CompletedProcess(command, 0, b'{"ok": true}', b"")

        with patch.object(sandbox, "isolation_available", return_value=True), \
             patch.object(sandbox.subprocess, "run", side_effect=check_command):
            self.assertEqual(run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}}), {"ok": True})

    def test_bootstrap_limit_failure_is_returned_as_isolation_error(self):
        def fail_during_child(command, **kwargs):
            self.assertNotIn("preexec_fn", kwargs)
            return sandbox.subprocess.CompletedProcess(command, 1,
                b'{"error":"required worker resource limits could not be applied",'
                b'"category":"isolation","error_type":"RuntimeError"}', b"")

        with patch.object(sandbox, "isolation_available", return_value=True), \
             patch.object(sandbox.subprocess, "run", side_effect=fail_during_child):
            with self.assertRaisesRegex(SandboxError, "resource limits") as raised:
                run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}})
        self.assertEqual(raised.exception.category, "isolation")

    def test_parallel_workers_apply_isolation_independently(self):
        def run_one(_):
            return run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}})

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run_one, range(3)))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.get("serializable") is True for result in results))
        self.assertTrue(all(isinstance(result.get("state"), dict) for result in results))


if __name__ == "__main__":
    unittest.main()
