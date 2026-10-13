import sys
import subprocess
from pathlib import Path
import copy
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
    """The real worker requires Linux plus libseccomp; AST checks work everywhere."""
    import ctypes.util

    return sys.platform.startswith("linux") and ctypes.util.find_library("seccomp") is not None


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


class CandidateASTBoundaries(unittest.TestCase):
    def test_minimal_candidate_has_the_documented_contract(self):
        tree = validate_source(MINIMAL_SOURCE)
        self.assertEqual(len(tree.body), 3)

    def test_import_open_and_dynamic_escape_calls_are_rejected(self):
        attacks = [
            "import os\n" + MINIMAL_SOURCE,
            MINIMAL_SOURCE.replace("state.append(session)", "open('/etc/passwd').read()\n    state.append(session)"),
            MINIMAL_SOURCE.replace("state.append(session)", "eval('1 + 1')\n    state.append(session)"),
            MINIMAL_SOURCE.replace("state.append(session)", "getattr(state, 'append')(session)\n    return state"),
            MINIMAL_SOURCE.replace("state.append(session)", "state.__class__\n    state.append(session)"),
            MINIMAL_SOURCE.replace("state.append(session)", "exec('x=1')\n    state.append(session)"),
            MINIMAL_SOURCE.replace("def initialize(config):", "def sorted(config):\n    return []\n\ndef initialize(config):"),
            MINIMAL_SOURCE.replace("def initialize(config):", "def initialize():"),
        ]
        for source in attacks:
            with self.subTest(source=source[:60]), self.assertRaises(SandboxError):
                validate_source(source)

    def test_candidate_cannot_retrieve_method_objects(self):
        source = MINIMAL_SOURCE.replace("state.append(session)", "method = state.append\n    state.append(session)")
        with self.assertRaises(SandboxError):
            validate_source(source)

    def test_fail_closed_without_linux_seccomp_worker(self):
        if worker_available():
            self.skipTest("libseccomp worker available")
        with self.assertRaisesRegex(SandboxError, "fail-closed"):
            run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}})


@unittest.skipUnless(worker_available(), "real candidate worker requires Linux libseccomp")
class GeneratedPolicyWorker(unittest.TestCase):
    def test_candidate_changes_search_and_host_checks_returned_ids(self):
        reverse_source = MINIMAL_SOURCE.replace(
            "for record in state:\n        sid = record[\"session_id\"]",
            "for record in state:\n        sid = record[\"session_id\"]",
        ).replace("return ids\n", "ids.reverse()\n    return ids\n")
        first = GeneratedPolicy(MINIMAL_SOURCE)
        second = GeneratedPolicy(reverse_source)
        for policy in (first, second):
            policy.ingest({"id": "a", "persona_id": 1, "session_id": 1, "text": "source A"})
            policy.ingest({"id": "b", "persona_id": 1, "session_id": 2, "text": "source B"})
        a = first.search("q", {"persona_id": 1, "checkpoint": 2})
        b = second.search("q", {"persona_id": 1, "checkpoint": 2})
        self.assertEqual([x["id"] for x in a], ["a", "b"])
        self.assertEqual([x["id"] for x in b], ["b", "a"])
        self.assertEqual(first.read(["a"])[0]["text"], "source A")
        with self.assertRaises(PermissionError):
            first.read(["future"])

        attacker = MINIMAL_SOURCE.replace("return ids", "ids.append('unknown')\n    return ids")
        restricted = GeneratedPolicy(attacker)
        restricted.ingest({"id": "known", "session_id": 1, "text": "original"})
        with self.assertRaisesRegex(SandboxError, "unknown or out-of-scope"):
            restricted.search("q")

    def test_unknown_and_future_ids_rejected_and_top_k_host_limited(self):
        future = MINIMAL_SOURCE.replace("return ids", "return ['future']")
        policy = GeneratedPolicy(future, {"top_k": 1})
        policy.ingest({"id": "old", "session_id": 1, "text": "old"})
        policy.ingest({"id": "future", "session_id": 8, "text": "future"})
        with self.assertRaisesRegex(SandboxError, "unknown or out-of-scope"):
            policy.search("q", {"checkpoint": 1})

        policy = GeneratedPolicy(MINIMAL_SOURCE.replace(
            "return ids", "ids.append('unknown')\n    return ids"), {"top_k": 1})
        policy.ingest({"id": "known", "session_id": 1, "text": "known"})
        with self.assertRaisesRegex(SandboxError, "unknown or out-of-scope"):
            policy.search("q")

        policy = GeneratedPolicy(MINIMAL_SOURCE, {"top_k": 1})
        policy.ingest({"id": "a", "session_id": 1, "text": "a"})
        policy.ingest({"id": "b", "session_id": 2, "text": "b"})
        self.assertEqual([x["id"] for x in policy.search("q")], ["a"])

    def test_search_mutation_does_not_leak_between_questions(self):
        mutating_search = MINIMAL_SOURCE.replace(
            "ids = []\n    for record in state:",
            "state.reverse()\n    ids = []\n    for record in state:",
        )
        policy = GeneratedPolicy(mutating_search)
        policy.ingest({"id": "a", "session_id": 1, "text": "a"})
        policy.ingest({"id": "b", "session_id": 2, "text": "b"})
        self.assertEqual([x["id"] for x in policy.search("q")], ["b", "a"])
        self.assertEqual([x["id"] for x in policy.search("q")], ["b", "a"])

    def test_wall_timeout(self):
        slow = MINIMAL_SOURCE.replace(
            "for record in state:\n        sid = record[\"session_id\"]",
            "for i in range(1000000000):\n        checkpoint = i\n    for record in state:\n        sid = record[\"session_id\"]",
        )
        policy = GeneratedPolicy(slow, timeout_s=0.25)
        with self.assertRaisesRegex(SandboxError, "wall-clock|CPU|process"):
            policy.search("q")

    def test_snapshot_isolation(self):
        policy = GeneratedPolicy(MINIMAL_SOURCE)
        policy.ingest({"id": "one", "session_id": 1, "text": "original"})
        snap = policy.snapshot()
        snap["state"][0]["text"] = "changed copy"
        self.assertEqual(policy.search("q")[0]["text"], "original")
        policy.restore(snap)
        self.assertEqual(policy.search("q")[0]["text"], "original")


@unittest.skipUnless(worker_available(), "seccomp probe requires Linux libseccomp")
class SeccompOSBoundary(unittest.TestCase):
    def test_trusted_worker_probe_blocks_open_and_socket_syscalls(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "rsi.sandbox", "--trusted-seccomp-test-probe"],
            cwd=str(root), env={}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))

    def test_resource_limit_setup_failure_fails_closed(self):
        def fail_during_preexec(*args, **kwargs):
            kwargs["preexec_fn"]()

        with patch.object(sandbox.sys, "platform", "linux"), \
             patch.object(sandbox.ctypes.util, "find_library", return_value="libseccomp.so.2"), \
             patch.object(sandbox, "_limits", side_effect=RuntimeError("no limit")), \
             patch.object(sandbox.subprocess, "run", side_effect=fail_during_preexec):
            with self.assertRaisesRegex(SandboxError, "resource-limited worker"):
                run_candidate(MINIMAL_SOURCE, "initialize", {"config": {}})


if __name__ == "__main__":
    unittest.main()
