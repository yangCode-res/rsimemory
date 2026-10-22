import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rsi.resume import (BASELINE_FILES, PROVENANCE_FILES, RUNTIME_FILES, ResumeError,
                        baseline_runtime_fingerprint, materialize_baseline_resume,
                        verify_baseline_resume)


class BaselineResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "rsi").mkdir(parents=True)
        source_rsi = Path(__file__).resolve().parents[1] / "rsi"
        for name in RUNTIME_FILES:
            shutil.copy2(source_rsi / name, self.repo / "rsi" / name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "rsi"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", "commit", "-qm", "frozen runtime"], check=True)
        self.commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.config = {
            "contract_version": "v0.3", "interface_version": "v0.3",
            "state_serializer": "cloudpickle-base64-v1", "language": "zh", "mode": "Efficient",
            "seed": 20261013, "split_version": "patient-v2", "split_fractions": {"train": 0.4, "validation": 0.2},
            "fixture": True, "model": "synthetic", "judge_model": "synthetic",
            "evaluation_metric_version": "strict-plus-e2e-v1", "max_steps": 4,
            "chunk_chars": 1200, "chunk_overlap": 120, "top_k": 5, "context_chars": 16000,
            "candidate_timeout_s": 30, "candidate_memory_mb": 2048,
            "worker_isolation": "Landlock-read-only-runtime + seccomp",
            "data_revision": "synthetic-r1", "implementation_commit": self.commit,
        }
        self.parent_source = {"version": "synthetic-v1",
                              "sha256": hashlib.sha256(b"# synthetic baseline only\n").hexdigest(), "language": "zh"}
        self.split = {"train": ["p1"], "validation": ["p2"], "test": ["p3"]}
        self.run_ids = {"train": ["p1"], "validation": ["p2"]}
        self.run_queries = {
            "train": [{"query_id": "q1", "persona_id": "p1", "session_id": 10, "query_type": "entity_exact_match"}],
            "validation": [{"query_id": "q2", "persona_id": "p2", "session_id": 20, "query_type": "entity_exact_match"}],
        }
        self.source = self.root / "archived" / "result"
        self.source.mkdir(parents=True)
        self._write_archive()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def _report(self, query, score):
        abnormal = int(score is None)
        return {"AVG": None if abnormal else float(score), "counts": {
            "total": 1, "scored": 1 - abnormal, "abnormal": abnormal,
            "by_type": {query["query_type"]: {"total": 1, "scored": 1 - abnormal, "abnormal": abnormal}}}}

    def _write_fold(self, policy, fold, query, score=1, completed=True):
        directory = self.source / policy / fold
        directory.mkdir(parents=True, exist_ok=True)
        if isinstance(query, list):
            query = query[0]
        row = {**query, "answer": "synthetic answer", "score": score,
               "status": "scored" if score is not None else "unscorable"}
        (directory / "answers.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (directory / "operations.jsonl").write_text("{}\n", encoding="utf-8")
        self.dump(directory / "report.json", self._report(query, score))
        self.dump(directory / "progress.json", {"status": "completed" if completed else "failed",
                    "total_questions": 1, "completed": 1 if completed else 0,
                    "scored": int(score is not None), "abnormal": int(score is None)})

    def _write_archive(self, semantic=False):
        self.dump(self.source / "config.json", self.config)
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.dump(self.source / "config-hash.json", {"sha256": config_hash})
        self.dump(self.source / "splits.json", self.split)
        self.dump(self.source / "run_patients.json", self.run_ids)
        (self.source / "parent-policy.py").write_text("# synthetic baseline only\n", encoding="utf-8")
        self.dump(self.source / "parent-source.json", self.parent_source)
        for fold in ("train", "validation"):
            self._write_fold("plain-bm25", fold, self.run_queries[fold])
            if semantic:
                self._write_fold("plain-semantic", fold, self.run_queries[fold])
        self.dump(self.source / "semantic-status.json", {"status": "evaluated" if semantic else "skipped"})
        tracked = list(PROVENANCE_FILES) + list(BASELINE_FILES) + ["semantic-status.json"]
        if semantic:
            tracked += [f"plain-semantic/{fold}/{name}" for fold in ("train", "validation")
                        for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json")]
        manifest_files = {}
        for relative in tracked:
            path = self.source / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_files[f"result/{relative}"] = {"sha256": digest, "size": path.stat().st_size}
        self.dump(self.source.parent / "_archive-manifest.json", {"verified": True, "files": manifest_files})

    def verify(self, **kwargs):
        return verify_baseline_resume(self.source, kwargs.pop("config", self.config),
            kwargs.pop("split", self.split), kwargs.pop("run_ids", self.run_ids),
            kwargs.pop("run_queries", self.run_queries), kwargs.pop("parent_source", self.parent_source),
            code_root=kwargs.pop("code_root", self.repo))

    def test_matching_source_materializes_cost_reports_and_preserves_source(self):
        plan = self.verify()
        source_bytes = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        out = self.root / "new-run"
        receipt = materialize_baseline_resume(plan, out)
        self.assertEqual(plan["train_report"]["counts"], json.loads((out / "plain-bm25/train/report.json").read_text())["counts"])
        self.assertEqual(plan["validation_report"], json.loads((out / "plain-bm25/validation/report.json").read_text()))
        self.assertTrue(receipt["source_manifest_verified"])
        self.assertEqual("reflection", receipt["resume_point"])
        self.assertEqual(source_bytes, {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})
        self.assertFalse((out / "config.json").exists())
        self.assertFalse((out / "policy-git").exists())

    def test_semantic_baseline_is_complete_and_returns_reports(self):
        self.config.update(semantic_model_path="synthetic-model", semantic_weight=1., bm25_weight=0.)
        self._write_archive(semantic=True)
        plan = self.verify()
        self.assertEqual(plan["semantic_train_report"]["AVG"], 1.)
        self.assertEqual(plan["semantic_status"]["status"], "evaluated")

    def test_frozen_configuration_change_is_rejected(self):
        changed = dict(self.config, top_k=6)
        with self.assertRaisesRegex(ResumeError, "field=top_k"):
            self.verify(config=changed)

    def test_diagnostic_protocol_can_change_but_data_revision_cannot(self):
        changed_protocol = dict(self.config, rationale_schema="evidence-v4-structured-diagnosis")
        self.assertIsNotNone(self.verify(config=changed_protocol))
        changed_data = dict(self.config, data_revision="synthetic-r2")
        with self.assertRaisesRegex(ResumeError, "field=data_revision"):
            self.verify(config=changed_data)

    def test_real_run_requires_verified_dataset_and_frozen_runtime_versions(self):
        with self.assertRaisesRegex(ResumeError,'verified dataset revision required'):
            self.verify(config=dict(self.config,fixture=False,data_revision='unverified'))
        with self.assertRaisesRegex(ResumeError,'field=python_version'):
            self.verify(config=dict(self.config,python_version='different-runtime'))

    def test_parent_policy_hash_must_match_parent_source_manifest(self):
        (self.source / "parent-policy.py").write_text("# altered baseline\n")
        self._rewrite_manifest()
        with self.assertRaisesRegex(ResumeError, "parent policy provenance hash mismatch"):
            self.verify()

    def test_dirty_archived_implementation_is_rejected(self):
        config = dict(self.config, implementation_dirty=True)
        self.dump(self.source / "config.json", config)
        self.dump(self.source / "config-hash.json", {"sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
        self._rewrite_manifest()
        with self.assertRaisesRegex(ResumeError, "implementation_dirty"):
            self.verify(config=config)

    def test_semantic_manifest_is_frozen(self):
        self.config.update(semantic_model_path="synthetic-model", semantic_weight=1., bm25_weight=0.,
                           semantic_manifest={"revision": "s1", "sha256": "a" * 64})
        self._write_archive(semantic=True)
        changed = dict(self.config, semantic_manifest={"revision": "s2", "sha256": "b" * 64})
        with self.assertRaisesRegex(ResumeError, "field=semantic_manifest"):
            self.verify(config=changed)

    def test_archive_config_hash_mismatch_is_rejected(self):
        self.dump(self.source / "config-hash.json", {"sha256": "0" * 64})
        self._rewrite_manifest()
        with self.assertRaisesRegex(ResumeError, "config hash mismatch"):
            self.verify()

    def test_source_file_tampering_is_rejected(self):
        (self.source / "plain-bm25/train/answers.jsonl").write_text("tampered\n")
        with self.assertRaisesRegex(ResumeError, "plain-bm25/train/answers.jsonl"):
            self.verify()

    def test_future_query_or_patient_mismatch_is_rejected(self):
        queries = {**self.run_queries, "validation": [{"query_id": "future-q", "persona_id": "future-p",
                    "session_id": 21, "query_type": "entity_exact_match"}]}
        with self.assertRaises(ResumeError):
            self.verify(run_queries=queries)

    def test_partial_replay_is_rejected(self):
        self._write_fold("plain-bm25", "train", self.run_queries["train"], completed=False)
        self._write_archive()
        self._write_fold("plain-bm25", "train", self.run_queries["train"], completed=False)
        # Rebuild only the manifest for the intentionally failed progress record.
        self._rewrite_manifest()
        with self.assertRaisesRegex(ResumeError, "baseline replay incomplete"):
            self.verify()

    def _rewrite_manifest(self):
        manifest = json.loads((self.source.parent / "_archive-manifest.json").read_text())
        for key in list(manifest["files"]):
            relative = key.removeprefix("result/")
            path = self.source / relative
            manifest["files"][key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
        self.dump(self.source.parent / "_archive-manifest.json", manifest)

    def test_null_strict_avg_is_allowed_when_replay_completed(self):
        self._write_fold("plain-bm25", "train", self.run_queries["train"], score=None)
        self._rewrite_manifest()
        plan = self.verify()
        self.assertIsNone(plan["train_report"]["AVG"])

    def test_source_output_overlap_is_rejected(self):
        plan = self.verify()
        with self.assertRaisesRegex(ResumeError, "paths overlap"):
            materialize_baseline_resume(plan, self.source / "child")

    def test_missing_fingerprint_without_trusted_git_history_is_rejected(self):
        config = dict(self.config)
        config.pop("implementation_commit")
        self.dump(self.source / "config.json", config)
        self.dump(self.source / "config-hash.json", {"sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
        self._rewrite_manifest()
        with self.assertRaisesRegex(ResumeError, "baseline runtime fingerprint missing"):
            self.verify(config=config)

    def test_archived_fingerprint_is_compared_to_current_runtime(self):
        config = dict(self.config)
        config.pop("implementation_commit")
        config["baseline_runtime_sha256"] = baseline_runtime_fingerprint(self.repo)
        self.dump(self.source / "config.json", config)
        self.dump(self.source / "config-hash.json", {"sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
        self._rewrite_manifest()
        self.assertEqual("config-fingerprint", self.verify(config=config)["runtime_provenance"]["method"])


if __name__ == "__main__":
    unittest.main()
