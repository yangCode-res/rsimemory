"""Strict validation and materialization of immutable baseline replay results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


RUNTIME_FILES = ("actor.py", "evaluation.py", "data.py", "policy.py", "model.py",
                 "runtime.py", "sandbox.py", "policy_source.py")
FOLDS = ("train", "validation")
FROZEN_FIELDS = (
    "contract_version", "interface_version", "state_serializer", "language", "mode",
    "seed", "split_seed", "split_version", "split_fractions", "split_patients",
    "fixture", "model", "actor_model", "judge_model", "judge_rubric",
    "evaluation_metric_version", "metric_version", "actor_max_tokens", "max_steps",
    "react_steps", "chunk_chars", "chunk_overlap", "top_k", "context_chars",
    "candidate_timeout_s", "candidate_memory_mb", "candidate_isolation",
    "worker_isolation", "speculation", "speculative_decoding", "data_revision",
    "policy_auxiliary_model", "parent_source_version", "python_version", "cloudpickle_version",
)
SEMANTIC_FIELDS = ("semantic_model_path", "semantic_weight", "bm25_weight",
                   "semantic_manifest", "semantic_model_manifest", "semantic_manifest_sha256")
BASELINE_FILES = tuple(f"plain-bm25/{fold}/{name}" for fold in FOLDS
                       for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json"))
PROVENANCE_FILES = ("config.json", "config-hash.json", "splits.json", "run_patients.json",
                    "parent-policy.py", "parent-source.json")


class ResumeError(ValueError):
    """A safe-to-display validation error that never embeds source contents."""


def _fail(reason: str, field: str | None = None, filename: str | None = None):
    details = [reason]
    if field:
        details.append(f"field={field}")
    if filename:
        details.append(f"file={filename}")
    raise ResumeError("; ".join(details))


def _json(path: Path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("invalid or unreadable JSON", filename=path.name)


def _jsonl(path: Path):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("invalid or unreadable JSONL", filename=path.name)
    return rows


def _sha(path: Path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def baseline_runtime_fingerprint(code_root=None) -> dict[str, str]:
    """Return SHA-256 fingerprints for the frozen replay implementation files."""
    root = Path(code_root) if code_root is not None else Path(__file__).resolve().parent
    if (root / "rsi").is_dir():
        root = root / "rsi"
    names = list(RUNTIME_FILES)
    if (root / "worker.py").is_file():
        names.append("worker.py")
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            _fail("runtime source missing", filename=name)
        result[name] = _sha(path)[0]
    return result


def _manifest(source: Path):
    candidates = (source / "_archive-manifest.json", source.parent / "_archive-manifest.json")
    for path in candidates:
        if path.is_file():
            data = _json(path)
            files = data.get("files") if isinstance(data, dict) else None
            if not isinstance(files, dict):
                _fail("archive manifest has no file map", filename=path.name)
            return path, files
    _fail("archive manifest missing", filename="_archive-manifest.json")


def _manifest_entry(files, source: Path, manifest_path: Path, relative: str):
    keys = [relative]
    try:
        rel_to_manifest = source.relative_to(manifest_path.parent).as_posix()
        keys.extend((f"{rel_to_manifest}/{relative}", f"result/{relative}"))
    except ValueError:
        pass
    for key in keys:
        if key in files:
            entry = files[key]
            if not isinstance(entry, dict):
                break
            return entry
    _fail("required file absent from archive manifest", filename=relative)


def _check_archived_file(source, files, manifest_path, relative):
    path = source / relative
    if not path.is_file():
        _fail("required source file missing", filename=relative)
    expected = _manifest_entry(files, source, manifest_path, relative)
    try:
        digest, size = _sha(path)
    except OSError:
        _fail("source file unreadable", filename=relative)
    if expected.get("sha256") != digest or expected.get("size") != size:
        _fail("archive integrity mismatch", filename=relative)
    return path


def _normalize_config(config):
    # Names used by older v0.3 configs are mapped onto the current CLI schema.
    aliases = {"split_seed": "seed", "react_steps": "max_steps", "actor_model": "model",
              "candidate_isolation": "worker_isolation", "speculative_decoding": "speculation"}
    normalized = dict(config)
    for old, new in aliases.items():
        if new not in normalized and old in normalized:
            normalized[new] = normalized[old]
    return normalized


def _check_config(source_config, config):
    old, new = _normalize_config(source_config), _normalize_config(config)
    if not new.get('fixture') and (not new.get('data_revision') or new['data_revision']=='unverified'):
        _fail('verified dataset revision required',field='data_revision')
    for field in FROZEN_FIELDS:
        if field in ("split_seed", "react_steps", "actor_model", "candidate_isolation", "speculative_decoding"):
            continue
        if field in old or field in new:
            if old.get(field) != new.get(field):
                _fail("frozen configuration mismatch", field=field)
    old_sem = bool(old.get("semantic_model_path") or old.get("semantic_weight"))
    new_sem = bool(new.get("semantic_model_path") or new.get("semantic_weight"))
    if old_sem != new_sem:
        _fail("semantic baseline request mismatch", field="semantic_model_path")
    if new_sem:
        for field in SEMANTIC_FIELDS:
            if field in old or field in new:
                if old.get(field) != new.get(field):
                    _fail("semantic configuration mismatch", field=field)


def _runtime_from_git(code_root, commit, names=None):
    root = Path(code_root).resolve()
    try:
        subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        _fail("source commit unavailable; baseline fingerprint required", field="implementation_commit")
    values = {}
    for name in names or RUNTIME_FILES:
        try:
            raw = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:rsi/{name}"],
                                          stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            _fail("runtime file unavailable at source commit", filename=name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def _check_runtime(source_config, code_root):
    if source_config.get("implementation_dirty") is True:
        _fail("source commit cannot vouch for dirty implementation", field="implementation_dirty")
    current = baseline_runtime_fingerprint(code_root)
    archived = source_config.get("baseline_runtime_sha256")
    if isinstance(archived, dict):
        for name in current:
            if archived.get(name) != current[name]:
                _fail("frozen runtime source mismatch", filename=name)
        return {"method": "config-fingerprint", "sha256": current}
    commit = source_config.get("implementation_commit")
    if not commit:
        _fail("baseline runtime fingerprint missing", field="baseline_runtime_sha256")
    historical = _runtime_from_git(code_root, str(commit), current)
    for name in current:
        if historical[name] != current[name]:
            _fail("frozen runtime source mismatch", filename=name)
    return {"method": "git-show", "commit": str(commit), "sha256": current}


def _fold_report(source, files, manifest_path, fold, queries, patient_ids, policy="plain-bm25"):
    prefix = f"{policy}/{fold}/"
    paths = {name: _check_archived_file(source, files, manifest_path, prefix + name)
             for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json")}
    progress, report = _json(paths["progress.json"]), _json(paths["report.json"])
    if progress.get("status") != "completed" or progress.get("completed") != progress.get("total_questions"):
        _fail("baseline replay incomplete", filename=prefix + "progress.json")
    if not isinstance(progress.get("total_questions"), int) or progress["total_questions"] != len(queries):
        _fail("baseline query count mismatch", filename=prefix + "progress.json")
    rows = _jsonl(paths["answers.jsonl"])
    expected = {}
    for query in queries:
        qid = query.get("query_id")
        if qid is None or qid in expected:
            _fail("run query ids missing or duplicated", field="query_id")
        if query.get("persona_id") not in patient_ids:
            _fail("query patient outside requested fold", field="persona_id")
        expected[qid] = query
    actual = {}
    for row in rows:
        qid = row.get("query_id")
        if qid in actual or qid not in expected:
            _fail("baseline answer query set mismatch", filename=prefix + "answers.jsonl")
        query = expected[qid]
        if (row.get("persona_id") != query.get("persona_id") or
                row.get("session_id") != query.get("session_id") or
                row.get("query_type") != query.get("query_type") or
                row.get("question") != query.get("question")):
            _fail("baseline answer patient/checkpoint mismatch", filename=prefix + "answers.jsonl")
        actual[qid] = row
    if set(actual) != set(expected):
        _fail("baseline answer query set mismatch", filename=prefix + "answers.jsonl")
    counts = report.get("counts", {})
    abnormal = sum(not (type(row.get("score")) is int and row["score"] in (0, 1)
                        and row.get("status") == "scored") for row in rows)
    if (counts.get("total") != len(rows) or counts.get("abnormal") != abnormal or
            progress.get("abnormal") != abnormal or progress.get("scored") != len(rows) - abnormal or
            counts.get("scored") != len(rows) - abnormal):
        _fail("report/progress count mismatch", filename=prefix + "report.json")
    return report, rows


def _path_overlap(source: Path, out: Path):
    s, o = source.resolve(), out.resolve()
    return s == o or s in o.parents or o in s.parents


def verify_baseline_resume(source_result, config, split, run_ids, run_queries,
                           parent_source, code_root=None):
    """Validate a completed immutable baseline and return a materialization plan."""
    source = Path(source_result).resolve()
    if not source.is_dir():
        _fail("source result directory missing", filename="source_result")
    manifest_path, files = _manifest(source)
    manifest_doc = _json(manifest_path)
    if manifest_doc.get("verified") is not True:
        _fail("archive manifest is not verified", filename="_archive-manifest.json")
    required = list(PROVENANCE_FILES) + list(BASELINE_FILES) + ["semantic-status.json"]
    archived_paths = {name: _check_archived_file(source, files, manifest_path, name) for name in PROVENANCE_FILES}
    source_config = _json(archived_paths["config.json"])
    _check_archived_file(source, files, manifest_path, "semantic-status.json")
    _check_config(source_config, config)
    if _json(archived_paths["splits.json"]) != split:
        _fail("split assignment mismatch", filename="splits.json")
    if _json(archived_paths["run_patients.json"]) != run_ids:
        _fail("run patient assignment mismatch", filename="run_patients.json")
    archived_parent = _json(archived_paths["parent-source.json"])
    if archived_parent != parent_source:
        _fail("parent source provenance mismatch", filename="parent-source.json")
    try:
        parent_policy_sha = _sha(archived_paths["parent-policy.py"])[0]
    except OSError:
        _fail("parent policy unreadable", filename="parent-policy.py")
    if archived_parent.get("sha256") != parent_policy_sha:
        _fail("parent policy provenance hash mismatch", filename="parent-policy.py")
    source_hash = _json(archived_paths["config-hash.json"])
    expected_hash = hashlib.sha256(json.dumps(source_config, sort_keys=True).encode()).hexdigest()
    if source_hash.get("sha256") != expected_hash:
        _fail("config hash mismatch", filename="config-hash.json")
    runtime = _check_runtime(source_config, code_root or Path(__file__).resolve().parent.parent)
    reports, rows = {}, {}
    for fold in FOLDS:
        if fold not in run_queries or fold not in run_ids:
            _fail("run fold missing", field=fold)
        reports[fold], rows[fold] = _fold_report(source, files, manifest_path, fold,
                                                  run_queries[fold], set(run_ids[fold]))
    semantic_status_path = source / "semantic-status.json"
    semantic_status = _json(semantic_status_path)
    semantic_requested = bool(config.get("semantic_model_path") or config.get("semantic_weight"))
    semantic_reports = {}
    if semantic_requested:
        if semantic_status.get("status") not in ("evaluated", "incomplete_scoring"):
            _fail("semantic baseline incomplete", filename="semantic-status.json")
        for fold in FOLDS:
            prefix = f"plain-semantic/{fold}/"
            for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json"):
                _check_archived_file(source, files, manifest_path, prefix + name)
            semantic_reports[fold] = _fold_report(source, files, manifest_path, fold,
                                                  run_queries[fold], set(run_ids[fold]),
                                                  policy="plain-semantic")[0]
    file_hashes = {}
    for relative in required:
        path = source / relative
        file_hashes[relative] = {"sha256": _sha(path)[0], "size": path.stat().st_size}
    for relative in (f"plain-semantic/{fold}/{name}" for fold in FOLDS
                     for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json")):
        if semantic_requested:
            path = source / relative
            file_hashes[relative] = {"sha256": _sha(path)[0], "size": path.stat().st_size}
    return {
        "source_result": str(source), "manifest": str(manifest_path),
        "source_config": source_config, "runtime_provenance": runtime,
        "train_report": reports["train"], "train_rows": rows["train"],
        "validation_report": reports["validation"],
        "semantic_status": semantic_status,
        "semantic_train_report": semantic_reports.get("train"),
        "semantic_validation_report": semantic_reports.get("validation"),
        "copy_files": [*BASELINE_FILES, "semantic-status.json"] +
                       ([f"plain-semantic/{fold}/{name}" for fold in FOLDS
                         for name in ("answers.jsonl", "operations.jsonl", "report.json", "progress.json")]
                        if semantic_requested else []),
        "file_hashes": file_hashes,
        "config_sha256": expected_hash,
        "parent_source": archived_parent,
        "source_manifest_verified": True,
    }


def materialize_baseline_resume(plan, out):
    """Copy only verified baseline artifacts into a disjoint, non-overlapping new output."""
    source, destination = Path(plan["source_result"]).resolve(), Path(out).resolve()
    if _path_overlap(source, destination):
        _fail("source and output paths overlap", field="out")
    if destination.exists() and not destination.is_dir():
        _fail("output path is not a directory", field="out")
    files = list(plan["copy_files"])
    receipt_path = destination / "baseline-reuse.json"
    if receipt_path.exists():
        _fail("refusing to overwrite output file", filename=receipt_path.name)
    # Validate the entire copy set before creating directories or writing any file.
    for relative in files:
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            _fail("invalid planned relative path", filename=rel_path.name)
        src, dst = source / rel_path, destination / rel_path
        if not src.is_file():
            _fail("planned source file missing", filename=relative)
        if dst.exists():
            _fail("refusing to overwrite output file", filename=relative)
        expected = plan.get("file_hashes", {}).get(relative)
        if not expected:
            _fail("planned file fingerprint missing", filename=relative)
        digest, size = _sha(src)
        if digest != expected.get("sha256") or size != expected.get("size"):
            _fail("source changed after validation", filename=relative)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in files:
        src, dst = source / relative, destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    receipt = {"source_result": str(source), "source_manifest_verified": True,
               "config_sha256": plan["config_sha256"],
               "runtime_provenance": plan["runtime_provenance"],
               "copied_files": files,
               "resume_point": "reflection", "candidate_state": "replay-from-empty"}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt
