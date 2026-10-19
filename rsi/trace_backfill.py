"""Conservative reconstruction of historical RSI run traces.

This module imports artifacts that were actually persisted by the run. It does
not call a model, infer proposal-to-case support, or treat a model hypothesis
as a verified cause.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

SCHEMA_VERSION = 1
PROVENANCE = {"historical_recorded", "historical_reconstructed", "unavailable"}


def _present(path: Path) -> bool:
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect historical artifact {path}: {exc}") from exc


def _json(path: Path) -> Any:
    if not _present(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read historical JSON artifact {path}: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not _present(path):
        return []
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError(f"non-object JSONL row in {path} at line {line_number}")
                    rows.append(value)
    except OSError as exc:
        raise ValueError(f"cannot read historical JSONL artifact {path}: {exc}") from exc
    return rows


def _stable_id(value: str, width: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:width]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _summary_context(summary: Any) -> dict[str, Any]:
    """Retain compact aggregate UI context without copying arbitrary summary payloads."""
    if not isinstance(summary, dict):
        return {"provenance": "unavailable"}
    comparison = summary.get("comparison")
    compact_comparison = {}
    if isinstance(comparison, dict):
        for name, metrics in list(comparison.items())[:32]:
            if isinstance(metrics, dict):
                compact_comparison[str(name)[:100]] = {
                    key: metrics[key] for key in
                    ("AVG", "E2E_AVG", "abnormal_rate", "unanswered_as_zero",
                     "policy_tokens", "policy_seconds", "policy_bytes", "valid")
                    if key in metrics and isinstance(metrics[key], (int, float, bool, type(None)))
                }
    cost = summary.get("cost")
    compact_cost = {}
    if isinstance(cost, dict):
        for group, roles in list(cost.items())[:8]:
            if isinstance(roles, dict):
                compact_cost[str(group)[:100]] = {}
                for role, values in list(roles.items())[:16]:
                    if isinstance(values, dict):
                        compact_cost[str(group)[:100]][str(role)[:100]] = {
                            key: value for key, value in list(values.items())[:8]
                            if isinstance(value, (int, float, bool, type(None)))
                        }
                    elif isinstance(values, (int, float, bool, type(None))):
                        compact_cost[str(group)[:100]][str(role)[:100]] = values
            elif isinstance(roles, (int, float, bool, str, type(None))):
                compact_cost[str(group)[:100]] = roles
    result = {"provenance": "historical_recorded"}
    if compact_comparison:
        result["comparison"] = compact_comparison
    if compact_cost:
        result["cost"] = compact_cost
    pareto = summary.get("pareto")
    if isinstance(pareto, list):
        result["pareto"] = [str(item)[:100] for item in pareto[:32]
                            if isinstance(item, (str, int, float))]
    return result


def _run_identity(run_dir: Path, config: Any, config_hash: Any) -> str:
    recorded = config_hash.get("sha256") if isinstance(config_hash, dict) else None
    if isinstance(recorded, str) and recorded:
        return recorded
    # The basename is stable across copies of this run while avoiding embedding
    # a machine-specific source path in trace identity or metadata.
    return _stable_id(_canonical({"run_name": run_dir.name, "config": config}), 64)


def select_diagnostic_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror evolution.reflect's stable failure-first / success-second choice."""
    failures = [row for row in rows if row.get("score") == 0][:4]
    successes = [row for row in rows if row.get("score") == 1][:2]
    return failures + successes


def _source_record(path: Path) -> dict[str, Any]:
    if not _present(path):
        return {"source": None, "sha256": None,
                "provenance": "unavailable", "reason": "recorded parent source file is absent"}
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read recorded parent source {path}: {exc}") from exc
    return {"source": source_bytes.decode("utf-8"),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "provenance": "historical_recorded"}


def _answers_files(run_dir: Path, scope: str) -> list[tuple[str, str, Path]]:
    if scope == "diagnostic":
        path = run_dir / "plain-bm25" / "train" / "answers.jsonl"
        return [("plain-bm25", "train", path)] if _present(path) else []
    # Scope is deliberately constrained to train/validation; a test directory
    # is never enumerated or read.
    try:
        top_level = [p for p in run_dir.iterdir()
                     if stat.S_ISDIR(p.stat().st_mode) and p.name != "branches"]
        branches = run_dir / "branches"
        candidate_dirs = ([p for p in branches.iterdir()
                           if stat.S_ISDIR(p.stat().st_mode) and p.name.startswith("candidate-")]
                          if _present(branches) else [])
    except OSError as exc:
        raise ValueError(f"cannot enumerate historical run answer directories: {exc}") from exc
    found = []
    for policy_dir in sorted(top_level + candidate_dirs):
        policy = policy_dir.name
        if policy_dir.parent.name == "branches":
            policy = f"branches/{policy}"
        for fold in ("train", "validation"):
            path = policy_dir / fold / "answers.jsonl"
            if _present(path):
                found.append((policy, fold, path))
    return found


def _trace(trace_id: str, span_id: str, name: str, kind: str,
           timestamp_ns: int, run_id: str, fold: str, policy: str,
           input_value: Any, output_value: Any, provenance: str,
           status: str = "ok", error_type: str | None = None,
           parent_span_id: str | None = None, links: list[dict[str, str]] | None = None,
           metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if provenance not in PROVENANCE:
        raise ValueError(f"invalid provenance: {provenance}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "kind": kind,
        # Historical wall-clock times were not retained. Equal timestamps
        # denote import-time placement, never reconstructed execution latency.
        "start_time_ns": timestamp_ns,
        "end_time_ns": timestamp_ns,
        "input": input_value,
        "output": output_value,
        "metadata": {
            "run_id": run_id,
            "fold": fold,
            "policy": policy,
            "timing": "import_time_not_original",
            **(metadata or {}),
        },
        "model": None,
        "usage": None,
        "status": status,
        "error_type": error_type,
        "provenance": provenance,
    }
    if parent_span_id is not None:
        result["parent_span_id"] = parent_span_id
    if links:
        result["links"] = links
    return result


def _case_trace(run_identity: str, run_id: str, policy: str, fold: str,
                row: dict[str, Any], timestamp_ns: int, decision_trace_id: str,
                diagnostic_span_id: str, selected: bool, input_status: str) -> dict[str, Any]:
    query_id = str(row.get("query_id", "unknown"))
    trace_id = _stable_id(f"rsi-case:{run_identity}:{policy}:{fold}:{query_id}", 32)
    span_id = _stable_id(f"case-span:{trace_id}", 16)
    query_keys = ("persona_id", "query_id", "session_id", "query_type", "question")
    result_keys = ("answer", "actor_status", "trajectory", "evidence", "tool_audit",
                   "score", "status", "reason", "call_log")
    recorded_query = {key: row[key] for key in query_keys if key in row}
    recorded_result = {key: row[key] for key in result_keys if key in row}
    provenance = "historical_recorded" if recorded_query or recorded_result else "unavailable"
    meta = {
        "selected_diagnostic_candidate": selected,
        "diagnostic_input_status": input_status if selected else "not_selected",
        "attribution": "unverified",
        "field_provenance": {key: "historical_recorded" for key in (*recorded_query, *recorded_result)},
    }
    return _trace(
        trace_id, span_id, "rsi.historical.case", "agent", timestamp_ns,
        run_id, fold, policy,
        {"recorded_query": recorded_query},
        {"recorded_result": recorded_result, "query_id": query_id},
        provenance, status="ok" if row.get("status") not in ("actor_failed", "error") else "error",
        error_type=row.get("status") if row.get("status") in ("actor_failed", "error") else None,
        links=[{"trace_id": decision_trace_id, "span_id": diagnostic_span_id,
                "relationship": "selected_diagnostic_example" if selected else "run_case"}],
        metadata=meta)


def build_backfill(run_dir: str | Path, *, scope: str = "diagnostic",
                   imported_at_ns: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build historical trace rows and an identifier-only index in memory."""
    run_dir = Path(run_dir)
    if scope not in ("diagnostic", "all"):
        raise ValueError("scope must be 'diagnostic' or 'all'")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    config = _json(run_dir / "config.json")
    config_hash = _json(run_dir / "config-hash.json")
    summary = _json(run_dir / "summary.json")
    run_identity = _run_identity(run_dir, config, config_hash)
    decision_trace_id = _stable_id(f"rsi-decision:{run_identity}", 32)
    run_id = str((config or {}).get("run_id") or run_dir.parent.name or run_dir.name)
    timestamp_ns = imported_at_ns if imported_at_ns is not None else time.time_ns()
    answers_paths = _answers_files(run_dir, scope)
    answer_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        (policy, fold): _jsonl(path) for policy, fold, path in answers_paths
    }
    baseline_rows = answer_rows.get(("plain-bm25", "train"), [])
    selected = select_diagnostic_examples(baseline_rows)
    chosen_ids = [str(row.get("query_id", "unknown")) for row in selected]
    reflection = _json(run_dir / "reflexion.json")
    if reflection is None:
        reflection = _json(run_dir / "reflexion-failure.json")
    interventions = _json(run_dir / "interventions.json")
    if not isinstance(interventions, list):
        interventions = None
    parent_policy_source = _source_record(run_dir / "policy-git" / "plain-bm25.py")

    rows: list[dict[str, Any]] = []
    root_span_id = _stable_id(f"decision-root:{decision_trace_id}", 16)
    diagnostic_span_id = _stable_id(f"diagnostics:{decision_trace_id}", 16)
    rows.append(_trace(
        decision_trace_id, root_span_id, "rsi.historical.decision", "chain",
        timestamp_ns, run_id, "historical", "rsi",
        {"run_artifacts": {"config": config is not None,
                           "summary": summary is not None,
                           "baseline_train_answers": bool(baseline_rows)}},
        {"chosen_case_ids": chosen_ids,
         "selected_case_count": len(selected),
         "diagnostic_example_count": (reflection.get("diagnostic_example_count")
                                      if isinstance(reflection, dict) else None),
         "historical_summary": _summary_context(summary)},
        "historical_reconstructed", metadata={"run_identity_sha256": run_identity,
                                              "scope": scope,
                                              "test_data_ingested": False}))

    count = len(selected)
    reflection_count = reflection.get("diagnostic_example_count") if isinstance(reflection, dict) else None
    confirmed_count = min(count, reflection_count) if isinstance(reflection_count, int) else None
    diagnostic_output = {
        "baseline_train_answer_count": len(baseline_rows),
        "selected_case_ids": chosen_ids,
        "selected_case_count": count,
        "reconstructed_reflexion_example_prefix_count": confirmed_count,
        "recorded_parent_policy": parent_policy_source,
        "interventions": interventions,
        "interventions_provenance": "historical_recorded" if interventions is not None else "unavailable",
    }
    rows.append(_trace(
        decision_trace_id, diagnostic_span_id, "rsi.historical.diagnostic_bundle", "span",
        timestamp_ns, run_id, "train", "plain-bm25",
        {"selection_rule": "first 4 score==0 followed by first 2 score==1 in answer-file order",
         "input_artifact": "plain-bm25/train/answers.jsonl"}, diagnostic_output,
        "historical_recorded" if baseline_rows else "unavailable",
        parent_span_id=root_span_id,
        metadata={"case_links_are_proposal_neutral": True,
                  "prompt_input_reconstruction": "prefix inferred from recorded count and old packing code; exact request absent",
                  "old_compact_example_list_cap": 5}))

    reflection_span_id = _stable_id(f"reflection:{decision_trace_id}", 16)
    if isinstance(reflection, dict):
        ref_provenance = "historical_recorded"
        reflection_output = reflection
        ref_status = "error" if reflection.get("status") == "failed" else "ok"
    else:
        ref_provenance = "unavailable"
        reflection_output = {"reason": "reflexion.json and reflexion-failure.json unavailable"}
        ref_status = "error"
    rows.append(_trace(
        decision_trace_id, reflection_span_id, "rsi.historical.reflection", "span",
        timestamp_ns, run_id, "train", "plain-bm25",
        {"diagnostic_bundle_span_id": diagnostic_span_id}, reflection_output,
        ref_provenance, status=ref_status,
        error_type="reflection_failure" if ref_status == "error" else None,
        parent_span_id=root_span_id,
        metadata={"hypotheses_are_model_supplied_not_verified_causes": True,
                  "proposal_case_links": "not_recorded"}))

    branches_dir = run_dir / "branches"
    for candidate_index in range(1, 4):
        name = f"candidate-{candidate_index}"
        branch = branches_dir / name
        proposal = _json(branch / "proposal.json")
        if proposal is None and isinstance(reflection, dict):
            proposals = reflection.get("proposals")
            if isinstance(proposals, list) and len(proposals) >= candidate_index:
                proposal = proposals[candidate_index - 1]
        source_path = branch / "policy.py"
        source = None
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError:
            pass
        failure = _json(branch / "failure.json")
        cost = _json(branch / "codegen-cost.json")
        if source is not None:
            output = {"source": source, "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                      "failure": failure}
            provenance = "historical_recorded"
            status = "error" if failure else "ok"
            error_type = "recorded_candidate_failure" if failure else None
        elif failure is not None:
            output = {"source": None, "sha256": None, "failure": failure}
            provenance = "historical_recorded"
            status = "error"
            error_type = str(failure.get("category") or failure.get("stage") or "candidate_failure")
        else:
            output = {"source": None, "sha256": None,
                      "reason": "candidate source and failure artifact unavailable"}
            provenance = "unavailable"
            status = "error"
            error_type = "candidate_artifacts_unavailable"
        span_id = _stable_id(f"codegen:{decision_trace_id}:{name}", 16)
        rows.append(_trace(
            decision_trace_id, span_id, "rsi.historical.codegen", "span",
            timestamp_ns, run_id, "train", name,
            {"proposal": proposal, "proposal_provenance": "historical_recorded" if proposal is not None else "unavailable"},
            output, provenance, status=status, error_type=error_type,
            parent_span_id=root_span_id,
            metadata={"codegen_cost": cost,
                      "codegen_cost_provenance": "historical_recorded" if cost is not None else "unavailable",
                      "proposal_case_links": "not_recorded"}))

    # Keep only aggregate artifacts; item-level validation answers are added
    # only by --scope all and still only from train/validation paths.
    report_bundle: dict[str, Any] = {"summary": summary,
                                     "reports": {},
                                     "cost_summaries": {}}
    policy_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name != "branches"]
    policy_dirs.extend(p for p in (run_dir / "branches").iterdir() if p.is_dir()) \
        if (run_dir / "branches").is_dir() else None
    for policy_dir in sorted(policy_dirs):
        relative_policy = policy_dir.relative_to(run_dir).as_posix()
        for fold in ("train", "validation"):
            report = _json(policy_dir / fold / "report.json")
            if report is not None:
                report_bundle["reports"][f"{relative_policy}/{fold}"] = report
        for filename in ("codegen-cost.json", "failure.json"):
            value = _json(policy_dir / filename)
            if value is not None:
                report_bundle["cost_summaries"][f"{relative_policy}/{filename}"] = value
    reports_span_id = _stable_id(f"reports:{decision_trace_id}", 16)
    report_provenance = "historical_recorded" if summary is not None or report_bundle["reports"] else "unavailable"
    rows.append(_trace(
        decision_trace_id, reports_span_id, "rsi.historical.validation_and_cost_summary", "span",
        timestamp_ns, run_id, "validation", "all-recorded-policies",
        {"source_artifacts": ["summary.json", "*/train/report.json", "*/validation/report.json"]},
        report_bundle, report_provenance, status="ok" if report_provenance != "unavailable" else "error",
        error_type=None if report_provenance != "unavailable" else "reports_unavailable",
        parent_span_id=root_span_id,
        metadata={"test_reports_read": False,
                  "policy_latency_fields_are_recorded_metadata_not_backfilled_span_duration": True}))

    case_rows: list[tuple[str, str, dict[str, Any], bool, str]] = []
    if scope == "diagnostic":
        for i, row in enumerate(selected):
            input_status = ("prefix_reconstructed_from_recorded_count"
                            if confirmed_count is not None and i < confirmed_count
                            else "candidate_selected_input_unconfirmed")
            case_rows.append(("plain-bm25", "train", row, True, input_status))
    else:
        chosen_keys = {(str(row.get("query_id")), str(row.get("persona_id"))) for row in selected}
        for (policy, fold), cases in answer_rows.items():
            for row in cases:
                key = (str(row.get("query_id")), str(row.get("persona_id")))
                is_selected = policy == "plain-bm25" and fold == "train" and key in chosen_keys
                case_rows.append((policy, fold, row, is_selected,
                                  "candidate_selected_input_unconfirmed" if is_selected else "not_selected"))
    for policy, fold, row, is_selected, input_status in case_rows:
        query_id = str(row.get("query_id", "unknown"))
        rows.append(_case_trace(run_identity, run_id, policy, fold, row,
                                timestamp_ns, decision_trace_id, diagnostic_span_id,
        is_selected, input_status))

    all_trace_ids = sorted({row["trace_id"] for row in rows})
    trace_refs = [{"trace_id": row["trace_id"], "span_id": row["span_id"],
                   "name": row["name"]} for row in rows]
    root_refs = [ref for ref, row in zip(trace_refs, rows)
                 if row["trace_id"] == decision_trace_id and row["span_id"] != root_span_id]
    root_refs.extend(ref for ref, row in zip(trace_refs, rows)
                     if row["name"] == "rsi.historical.case"
                     and row["metadata"].get("selected_diagnostic_candidate"))
    rows[0]["metadata"]["trace_refs"] = root_refs
    rows[0]["metadata"]["trace_ref_count"] = len(trace_refs)
    rows[0]["metadata"]["trace_refs_omitted"] = max(0, len(trace_refs) - len(root_refs))
    index = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "decision_trace_id": decision_trace_id,
        "trace_ids": all_trace_ids,
        "trace_refs": trace_refs,
        "trace_count": len({row["trace_id"] for row in rows}),
        "observation_count": len(rows),
        "case_trace_count": len(case_rows),
        "selected_diagnostic_case_count": count,
        "scope": scope,
        "test_data_ingested": False,
        "timing": "import_time_not_original",
    }
    return rows, index


def write_backfill(run_dir: str | Path, output: str | Path, *, scope: str = "diagnostic",
                   imported_at_ns: int | None = None) -> dict[str, Any]:
    """Create a new spool directory with traces.jsonl and identifier-only index."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    traces, index = build_backfill(run_dir, scope=scope,
                                   imported_at_ns=imported_at_ns)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output, 0o700)
    try:
        with (output / "traces.jsonl").open("x", encoding="utf-8") as stream:
            for trace in traces:
                stream.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.chmod(output / "traces.jsonl", 0o600)
        with (output / "index.json").open("x", encoding="utf-8") as stream:
            json.dump(index, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(output / "index.json", 0o600)
    except BaseException:
        # Do not leave a partial-looking spool that could be mistaken for a
        # complete backfill; only remove files created in the fresh directory.
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
        raise
    return index
