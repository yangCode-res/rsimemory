"""Training-only diagnostic index and deterministic coverage sampling.

Records describe observable run behavior. In particular, source-session overlap
is only a proxy for source retrieval and must not be read as factual recall.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


_EXAMPLE_ROW_FIELDS = (
    "query_id", "query_type", "question", "answer", "score", "reason",
    "evidence", "trajectory", "tool_audit",
)


def _session_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _lookup(query_lookup: Any, query_id: Any) -> dict[str, Any]:
    if not isinstance(query_lookup, dict):
        return {}
    value = query_lookup.get(query_id)
    return value if isinstance(value, dict) else {}


def _normalize_session(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 else None
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            return None
        return str(int(number))
    value = str(value).strip()
    if not value:
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return str(int(number))


def _not_future(session: str, checkpoint: Any) -> bool:
    current = _normalize_session(checkpoint)
    if current is None:
        return False
    return int(session) <= int(current)


def _tool_observations(row: dict[str, Any]) -> tuple[int, int, bool, set[str], bool]:
    audit = row.get("tool_audit")
    if isinstance(audit, dict):
        audit = audit.get("operations", audit.get("events", []))
    operations = [op for op in _items(audit) if isinstance(op, dict)]
    search_count = sum(op.get("action") == "search" for op in operations)
    read_count = sum(op.get("action") == "read" for op in operations)
    trajectory = [step for step in _items(row.get("trajectory")) if isinstance(step, dict)]
    forced_final = row.get("actor_status") == "forced_final" or any(
        step.get("action") == "forced_final" for step in trajectory
    )
    # Session overlap is deliberately a coarse proxy: it does not establish
    # that the retrieved text contained, or the actor used, a required fact.
    checkpoint = row.get("session_id")
    retrieved_sessions = {
        normalized
        for op in operations
        if op.get("action") in ("search", "read")
        if (normalized := _normalize_session(op.get("session_id"))) is not None
        and _not_future(normalized, checkpoint)
    }
    retrieved_sessions.update(
        normalized
        for op in operations if op.get("action") in ("search", "read")
        for sid in _items(op.get("session_ids"))
        if (normalized := _normalize_session(sid)) is not None
        and _not_future(normalized, checkpoint)
    )
    for op in operations:
        if op.get("action") not in ("search", "read"):
            continue
        for chunk_id in _items(op.get("ids")):
            match = re.search(r"(?:^|[-_])s(\d+)(?:[-_]|$)", str(chunk_id))
            normalized = _normalize_session(match.group(1)) if match else None
            if normalized is not None and _not_future(normalized, checkpoint):
                retrieved_sessions.add(normalized)
    # Actor evidence is the persisted observation payload for the search/read
    # IDs recorded in the audit; use its session IDs only as a retrieval proxy.
    for item in _items(row.get("evidence")):
        if isinstance(item, dict):
            normalized = _normalize_session(item.get("session_id"))
            if normalized is not None and _not_future(normalized, checkpoint):
                retrieved_sessions.add(normalized)
    source_observed = isinstance(row.get('evidence'), list) or any(
        op.get('action') in ('search', 'read')
        and any(key in op for key in ('ids', 'session_id', 'session_ids'))
        for op in operations
    )
    return search_count, read_count, bool(forced_final), retrieved_sessions, source_observed


def _gold_sessions(query: dict[str, Any], checkpoint: Any) -> set[str]:
    result: set[str] = set()
    for item in _items(query.get("evidence", query.get("source_evidence"))):
        if isinstance(item, dict):
            normalized = _normalize_session(item.get("session_id"))
            if normalized is not None and _not_future(normalized, checkpoint):
                result.add(normalized)
    return result


def _error_group(score: Any, status: Any, trajectory: Any, forced_final: bool,
                 search_count: int, read_count: int, source_hit: Any,
                 grading_status: Any = None) -> str:
    if type(score) is not int or score not in (0, 1):
        if grading_status == 'actor_failed' or status in ('forced_final_invalid', 'forced_final_failed'):
            return 'abnormal_actor_failure_observed'
        if grading_status == 'judge_failed':
            return 'abnormal_judge_failure_observed'
        if grading_status in ('missing_reference','invalid_query_type','invalid_answer','unsupported_query_type'):
            return 'abnormal_reference_or_schema_observed'
        return "abnormal_score"
    if score == 1:
        return "success"
    status_text = str(status or "").casefold()
    if forced_final:
        return "forced_final_observed"
    if status_text in {"actor_failed", "max_steps", "forced_final_invalid", "forced_final_failed"}:
        return "actor_error_observed"
    steps = [step for step in _items(trajectory) if isinstance(step, dict)]
    if any(step.get("action") == "invalid_protocol" for step in steps):
        return "invalid_protocol_observed"
    if any(step.get("action") in ("search", "read") and step.get("error") for step in steps):
        return "tool_error_observed"
    if search_count == 0:
        return "no_search_observed"
    if read_count == 0:
        return "no_read_observed"
    if source_hit is False:
        return "gold_session_miss_proxy"
    if source_hit is True:
        return "gold_session_hit_proxy"
    return "failure_other_observed"


def _stage_labels(rows: list[dict[str, Any]]) -> dict[str, str]:
    checkpoints = sorted({row.get("session_id") for row in rows if row.get("session_id") is not None},
                         key=_session_key)
    result: dict[str, str] = {}
    for row in rows:
        sid = row.get("session_id")
        case_id = str(row["_case_id"])
        if sid is None:
            result[case_id] = "unknown"
            continue
        try:
            numeric = int(sid)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and 10 <= numeric <= 100:
            result[case_id] = "early" if numeric <= 30 else "middle" if numeric <= 70 else "late"
        else:
            # Small synthetic fixtures often use checkpoints such as 1, 2, 3.
            # Divide their distinct ordered checkpoints into three ordinal bins.
            position = checkpoints.index(sid)
            n = len(checkpoints)
            band = min(2, (position * 3) // max(1, n))
            result[case_id] = ("early", "middle", "late")[band]
    return result


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record[field]) for record in records)
    return dict(sorted(counts.items()))


def _outcome_distributions(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    axes = {"by_patient": "patient", "by_type": "query_type", "by_stage": "stage"}
    result: dict[str, dict[str, dict[str, int]]] = {}
    for output_key, field in axes.items():
        groups: dict[str, dict[str, int]] = {}
        for record in records:
            label = str(record[field])
            outcomes = groups.setdefault(label, {"total": 0, "success": 0, "failure": 0, "abnormal": 0})
            outcomes["total"] += 1
            score = record["score"]
            outcome = "success" if type(score) is int and score == 1 else (
                "failure" if type(score) is int and score == 0 else "abnormal"
            )
            outcomes[outcome] += 1
        result[output_key] = dict(sorted(groups.items()))
    return result


def _stable_tie(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def _coverage_select(records: list[dict[str, Any]], limit: int, seed: int,
                     already: set[str] | None = None,
                     preferred_failures: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}
    chosen = set(already or ())
    covered: dict[str, set[str]] = {axis: set() for axis in ("patient", "query_type", "stage", "error_group")}
    covered_tuples: set[tuple[str, str, str]] = set()
    axes = ("patient", "query_type", "stage", "error_group")
    failures = preferred_failures or []
    weights = {"patient": 4, "query_type": 4, "stage": 3, "error_group": 2}

    while len(selected) < max(0, limit):
        candidates = [record for record in records if record["case_id"] not in chosen]
        if not candidates:
            break
        ranked: list[tuple[int, bool, int, str, dict[str, Any], list[str]]] = []
        for record in candidates:
            newly_covered = [axis for axis in axes if record[axis] not in covered[axis]
                             and record[axis] != "unknown"]
            # A success is selected as a useful contrast when it resembles an
            # already selected failure on patient, type, or stage.
            similarity = 0
            if record["score"] == 1 and failures:
                similarity = max(sum(record[axis] == failure[axis] for axis in axes[:3])
                                 for failure in failures)
            gain = sum(weights[axis] for axis in newly_covered)
            tuple_key = (record["patient"], record["query_type"], record["stage"])
            tuple_is_new = tuple_key not in covered_tuples
            if record["score"] == 1:
                gain += similarity
            ranked.append((gain, tuple_is_new, similarity, _stable_tie(seed, record["case_id"]), record, newly_covered))
        _, _, similarity, _, best, newly_covered = max(
            ranked, key=lambda item: (item[0], item[1], item[2], item[3]))
        tuple_key = (best["patient"], best["query_type"], best["stage"])
        new_tuple = tuple_key not in covered_tuples
        chosen.add(best["case_id"])
        selected.append(best)
        for axis in axes:
            if best[axis] != "unknown":
                covered[axis].add(best[axis])
        covered_tuples.add(tuple_key)
        why = [f"new_{axis}_coverage:{best[axis]}" for axis in newly_covered]
        if new_tuple:
            why.append("new_patient_type_stage_combination")
        if best["score"] == 1 and failures and similarity:
            why.append(f"similar_success_control:{similarity}_shared_axes")
        if not why:
            why.append("additional_case_for_requested_quota")
        reasons[best["case_id"]] = why
    return selected, reasons


def build_training_diagnostics(answers, query_lookup, failure_limit=12,
                               success_limit=4, seed=20261013):
    """Build a full training-answer index, distributions, and coverage sample.

    Inputs are expected to contain training answers only. The function does
    not load data or inspect validation/test splits.
    """
    answer_rows = [row for row in (answers or []) if isinstance(row, dict)]
    answer_rows = [dict(row) for row in answer_rows]
    # Stable IDs are preserved when supplied. Otherwise derive one from the
    # row identity fields and ordinal to remain robust to sparse fixtures.
    seen: Counter[str] = Counter()
    for ordinal, row in enumerate(answer_rows):
        base = row.get("query_id")
        if base is None:
            base = f"case-{ordinal + 1}"
        case_id = str(base)
        seen[case_id] += 1
        if seen[case_id] > 1:
            case_id = f"{case_id}#{seen[str(base)]}"
        row["_case_id"] = case_id
    stages = _stage_labels(answer_rows)

    index: list[dict[str, Any]] = []
    examples_by_id: dict[str, dict[str, Any]] = {}
    for row in answer_rows:
        case_id = row["_case_id"]
        query = _lookup(query_lookup, row.get("query_id"))
        search_count, read_count, forced_final, retrieved_sessions, has_source_observation = _tool_observations(row)
        gold_sessions = _gold_sessions(query, row.get("session_id"))
        source_hit = (bool(gold_sessions & retrieved_sessions) if gold_sessions and has_source_observation else None)
        score = row.get("score")
        patient = row.get("persona_id", row.get("patient_id", "unknown"))
        query_type = row.get("query_type", query.get("query_type", "unknown"))
        status = row.get("actor_status", row.get("status"))
        group = _error_group(score, status, row.get("trajectory"), forced_final,
                             search_count, read_count, source_hit, row.get('status'))
        record = {
            "case_id": case_id,
            "query_id": row.get("query_id"),
            "patient": str(patient) if patient is not None else "unknown",
            "session_id": row.get("session_id"),
            "stage": stages[case_id],
            "query_type": str(query_type) if query_type is not None else "unknown",
            "score": score,
            "status": status,
            "grading_status": row.get('status'),
            "actor_status": row.get('actor_status'),
            "reason": row.get("reason"),
            "error_group": group,
            "observations": {
                "search_count": search_count,
                "read_count": read_count,
                "forced_final": forced_final,
                "gold_source_session_ids": sorted(gold_sessions),
                "retrieved_source_session_ids": sorted(retrieved_sessions),
                "gold_source_session_hit_proxy": source_hit if source_hit is not None else "unknown",
            },
        }
        index.append(record)
        example = {key: row.get(key) for key in _EXAMPLE_ROW_FIELDS}
        example["query_id"] = row.get("query_id", case_id)
        example["query_type"] = row.get("query_type", query.get("query_type"))
        example["question"] = row.get("question", query.get("question"))
        example["reference_answers"] = query.get("reference_answers", query.get("positive_answers", []))
        example["source_evidence"] = query.get("evidence", [])
        example["patient"] = record["patient"]
        example["session_id"] = record["session_id"]
        example["stage"] = record["stage"]
        example["error_group"] = record["error_group"]
        example["observation"] = record["observations"]
        for key in ("required_facts", "causal_chain"):
            if key in query:
                example[key] = query[key]
        examples_by_id[case_id] = example

    successful = [record for record in index if type(record["score"]) is int and record["score"] == 1]
    failed = [record for record in index if type(record["score"]) is int and record["score"] == 0]
    abnormal = [record for record in index if type(record["score"]) is not int or record["score"] not in (0, 1)]
    summary = {
        "total_questions": len(index),
        "patients": _counts(index, "patient"),
        "tasks": _counts(index, "query_type"),
        "time_periods": _counts(index, "stage"),
        "success_count": len(successful),
        "failure_count": len(failed),
        "abnormal_count": len(abnormal),
        "error_groups": _counts(index, "error_group"),
        "outcome_distributions": _outcome_distributions(index),
    }

    selected_failures, failure_reasons = _coverage_select(failed, failure_limit, seed)
    selected_successes, success_reasons = _coverage_select(
        successful, success_limit, seed, already={r["case_id"] for r in selected_failures},
        preferred_failures=selected_failures,
    )
    chosen = selected_failures + selected_successes
    reasons = {**failure_reasons, **success_reasons}
    axes = ("patient", "query_type", "stage", "error_group")
    covered = {axis: sorted({record[axis] for record in chosen if record[axis] != "unknown"})
               for axis in axes}
    observed = {axis: sorted({record[axis] for record in index if record[axis] != "unknown"})
                for axis in axes}
    missing = {axis: [value for value in values if value not in covered[axis]]
               for axis, values in observed.items()}
    selection = {
        "version": "training-diagnostics-coverage-v1",
        "seed": seed,
        "targets": {"failures": failure_limit, "successes": success_limit},
        "actual": {"failures": len(selected_failures), "successes": len(selected_successes),
                   "total": len(chosen)},
        "selected_case_ids": [record["case_id"] for record in chosen],
        "selection_reasons": reasons,
        "coverage": covered,
        "uncovered": missing,
        "shortfall": {"failures": max(0, failure_limit-len(selected_failures)),
                      "successes": max(0, success_limit-len(selected_successes))},
    }
    return {
        "index": index,
        "summary": summary,
        "selection": selection,
        "examples": [examples_by_id[record["case_id"]] for record in chosen],
    }
