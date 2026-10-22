"""Structured, auditable stages for RSI reflexion.

Validation here checks the response contract and evidence references. It does
not decide whether a proposed causal explanation is true.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from .model import ModelServiceUnavailable


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["clusters"],
    "properties": {"clusters": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["label", "observed_failure", "supporting_case_ids",
                     "cause_hypotheses", "attribution_status", "discriminating_test"],
        "properties": {
            "label": {"type": "string"},
            "observed_failure": {"type": "string", "description": "Concrete observed failure from supplied cases; state observations, not an inferred cause or a confidence word."},
            "supporting_case_ids": {"type": "array", "items": {"type": "string"}},
            "cause_hypotheses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["mechanism", "supporting_observations", "counter_evidence", "confidence"],
                "properties": {
                    "mechanism": {"type": "string", "description": "A causal mechanism hypothesis, distinct from the observed failure. Do not state only a confidence level."},
                    "supporting_observations": {"type": "array", "items": {"type": "string"}},
                    "counter_evidence": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"enum": ["low", "medium", "high"], "description": "Confidence in this hypothesis, not an explanation or mechanism."},
                },
            }},
            "attribution_status": {"enum": ["hypothesized", "unresolved"], "description": "Use unresolved and an empty cause_hypotheses array when available evidence does not support a useful hypothesis."},
            "discriminating_test": {"type": "string"},
        },
    }}}
}

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["proposals"],
    "properties": {"proposals": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["direction", "hypothesis", "prediction", "risk", "basis", "rationale",
                     "mechanism", "falsification_test", "supporting_case_ids",
                     "supporting_cluster_labels", "uncertainty"],
        "properties": {
            **{key: {"type": "string"} for key in
               ("direction", "hypothesis", "prediction", "risk", "rationale", "mechanism",
                "falsification_test")},
            "basis": {"enum": ["training_cases", "parent_code", "exploratory"]},
            "supporting_case_ids": {"type": "array", "items": {"type": "string"}},
            "supporting_cluster_labels": {"type": "array", "items": {"type": "string"}},
            "uncertainty": {"enum": ["low", "medium", "high"]},
        },
    }}}
}


class ProtocolValidationError(ValueError):
    """A response failed the structured protocol contract."""

    def __init__(self, issues: Iterable[str]):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues) or "Protocol validation failed")


class ReflexionStageFailure(ValueError):
    """A reflexion stage exhausted its bounded attempts."""

    def __init__(self, stage: str, category: str, attempts: dict[str, Any]):
        self.stage = stage
        self.category = category
        self.attempts = attempts
        failed_attempts = attempts.get("attempts", [])
        issues = failed_attempts[-1].get("validation_errors", []) if failed_attempts else []
        # These validator messages contain field paths and error types only;
        # cited source values are deliberately omitted.
        detail = "; ".join(str(issue) for issue in issues[:3])
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Reflexion stage {stage!r} failed ({category}) after "
                         f"{len(failed_attempts)} attempt(s){suffix}")


_CONFIDENCE = {"low", "medium", "high"}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _confidence_only(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _CONFIDENCE


def _case_ids(value: Any, actual: set[str], path: str, issues: list[str]) -> None:
    if not isinstance(value, list):
        issues.append(f"{path} must be an array of case IDs")
        return
    for index, case_id in enumerate(value):
        if not isinstance(case_id, str):
            issues.append(f"{path}[{index}] must be a string case ID")
        elif case_id not in actual:
            issues.append(f"{path}[{index}] cites a case ID absent from input")


def _unknown_fields(obj: dict[str, Any], allowed: set[str], path: str, issues: list[str]) -> None:
    for key in obj.keys() - allowed:
        issues.append(f"{path} has unexpected field {key!r}")


def _string_array(value: Any, path: str, issues: list[str], *, nonempty_items: bool = True) -> None:
    if not isinstance(value, list):
        issues.append(f"{path} must be an array of strings")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or (nonempty_items and not item.strip()):
            issues.append(f"{path}[{index}] must be a non-empty string")


def validate_diagnosis(obj: Any, case_ids: Iterable[str]) -> None:
    issues: list[str] = []
    actual = set(case_ids)
    if not isinstance(obj, dict):
        raise ProtocolValidationError(["root must be an object"])
    _unknown_fields(obj, {"clusters"}, "root", issues)
    clusters = obj.get("clusters")
    if not isinstance(clusters, list):
        raise ProtocolValidationError(["clusters must be an array"])
    labels: set[str] = set()
    for ci, cluster in enumerate(clusters):
        path = f"clusters[{ci}]"
        if not isinstance(cluster, dict):
            issues.append(f"{path} must be an object")
            continue
        _unknown_fields(cluster, {"label", "observed_failure", "supporting_case_ids", "cause_hypotheses",
                                  "attribution_status", "discriminating_test"}, path, issues)
        for field in ("label", "observed_failure", "discriminating_test"):
            if not _nonempty(cluster.get(field)):
                issues.append(f"{path}.{field} must be a non-empty string")
        label = cluster.get("label")
        if isinstance(label, str):
            if label in labels:
                issues.append(f"{path}.label must be unique")
            labels.add(label)
        if _confidence_only(cluster.get("observed_failure")):
            issues.append(f"{path}.observed_failure must describe an observation, not only confidence")
        _case_ids(cluster.get("supporting_case_ids"), actual, f"{path}.supporting_case_ids", issues)
        if not cluster.get("supporting_case_ids"):
            issues.append(f"{path}.supporting_case_ids must identify at least one case for a non-empty cluster")
        hypotheses = cluster.get("cause_hypotheses")
        status = cluster.get("attribution_status")
        if status not in ("hypothesized", "unresolved"):
            issues.append(f"{path}.attribution_status must be hypothesized or unresolved")
        if not isinstance(hypotheses, list):
            issues.append(f"{path}.cause_hypotheses must be an array")
            continue
        if status == "hypothesized" and not hypotheses:
            issues.append(f"{path}.cause_hypotheses must be non-empty when attribution is hypothesized")
        for hi, hypothesis in enumerate(hypotheses):
            hpath = f"{path}.cause_hypotheses[{hi}]"
            if not isinstance(hypothesis, dict):
                issues.append(f"{hpath} must be an object")
                continue
            _unknown_fields(hypothesis, {"mechanism", "supporting_observations", "counter_evidence", "confidence"},
                            hpath, issues)
            mechanism = hypothesis.get("mechanism")
            if not _nonempty(mechanism):
                issues.append(f"{hpath}.mechanism must be a non-empty string")
            elif _confidence_only(mechanism):
                issues.append(f"{hpath}.mechanism must explain a mechanism, not only confidence")
            _string_array(hypothesis.get("supporting_observations"), f"{hpath}.supporting_observations", issues)
            _string_array(hypothesis.get("counter_evidence"), f"{hpath}.counter_evidence", issues)
            if hypothesis.get("confidence") not in _CONFIDENCE:
                issues.append(f"{hpath}.confidence must be low, medium, or high")
    if issues:
        raise ProtocolValidationError(issues)


def validate_proposals(obj: Any, case_ids: Iterable[str], cluster_labels: Iterable[str], count: int = 3) -> None:
    issues: list[str] = []
    actual_ids, actual_labels = set(case_ids), set(cluster_labels)
    if not isinstance(obj, dict):
        raise ProtocolValidationError(["root must be an object"])
    _unknown_fields(obj, {"proposals"}, "root", issues)
    proposals = obj.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != count:
        raise ProtocolValidationError([f"proposals must contain exactly {count} items"])
    required_text = ("direction", "hypothesis", "prediction", "risk", "rationale", "mechanism",
                     "falsification_test")
    for pi, proposal in enumerate(proposals):
        path = f"proposals[{pi}]"
        if not isinstance(proposal, dict):
            issues.append(f"{path} must be an object")
            continue
        _unknown_fields(proposal, set(required_text) | {"basis", "supporting_case_ids",
                                                       "supporting_cluster_labels", "uncertainty"}, path, issues)
        for field in required_text:
            if not _nonempty(proposal.get(field)):
                issues.append(f"{path}.{field} must be a non-empty string")
        if proposal.get("basis") not in ("training_cases", "parent_code", "exploratory"):
            issues.append(f"{path}.basis must be training_cases, parent_code, or exploratory")
        if proposal.get("uncertainty") not in _CONFIDENCE:
            issues.append(f"{path}.uncertainty must be low, medium, or high")
        _case_ids(proposal.get("supporting_case_ids"), actual_ids, f"{path}.supporting_case_ids", issues)
        labels = proposal.get("supporting_cluster_labels")
        if not isinstance(labels, list):
            issues.append(f"{path}.supporting_cluster_labels must be an array of labels")
        else:
            for index, label in enumerate(labels):
                if not isinstance(label, str):
                    issues.append(f"{path}.supporting_cluster_labels[{index}] must be a string")
                elif label not in actual_labels:
                    issues.append(f"{path}.supporting_cluster_labels[{index}] cites a cluster label absent from input")
        if proposal.get("basis") == "training_cases" and not proposal.get("supporting_case_ids"):
            issues.append(f"{path}.basis training_cases requires at least one supporting case")
    if issues:
        raise ProtocolValidationError(issues)


def _persist(artifact_dir: str | Path | None, stage: str, record: dict[str, Any], attempt: dict[str, Any] | None = None) -> None:
    if artifact_dir is None:
        return
    directory = Path(artifact_dir) / stage
    directory.mkdir(parents=True, exist_ok=True)
    if attempt is not None:
        path = directory / f"attempt-{len(record['attempts']):03d}.json"
        _write_private_json(path, attempt)
    path = directory / "result.json"
    _write_private_json(path, record)


def _write_private_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalized_cost(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    return {"usage": copy.deepcopy(metadata.get("usage", {})) or {},
            "latency_s": metadata.get("latency_s"), "role": metadata.get("role", "reflexion")}


def _new_call_metadata(client: Any, call_start: int | None) -> dict[str, Any] | None:
    calls = getattr(client, "calls", None)
    if call_start is None or calls is None or len(calls) <= call_start:
        return None
    latest = calls[-1]
    return latest if isinstance(latest, dict) else None


def call_validated_stage(client: Any, stage: str, messages: list[dict[str, str]], schema: dict[str, Any],
                         validator: Callable[[Any], None], max_corrections: int = 1,
                         artifact_dir: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Call one stage, retaining every full request and response for audit.

    ``max_corrections`` bounds all retries, including protocol correction
    attempts. ModelServiceUnavailable is recorded then propagated immediately.
    """
    if not isinstance(max_corrections, int) or max_corrections < 0:
        raise ValueError("max_corrections must be a non-negative integer")
    base_messages = copy.deepcopy(messages)
    record: dict[str, Any] = {"stage": stage, "status": "running", "attempts": []}
    last_category = "unknown"
    for index in range(max_corrections + 1):
        attempt_messages = copy.deepcopy(base_messages)
        if index:
            previous = record["attempts"][-1]
            previous_response = previous.get("response")
            previous_text = previous_response.get("text", "") if isinstance(previous_response, dict) else ""
            correction = (
                "Correct the prior response to satisfy the requested JSON structure and the listed field errors. "
                "Keep observations separate from cause hypotheses. An attribution may remain unresolved with "
                "an empty hypothesis list. Do not invent observations, evidence, or causes. Return only JSON.\n"
                "Field or attempt errors:\n" + "\n".join(previous.get("validation_errors", []))
            )
            attempt_messages.extend([
                {"role": "assistant", "content": str(previous_text)},
                {"role": "user", "content": correction},
            ])
        attempt: dict[str, Any] = {
            "attempt": index + 1, "input_messages": attempt_messages, "response": None,
            "parsed": None, "validation_errors": [], "status": "running", "usage": {}, "cost": None,
        }
        record["attempts"].append(attempt)
        _persist(artifact_dir, stage, record, attempt)
        client_calls = getattr(client, "calls", None)
        call_start = len(client_calls) if hasattr(client_calls, "__len__") else None
        try:
            stage_client = client.for_stage(stage) if hasattr(client,"for_stage") else client
            response = stage_client.complete(attempt_messages, role="reflexion", json_mode=True, json_schema=schema)
            attempt["response"] = copy.deepcopy(response)
            attempt["usage"] = copy.deepcopy(response.get("usage", {})) if isinstance(response, dict) else {}
            attempt["cost"] = _normalized_cost(response)
            raw = response.get("text") if isinstance(response, dict) else None
            try:
                if not isinstance(raw, str):
                    raise TypeError("response.text must be a string")
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                attempt["validation_errors"] = [f"response.text is not valid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else type(exc).__name__}"]
                last_category = "invalid_json"
            else:
                attempt["parsed"] = parsed
                try:
                    validator(parsed)
                except ProtocolValidationError as exc:
                    attempt["validation_errors"] = list(exc.issues)
                    last_category = "validation_error"
                except (ValueError, TypeError, KeyError) as exc:
                    attempt["validation_errors"] = [str(exc)]
                    last_category = "validation_error"
                else:
                    attempt["status"] = "validated"
                    record.update(status="validated", result=parsed)
                    _persist(artifact_dir, stage, record, attempt)
                    return parsed, record
        except ModelServiceUnavailable as exc:
            attempt["status"] = "service_unavailable"
            attempt["validation_errors"] = ["ModelServiceUnavailable"]
            latest_call = _new_call_metadata(client, call_start)
            if latest_call is not None:
                attempt["usage"] = copy.deepcopy(latest_call.get("usage", {}))
                attempt["cost"] = _normalized_cost(latest_call)
            last_category = "service_unavailable"
            record["status"] = last_category
            exc.stage = stage
            exc.category = last_category
            exc.attempts = record
            _persist(artifact_dir, stage, record, attempt)
            raise
        except Exception as exc:
            latest_call = _new_call_metadata(client, call_start)
            if latest_call is not None:
                attempt["usage"] = copy.deepcopy(latest_call.get("usage", {}))
                attempt["cost"] = _normalized_cost(latest_call)
            attempt["status"] = "call_error"
            attempt["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
            last_category = getattr(exc, "category", "call_error")
        if attempt["status"] == "running":
            attempt["status"] = "invalid"
        _persist(artifact_dir, stage, record, attempt)
    record["status"] = "failed"
    record["category"] = last_category
    _persist(artifact_dir, stage, record)
    raise ReflexionStageFailure(stage, last_category, record)
