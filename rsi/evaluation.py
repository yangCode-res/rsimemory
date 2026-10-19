"""Offline strict scoring for RSI's six task families."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any
from .model import ModelServiceUnavailable


TYPES = (
    "entity_exact_match",
    "temporal_localization",
    "state_update",
    "multiple_choice",
    "inference_generation",
    "multi_hop_clinical_deduction",
)
_JUDGE_TYPES = {"temporal_localization", "state_update", "inference_generation",
                "multi_hop_clinical_deduction"}


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    # Keep letters/digits (including CJK); discard punctuation and spacing.
    return "".join(ch for ch in text if ch.isalnum())


def _entity_normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join("".join(ch if ch.isalnum() else "|" if ch in ".!?。！？;；\n" else " "
                      for ch in text).split())


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if isinstance(v, (str, int, float))]
    if isinstance(value, (tuple, list, set)):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                for key in ("answer", "text", "label", "value"):
                    if isinstance(item.get(key), (str, int, float)):
                        result.append(str(item[key]))
                        break
        return result
    return []


def _positive_answers(query: dict[str, Any]) -> list[str]:
    # Only explicit positive-answer fields count; do not guess from other
    # annotation fields whose meaning may vary by dataset version.
    for key in ("positive_answers", "reference_answers"):
        values = _as_strings(query.get(key))
        if values:
            return values
    return []


def _contains_entity(response: str, entity: str) -> bool:
    """Match an entity mention unless its local context explicitly negates it."""
    answer_n, entity_n = _entity_normalize(response), _entity_normalize(entity)
    if not answer_n or not entity_n:
        return False
    if len(entity_n) == 1 and any("\u4e00" <= ch <= "\u9fff" for ch in entity_n):
        if answer_n != entity_n:
            return False
    start = 0
    while True:
        index = answer_n.find(entity_n, start)
        if index < 0:
            return False
        # Normalization removes punctuation. To prevent a short Latin entity
        # matching inside a longer word, require alphanumeric boundaries.
        before = answer_n[index - 1] if index else ""
        after_at = index + len(entity_n)
        after = answer_n[after_at] if after_at < len(answer_n) else ""
        latinish = all(ord(ch) < 128 for ch in entity_n)
        bounded = not latinish or (not before.isalnum() and not after.isalnum())
        if bounded and not _is_negated_entity_mention(answer_n, index, after_at):
            return True
        start = index + 1


def _is_negated_entity_mention(answer_n: str, start: int, end: int) -> bool:
    """Detect common nearby English and Chinese negation without judging semantics."""
    separators = "|"
    left = max((answer_n.rfind(mark, 0, start) for mark in separators), default=-1) + 1
    right_candidates = [answer_n.find(mark, end) for mark in separators]
    right_candidates = [index for index in right_candidates if index >= 0]
    right = min(right_candidates, default=len(answer_n))
    clause = answer_n[left:right]
    local_before = clause[:max(0, start - left)].strip()
    local_after = clause[max(0, end - left):].strip()
    english_before = re.search(
        r"(?:\bno\b|\bnot\b|\bwithout\b|\bden(?:y|ies|ied)\b|"
        r"\bnegative for\b|\babsence of\b|\bnever\b|\bdoes not have\b|"
        r"\bdoesn't have\b|\bdo not have\b)(?:\s+\w+){0,5}\s*$",
        local_before,
    )
    english_after = re.match(
        r"^(?:(?:is|was|has been)\s+)?(?:ruled out|excluded|absent|negative)\b",
        local_after,
    )
    chinese_before = re.search(
        r"(?:没有|並無|并无|无|否认|未见|未发现|未患|不患|排除|不支持|不存在|不再有).{0,8}$",
        local_before,
    )
    chinese_after = re.match(r"^(?:已排除|排除|不支持|否认|不存在|阴性)", local_after)
    return bool(english_before or english_after or chinese_before or chinese_after)


def _option_map(query: dict[str, Any]) -> dict[str, str]:
    raw = query.get("options")
    if isinstance(raw, dict):
        return {str(key).strip().upper(): str(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {chr(65 + index): str(value) for index, value in enumerate(raw)}
    return {}


def _choice_set(values: Any, options: dict[str, str]) -> set[str]:
    candidates = _as_strings(values)
    normalized_options = {_normalize(label): label for label in options}
    normalized_options.update({_normalize(text): label for label, text in options.items()})
    result = set()
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized in normalized_options:
            result.add(normalized_options[normalized])
            continue
        # Allow a concise selection such as "A、C" or "A and C". Any extra
        # selected option remains visible and causes strict set mismatch.
        for label in options:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])",
                         candidate, flags=re.IGNORECASE):
                result.add(label)
    return result


def _multiple_choice(query: dict[str, Any], answer: str) -> tuple[int | None, str]:
    options = _option_map(query)
    gold_values = query.get("correct_options", query.get("positive_answers"))
    gold = _choice_set(gold_values, options) if options else set()
    if not options or not gold:
        return None, "missing_or_invalid_choice_key"
    predicted = _choice_set([answer], options)
    return int(predicted == gold), "exact option-set match" if predicted == gold else \
        f"option set mismatch (expected {sorted(gold)}, got {sorted(predicted)})"


def _judge_prompt(query: dict[str, Any], answer: str) -> list[dict[str, str]]:
    kind = str(query.get("query_type", query.get("type", "")))
    rubric = {
        "temporal_localization": "Check whether the answer identifies the correct event and time or temporal ordering in the supplied reference.",
        "state_update": "Check whether the answer gives the correct updated state at the requested point in time and avoids stale or contradictory state.",
        "inference_generation": "Check whether the requested inference is supported by the reference and answers the question directly.",
        "multi_hop_clinical_deduction": "Check every required fact and causal link. Full credit requires complete coverage of the relevant chain and no material unsupported or contradictory claim.",
    }.get(kind, "Compare the answer with the reference and judge correctness strictly.")
    # This concise rubric is a custom judging template, not an official verbatim
    # rubric. Keep only the offline question/reference fields in this payload.
    reference = {key: query[key] for key in
                 ("question", "positive_answers", "reference_answers", "reference",
                  "evidence", "required_facts", "causal_chain") if key in query}
    payload = {"task": kind, "question": reference.get("question", ""),
               "reference": reference, "candidate_answer": answer, "criterion": rubric}
    system = (
        "You are a strict offline evaluator. This is a custom judging template, not the "
        "original paper's official rubric. Score only the supplied question and reference. "
        "Return exactly one JSON object: {\"score\":0 or 1,\"reason\":\"brief explanation\"}. "
        "Use 1 only when the answer satisfies the criterion; otherwise use 0. Do not add keys."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def evaluate(query: dict[str, Any], answer: str, client: Any) -> dict[str, Any]:
    """Score one private query; model/judge failures remain null, never correct."""
    kind = str(query.get("query_type", query.get("type", "")))
    if kind not in TYPES:
        return {"score": None, "status": "invalid_query_type", "reason": "unknown query type",
                "call_log": [], "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}
    if not isinstance(answer, str):
        return {"score": None, "status": "invalid_answer", "reason": "answer must be text",
                "call_log": [], "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}
    if kind == "entity_exact_match":
        positives = _positive_answers(query)
        if not positives:
            return {"score": None, "status": "missing_reference", "reason": "no positive answers",
                    "call_log": [], "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}
        hit = any(_contains_entity(answer, gold) for gold in positives)
        return {"score": int(hit), "status": "scored",
                "reason": "normalized entity match" if hit else "no normalized positive entity match",
                "call_log": [], "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}
    if kind == "multiple_choice":
        score, reason = _multiple_choice(query, answer)
        return {"score": score, "status": "scored" if score is not None else "missing_reference",
                "reason": reason, "call_log": [],
                "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}
    if kind not in _JUDGE_TYPES:
        return {"score": None, "status": "unsupported_query_type", "reason": kind,
                "call_log": [], "cost": {"calls": 0, "latency_s": 0.0, "usage": {}}}

    response = None
    log: list[dict[str, Any]] = []
    try:
        response = client.complete(_judge_prompt(query, answer), role="judge", json_mode=True)
        if not isinstance(response, dict) or not isinstance(response.get("text"), str):
            raise ValueError("client.complete must return a dict containing text")
        usage = response.get("usage") or {}
        latency = float(response.get("latency_s") or 0.0)
        log = [{"role": "judge", "usage": usage, "latency_s": latency}]
        decoded = json.loads(response["text"])
        if not isinstance(decoded, dict) or set(decoded) != {"score", "reason"}:
            raise ValueError("judge JSON must contain only score and reason")
        score = decoded["score"]
        if type(score) is not int or score not in (0, 1) or not isinstance(decoded["reason"], str):
            raise ValueError("judge score must be integer 0/1 and reason must be text")
        numeric_usage = {key: value for key, value in usage.items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)}
        return {"score": score, "status": "scored", "reason": decoded["reason"],
                "call_log": log, "cost": {"calls": 1, "latency_s": latency,
                                            "usage": numeric_usage}}
    except ModelServiceUnavailable:
        raise
    except Exception as exc:
        latency = locals().get("latency", 0.0)
        usage = locals().get("usage", {})
        logs = log
        if not logs:
            logs = [{"role": "judge", "usage": usage if isinstance(usage, dict) else {},
                     "latency_s": latency, "error": f"{type(exc).__name__}: {exc}"}]
        numeric_usage = {key: value for key, value in usage.items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)} \
            if isinstance(usage, dict) else {}
        return {"score": None, "status": "judge_failed", "reason": f"{type(exc).__name__}: {exc}",
                "call_log": logs, "cost": {"calls": len(logs), "latency_s": latency,
                                            "usage": numeric_usage}}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep strict AVG and add six-type E2E AVG with actor non-answers as zero.

    Judge failures, malformed scores, unknown types and missing types remain
    unscorable. Transport failures abort replay upstream, never become zero.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    abnormal = 0
    scored_total = 0
    for row in rows:
        kind = str(row.get("query_type", row.get("type", "")))
        status = str(row.get("status", row.get("grading_status", "missing_status")))
        score = row.get("score")
        valid = (kind in TYPES and status == "scored" and isinstance(score, int)
                 and not isinstance(score, bool) and score in (0, 1))
        status_counts[status] += 1
        if not valid:
            abnormal += 1
        else:
            scored_total += 1
        unanswered = kind in TYPES and status == "actor_failed" and score is None
        grouped[kind].append({"valid": valid, "score": score,
                              "unanswered_as_zero": unanswered})

    accuracy: dict[str, float | None] = {}
    counts_by_type: dict[str, dict[str, int]] = {}
    e2e_accuracy: dict[str, float | None] = {}
    complete = True
    e2e_complete = True
    unanswered_total = 0
    for kind in TYPES:
        items = grouped.get(kind, [])
        valid_items = [item for item in items if item["valid"]]
        unanswered_count = sum(item['unanswered_as_zero'] for item in items)
        unanswered_total += unanswered_count
        counts_by_type[kind] = {"total": len(items), "scored": len(valid_items),
                                "abnormal": len(items) - len(valid_items)}
        if not items or len(valid_items) != len(items):
            accuracy[kind] = None
            complete = False
        else:
            accuracy[kind] = sum(item["score"] for item in valid_items) / len(items)
        if not items or len(valid_items) + unanswered_count != len(items):
            e2e_accuracy[kind] = None
            e2e_complete = False
        else:
            # Keep every question in the denominator, including non-answers.
            e2e_accuracy[kind] = sum(item['score'] for item in valid_items) / len(items)
    unknown_types = {kind: len(items) for kind, items in grouped.items() if kind not in TYPES}
    if unknown_types:
        complete = False
        e2e_complete = False
    avg = sum(value for value in accuracy.values() if value is not None) / len(TYPES) \
        if complete else None
    e2e_avg = sum(value for value in e2e_accuracy.values() if value is not None) / len(TYPES) \
        if e2e_complete else None
    return {"type_accuracy": accuracy, "AVG": avg,
            "metric_version": "strict-plus-e2e-v1",
            "e2e_type_accuracy": e2e_accuracy, "E2E_AVG": e2e_avg,
            "abnormal_rate": abnormal / len(rows) if rows else None,
            "counts": {"total": len(rows), "scored": scored_total, "abnormal": abnormal,
                       "unanswered_as_zero": unanswered_total,
                       "e2e_unscorable": abnormal - unanswered_total,
                       "by_status": dict(status_counts), "by_type": counts_by_type,
                       "unknown_type": unknown_types}}
