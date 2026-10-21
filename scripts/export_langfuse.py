"""Export RSI JSONL trace records to Langfuse's OTLP/HTTP JSON endpoint.

The exporter preserves recorder-generated IDs. This supports deliberate
retries, but OTLP ingestion does not promise deduplication, so retries may
produce duplicate observations. Fully accepted batches are skipped from a
receipt; partial-success batches are blocked because OTLP does not identify
which spans were accepted, and need separate manual resolution.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINT = "/api/public/otel/v1/traces"
DEFAULT_BATCH_BYTES = 512 * 1024


class ExportError(Exception):
    pass


class _NoRedirectHandler(HTTPRedirectHandler):
    """Do not forward sensitive OTLP payloads or auth headers on redirects."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_urlopen(request, timeout):
    return build_opener(_NoRedirectHandler).open(request, timeout=timeout)


def _is_loopback(url):
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _base_url(env):
    value = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST")
    if not value:
        raise ExportError("Set LANGFUSE_BASE_URL or LANGFUSE_HOST")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ExportError("Langfuse base URL must use HTTP or HTTPS")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ExportError("Langfuse base URL must not contain credentials or query data")
    return value.rstrip("/")


def _json_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _attribute(key, value):
    if isinstance(value, bool):
        typed = {"boolValue": value}
    elif isinstance(value, int) and not isinstance(value, bool):
        typed = {"intValue": str(value)}
    elif isinstance(value, float):
        typed = {"doubleValue": value}
    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
        typed = {"arrayValue": {"values": [{"stringValue": x} for x in value]}}
    else:
        typed = {"stringValue": _json_value(value)}
    return {"key": key, "value": typed}


def _normalized_usage(usage):
    """Return Langfuse's flat numeric usage map from common provider shapes."""
    if not isinstance(usage, dict):
        return {}
    flat = {key: value for key, value in usage.items()
            if isinstance(key, str) and isinstance(value, int) and
            not isinstance(value, bool) and value >= 0}
    aliases = {
        "input": ("input", "input_tokens", "prompt_tokens"),
        "output": ("output", "output_tokens", "completion_tokens"),
        "total": ("total", "total_tokens"),
    }
    removed = {alias for names in aliases.values() for alias in names}
    normalized = {key: value for key, value in flat.items() if key not in removed}
    for canonical, names in aliases.items():
        for alias in names:
            if alias in flat:
                normalized[canonical] = flat[alias]
                break
    return normalized


def record_to_span(record, trace_name=None):
    """Map one local record to a Langfuse-compatible OTLP JSON span."""
    for key, size in (("trace_id", 32), ("span_id", 16)):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != size or any(c not in "0123456789abcdef" for c in value):
            raise ExportError(f"Invalid {key} in trace record")
    attrs = []
    if record.get("input") is not None:
        attrs.append(_attribute("langfuse.observation.input", _json_value(record["input"])))
    if record.get("output") is not None:
        attrs.append(_attribute("langfuse.observation.output", _json_value(record["output"])))
    attrs.append(_attribute("langfuse.observation.type", record.get("kind") or "span"))
    # Langfuse recommends propagating trace-level values to every span.
    # export() resolves this from the root record for historical JSONL files.
    resolved_trace_name = (trace_name or record.get("trace_name") or
                           record.get("name") or "rsi")
    attrs.append(_attribute("langfuse.trace.name", resolved_trace_name))
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ExportError("Trace metadata must be an object")
    for key, value in metadata.items():
        if key == "provenance" and record.get("provenance") is not None:
            continue
        if key == "run_id":
            attrs.append(_attribute("langfuse.session.id", str(value)))
        attrs.append(_attribute(f"langfuse.observation.metadata.{key}", value))
        attrs.append(_attribute(f"langfuse.trace.metadata.{key}", value))
    if record.get("provenance") is not None:
        attrs.append(_attribute("langfuse.observation.metadata.provenance",
                                record["provenance"]))
    if record.get("model") is not None:
        attrs.append(_attribute("langfuse.observation.model.name", str(record["model"])))
    usage = _normalized_usage(record.get("usage"))
    if usage:
        attrs.append(_attribute("langfuse.observation.usage_details", _json_value(usage)))
        if "input" in usage:
            attrs.append(_attribute("gen_ai.usage.input_tokens", usage["input"]))
        if "output" in usage:
            attrs.append(_attribute("gen_ai.usage.output_tokens", usage["output"]))
    span = {
        "traceId": record["trace_id"],
        "spanId": record["span_id"],
        "name": str(record.get("name") or "rsi"),
        "startTimeUnixNano": str(int(record["start_time_ns"])),
        "endTimeUnixNano": str(int(record["end_time_ns"])),
        "attributes": attrs,
    }
    for field, attr in (("environment", "langfuse.environment"),
                        ("tags", "langfuse.trace.tags"),
                        ("prompt_name", "langfuse.observation.prompt.name"),
                        ("prompt_version", "langfuse.observation.prompt.version")):
        if record.get(field) is not None:
            attrs.append(_attribute(attr, record[field]))
    if record.get("kind") == "generation":
        parameters = {key: metadata[key] for key in ("temperature", "max_tokens", "timeout_s")
                      if key in metadata}
        if parameters:
            attrs.append(_attribute("langfuse.observation.model.parameters", _json_value(parameters)))
    for key, value in record.get('experiment_attributes', {}).items():
        if key.startswith('langfuse.experiment.'):
            attrs.append(_attribute(key, value))
    parent = record.get("parent_span_id")
    if parent:
        if len(parent) != 16 or any(c not in "0123456789abcdef" for c in parent):
            raise ExportError("Invalid parent_span_id in trace record")
        span["parentSpanId"] = parent
    if record.get("links"):
        span["links"] = [{"traceId": link["trace_id"], "spanId": link["span_id"]}
                         for link in record["links"]]
    if record.get("status") == "error":
        attrs.append(_attribute("langfuse.observation.level", "ERROR"))
        if record.get("error_type"):
            attrs.append(_attribute("langfuse.observation.status_message", str(record["error_type"])))
        span["status"] = {"code": 2}
    return span


def iter_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise ExportError(f"Invalid JSON on input line {line_no}") from None
            if not isinstance(record, dict):
                raise ExportError(f"Input line {line_no} must contain a JSON object")
            yield record


def _payload(spans):
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def _trace_names(records):
    """Resolve a stable trace name from roots, including legacy JSONL rows."""
    names = {}
    for record in records:
        if not record.get("parent_span_id"):
            names.setdefault(record.get("trace_id"),
                             record.get("trace_name") or record.get("name") or "rsi")
    for record in records:
        names.setdefault(record.get("trace_id"),
                         record.get("trace_name") or record.get("name") or "rsi")
    return names


def _batches(records, max_bytes, trace_names=None):
    trace_names = trace_names or {}
    batch = []
    overhead = len(json.dumps(_payload([]), separators=(",", ":")).encode())
    size = overhead
    for record in records:
        span = record_to_span(record, trace_names.get(record.get("trace_id")))
        length = len(json.dumps(span, separators=(",", ":")).encode())
        if overhead + length > max_bytes:
            raise ExportError("A single OTLP span exceeds the configured batch size")
        addition = length + (1 if batch else 0)
        if batch and size + addition > max_bytes:
            yield batch
            batch = []
            size = overhead
            addition = length
        batch.append(span)
        size += addition
    if batch:
        yield batch


def _receipt(path):
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            raise ExportError("Existing receipt is unreadable") from None
    return None


def _write_receipt(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".langfuse-receipt-", suffix=".tmp",
                                     dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some filesystems do not support syncing directory descriptors.
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _rejected_spans(response):
    try:
        data = json.loads(response.decode("utf-8")) if response else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ExportError("Langfuse returned an invalid OTLP response") from None
    if not isinstance(data, dict):
        raise ExportError("Langfuse returned an invalid OTLP response")
    if "partialSuccess" in data and "partial_success" in data:
        raise ExportError("Langfuse returned an ambiguous OTLP partial-success response")
    partial = data.get("partialSuccess", data.get("partial_success", {}))
    if partial is None:
        return 0
    if not isinstance(partial, dict):
        raise ExportError("Langfuse returned an invalid OTLP partial-success response")
    if "rejectedSpans" in partial and "rejected_spans" in partial:
        raise ExportError("Langfuse returned an ambiguous rejected-span count")
    count = partial.get("rejectedSpans", partial.get("rejected_spans", 0))
    if isinstance(count, str) and count.isascii() and count.isdecimal():
        count = int(count)
    if (isinstance(count, bool) or not isinstance(count, int) or count < 0 or
            count > (2**63 - 1)):
        raise ExportError("Langfuse returned an invalid rejected-span count")
    return count


def _input_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_receipt(receipt, receipt_type, input_hash, endpoint=None,
                   public_fingerprint=None):
    if receipt is None:
        return
    if receipt.get("schema_version") != 1 or receipt.get("receipt_type") != receipt_type:
        raise ExportError("Receipt type or version mismatch; choose a new --output path")
    if receipt.get("input_sha256") != input_hash:
        raise ExportError("Receipt belongs to different input contents; choose a new --output path")
    if receipt_type == "delivery" and (
            receipt.get("endpoint") != endpoint or
            receipt.get("public_key_fingerprint") != public_fingerprint):
        raise ExportError("Receipt belongs to a different Langfuse project; choose a new --output path")
    accepted = receipt.get("accepted_ids", [])
    if not isinstance(accepted, list) or any(not isinstance(item, str) for item in accepted):
        raise ExportError("Receipt has invalid accepted span IDs")
    blocked = receipt.get("blocked_ids", [])
    if not isinstance(blocked, list) or any(not isinstance(item, str) for item in blocked):
        raise ExportError("Receipt has invalid blocked span IDs")
    rejected = receipt.get("rejected_batches", [])
    if not isinstance(rejected, list):
        raise ExportError("Receipt has invalid rejected batches")


def export(input_path, output_path, *, dry_run=False, allow_remote=False,
           max_batch_bytes=DEFAULT_BATCH_BYTES, env=None, opener=None):
    """Export records and write a compact receipt; returns the receipt object."""
    env = os.environ if env is None else env
    output_path = Path(output_path)
    receipt = _receipt(output_path)
    input_hash = _input_sha256(input_path)
    records = list(iter_records(input_path))
    batches = list(_batches(records, max_batch_bytes, _trace_names(records)))
    all_ids = [s["spanId"] for batch in batches for s in batch]
    if dry_run:
        _check_receipt(receipt, "dry_run", input_hash)
        if receipt is None:
            receipt = {"schema_version": 1, "receipt_type": "dry_run",
                       "accepted_ids": [], "rejected_batches": []}
        accepted = set(receipt.get("accepted_ids", []))
        blocked = set()
    else:
        public = env.get("LANGFUSE_PUBLIC_KEY")
        secret = env.get("LANGFUSE_SECRET_KEY")
        if not public or not secret:
            raise ExportError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")
        base = _base_url(env)
        if not _is_loopback(base) and not allow_remote:
            raise ExportError("External Langfuse destination requires --allow-remote")
        endpoint = base + ENDPOINT
        public_fingerprint = hashlib.sha256(public.encode()).hexdigest()
        _check_receipt(receipt, "delivery", input_hash, endpoint, public_fingerprint)
        if receipt is None:
            receipt = {"schema_version": 1, "receipt_type": "delivery",
                       "accepted_ids": [], "rejected_batches": [],
                       "input_sha256": input_hash, "endpoint": endpoint,
                       "public_key_fingerprint": public_fingerprint,
                       "delivery_semantics": "accepted_by_langfuse_ingestion_not_ui_visibility"}
        accepted = set(receipt.get("accepted_ids", []))
        # A partial-success response may mean some spans in a batch were
        # accepted, but OTLP supplies no accepted IDs. Keep the whole batch in
        # a blocked state so a receipt-based retry cannot duplicate it.
        blocked = set(receipt.get("blocked_ids", []))
        prior_rejections = receipt.get("rejected_batches", [])
        if isinstance(prior_rejections, list):
            for prior in prior_rejections:
                if isinstance(prior, dict) and isinstance(prior.get("span_ids"), list):
                    blocked.update(item for item in prior["span_ids"] if isinstance(item, str))
        blocked.update(item for item in receipt.get("blocked_ids", [])
                       if isinstance(item, str))
    pending = [[s for s in batch if s["spanId"] not in accepted and
                s["spanId"] not in blocked] for batch in batches]
    pending = [batch for batch in pending if batch]
    receipt.update({"schema_version": 1, "input_records": len(records),
                    "span_ids": all_ids, "dry_run": bool(dry_run),
                    "batches": len(batches), "accepted_ids": sorted(accepted),
                    "input_sha256": input_hash})
    if dry_run:
        receipt["pending_ids"] = [s["spanId"] for batch in pending for s in batch]
        receipt["rejected_batches"] = []
        _write_receipt(output_path, receipt)
        return receipt
    if not pending:
        receipt["pending_ids"] = []
        receipt["blocked_ids"] = sorted(blocked)
        receipt["delivery_status"] = "blocked_partial_success" if blocked else "accepted"
        _write_receipt(output_path, receipt)
        return receipt
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode("ascii")
    send = opener or _safe_urlopen
    rejected_batches = list(receipt.get("rejected_batches", []))
    rejected_total = 0
    accepted_now = []
    for index, batch in enumerate(pending):
        body = json.dumps(_payload(batch), ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        request = Request(endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        })
        try:
            response_obj = send(request, timeout=30)
            with response_obj as response:
                status, response_body = response.status, response.read()
        except HTTPError as exc:
            exc.close()
            raise ExportError(f"Langfuse rejected export with HTTP {exc.code}") from None
        except (URLError, OSError, TimeoutError):
            raise ExportError("Langfuse export request failed") from None
        if status < 200 or status >= 300:
            raise ExportError(f"Langfuse rejected export with HTTP {status}")
        rejected = _rejected_spans(response_body)
        if rejected > len(batch):
            raise ExportError("Langfuse returned an impossible rejected-span count")
        if rejected:
            rejected_total += rejected
            batch_ids = [s["spanId"] for s in batch]
            blocked.update(batch_ids)
            rejected_batches.append({"batch_index": index, "span_ids": batch_ids,
                                     "rejected_spans": rejected,
                                     "state": "blocked_partial_success"})
            receipt["accepted_ids"] = sorted(accepted)
            receipt["blocked_ids"] = sorted(blocked)
            receipt["pending_ids"] = [s["spanId"] for b in pending for s in b
                                       if s["spanId"] not in accepted and s["spanId"] not in blocked]
            receipt["rejected_batches"] = rejected_batches
            receipt["rejected_spans"] = sum(
                item.get("rejected_spans", 0) for item in rejected_batches
                if isinstance(item, dict) and isinstance(item.get("rejected_spans", 0), int)
                and not isinstance(item.get("rejected_spans", 0), bool))
            receipt["delivery_status"] = "blocked_partial_success"
            _write_receipt(output_path, receipt)
        else:
            accepted_now.extend(s["spanId"] for s in batch)
            accepted.update(s["spanId"] for s in batch)
            receipt["accepted_ids"] = sorted(accepted)
            receipt["blocked_ids"] = sorted(blocked)
            receipt["pending_ids"] = [s["spanId"] for b in pending for s in b
                                       if s["spanId"] not in accepted and s["spanId"] not in blocked]
            receipt["rejected_batches"] = rejected_batches
            receipt["delivery_status"] = "blocked_partial_success" if blocked else "accepted"
            _write_receipt(output_path, receipt)
    receipt["accepted_ids"] = sorted(accepted)
    receipt["accepted_this_run"] = accepted_now
    receipt["rejected_batches"] = rejected_batches
    receipt["rejected_spans"] = sum(
        item.get("rejected_spans", 0) for item in rejected_batches
        if isinstance(item, dict) and isinstance(item.get("rejected_spans", 0), int)
        and not isinstance(item.get("rejected_spans", 0), bool))
    receipt["blocked_ids"] = sorted(blocked)
    receipt["pending_ids"] = [s["spanId"] for b in pending for s in b
                               if s["spanId"] not in accepted and s["spanId"] not in blocked]
    receipt["delivery_status"] = "blocked_partial_success" if blocked else "accepted"
    _write_receipt(output_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON retry receipt")
    parser.add_argument("--dry-run", action="store_true", help="validate and summarize without credentials or network")
    parser.add_argument("--allow-remote", action="store_true", help="allow sending to a non-loopback endpoint")
    parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES)
    args = parser.parse_args(argv)
    if args.batch_bytes < 256:
        parser.error("--batch-bytes must be at least 256")
    try:
        receipt = export(args.input, args.output, dry_run=args.dry_run,
                         allow_remote=args.allow_remote,
                         max_batch_bytes=args.batch_bytes)
    except (ExportError, OSError) as exc:
        print(f"Langfuse export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: receipt.get(k) for k in (
        "input_records", "batches", "span_ids", "accepted_ids", "pending_ids",
        "blocked_ids", "rejected_spans", "delivery_status", "dry_run")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
