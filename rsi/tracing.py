"""Small, local-only trace recorder for RSI.

Recording is deliberately separate from export: this module only appends
completed spans to a private JSONL spool and never opens a network connection.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import hashlib
import os
from pathlib import Path
import time
import uuid
import warnings


_ACTIVE: ContextVar["_State | None"] = ContextVar("rsi_trace_state", default=None)
_write_failures = 0


def _id(nbytes: int) -> str:
    return uuid.uuid4().hex[: nbytes * 2]


def append_record(path, record) -> bool:
    """Append one record with owner-only permissions; return False on failure.

    Errors are intentionally reduced to a counter: a filesystem exception can
    contain path or data details that should not enter application logs.
    """
    global _write_failures
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False) + "\n"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except Exception:
        _write_failures += 1
        try:
            warnings.warn(f"RSI trace spool write failed (count={_write_failures})",
                          RuntimeWarning, stacklevel=2)
        except Exception:
            # Warnings can be promoted to exceptions by caller configuration.
            # A diagnostic must never replace the application's exception.
            pass
        return False


def write_failure_count() -> int:
    """Number of fail-open spool/serialization errors in this process."""
    return _write_failures


@dataclass
class _State:
    path: Path
    trace_id: str
    session_id: str | None
    trace_metadata: dict[str, object] = field(default_factory=dict)
    trace_name: str = "rsi-evolution-round"
    stack: tuple["Observation", ...] = ()


class Observation:
    """Mutable handle yielded by :func:`trace_span`."""
    def __init__(self, state: _State | None = None, *, name="", kind="span",
                 parent_span_id=None, input=None, metadata=None, links=None,
                 root=False):
        self._state = state
        self._root = root
        self.name = name
        self.kind = kind
        self.span_id = _id(8) if state is not None else None
        self.parent_span_id = parent_span_id
        self.start_time_ns = time.time_ns() if state is not None else None
        self.input = input
        self.output = None
        self.metadata = dict(metadata or {})
        self.model = None
        self.usage = None
        self.status = "ok"
        self.error_type = None
        self.links = list(links or [])
        self._ended = False

    def update(self, *, output=None, metadata=None, usage=None, model=None,
               status=None, input=None):
        """Update captured fields. Mapping metadata is merged into prior data."""
        if output is not None:
            self.output = output
        if input is not None:
            self.input = input
        if metadata:
            self.metadata.update(metadata)
        if usage is not None:
            self.usage = usage
        if model is not None:
            self.model = model
        if status is not None:
            self.status = str(status)
        return self

    def _finish(self, exc_type=None):
        if self._ended or self._state is None:
            return
        self._ended = True
        end = time.time_ns()
        is_error = exc_type is not None or self.status == "error"
        if exc_type is not None:
            self.status, self.error_type = "error", exc_type.__name__
        record = {
            "schema_version": 1,
            "trace_id": self._state.trace_id,
            "span_id": self.span_id,
            "trace_name": self._state.trace_name,
            "name": self.name,
            "kind": self.kind,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": end,
            "input": self.input,
            "output": self.output,
            "metadata": {**self._state.trace_metadata, **self.metadata},
            "model": self.model,
            "usage": self.usage,
            "status": "error" if is_error else "ok",
            "error_type": self.error_type,
            "provenance": "live",
        }
        if self.parent_span_id is not None:
            record["parent_span_id"] = self.parent_span_id
        if self.links:
            record["links"] = self.links
        append_record(self._state.path, record)


class _NoopObservation(Observation):
    def __init__(self):
        super().__init__()


def trace_run(path, run_id, metadata=None):
    """Record one evolution run with a root ``rsi-evolution-round`` span."""
    from contextlib import contextmanager

    @contextmanager
    def manager():
        inherited = _ACTIVE.get()
        attrs = dict(metadata or {})
        attrs.setdefault("run_id", str(run_id))
        state = _State(Path(path), _id(16), str(run_id), attrs,
                       trace_name="rsi-evolution-round")
        token = _ACTIVE.set(state)
        root = Observation(state, name="rsi-evolution-round", kind="span", root=True)
        try:
            state.stack = (root,)
            yield root
        except BaseException as exc:
            root._finish(type(exc))
            raise
        else:
            root._finish()
        finally:
            _ACTIVE.reset(token)
    return manager()


def trace_span(name, kind="span", input=None, metadata=None, new_trace=False):
    """Record a nested span, or a linked independent trace when requested.

    Outside an active ``trace_run`` this returns a context manager that yields
    a no-op handle. ``new_trace`` roots the observation in a fresh trace and
    records a link to the active span while preserving the inherited session.
    """
    from contextlib import contextmanager

    @contextmanager
    def manager():
        parent_state = _ACTIVE.get()
        if parent_state is None:
            yield _NoopObservation()
            return
        links = []
        if new_trace:
            parent = parent_state.stack[-1] if parent_state.stack else None
            if parent is not None:
                links.append({"trace_id": parent_state.trace_id,
                              "span_id": parent.span_id})
            state = _State(parent_state.path, _id(16), parent_state.session_id,
                           _inherited_metadata(parent_state), trace_name=str(name))
            parent_span = None
        else:
            state = parent_state
            parent_span = state.stack[-1].span_id if state.stack else None
        token = _ACTIVE.set(state) if new_trace else None
        inherited = (_inherited_metadata(state) if not new_trace
                     else dict(state.trace_metadata))
        inherited.update(metadata or {})
        obs = Observation(state, name=str(name), kind=str(kind),
                          parent_span_id=parent_span, input=input,
                          metadata=inherited, links=links)
        old_stack = state.stack
        state.stack = old_stack + (obs,)
        try:
            yield obs
        except BaseException as exc:
            obs._finish(type(exc))
            raise
        else:
            obs._finish()
        finally:
            state.stack = old_stack
            if token is not None:
                _ACTIVE.reset(token)
    return manager()


def _inherited_metadata(state: _State) -> dict[str, object]:
    """Flatten trace and active span metadata for child observations."""
    result = dict(state.trace_metadata)
    for observation in state.stack:
        result.update(observation.metadata)
    return result



def patient_label(persona_id, data_revision=None):
    """Stable dataset-scoped alias; never expose the original patient key."""
    identity=json.dumps([data_revision,str(persona_id)],sort_keys=True,ensure_ascii=False)
    return 'patient-' + hashlib.sha256(identity.encode()).hexdigest()[:10]
