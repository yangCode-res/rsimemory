#!/usr/bin/env python3
"""Fail-closed preflight for the v0.3 object-state policy worker.

This checks the isolated worker with synthetic data, then validates only the
patient-ID column of the Chinese dataset. It never reads question/answer text,
starts a model, or prints patient identifiers.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPLIT_SEED = 20261013
SYNTHETIC_ID = "preflight-v03-synthetic-1"
SYNTHETIC_TEXT = "synthetic preflight payload"

# Deliberately ordinary Python: imports, class identity, lambda, set, datetime,
# numpy, and networkx must all survive the worker's opaque state snapshots.
POLICY_SOURCE = r'''
import datetime
import networkx as nx
import numpy as np


class Memory:
    def __init__(self):
        self.ids = []
        self.seen = set()
        self.stamp = datetime.datetime(2020, 1, 1)
        self.selector = lambda value: value + 1
        self.count = 0
        self.vector = np.array([1.0, 2.0])
        self.graph = nx.path_graph(2)

    def add(self, session):
        self.ids.append(session["id"])
        self.seen.add(session["id"])
        self.stamp += datetime.timedelta(days=1)
        self.count = self.selector(self.count)
        self.graph.add_edge(len(self.graph) - 1, len(self.graph))


def initialize(config):
    return Memory()


def ingest(state, session):
    if not isinstance(state, Memory):
        raise TypeError("state class identity changed")
    state.add(session)
    return state


def search(state, query, filters):
    if not isinstance(state, Memory):
        raise TypeError("state class identity changed")
    if not isinstance(state.stamp, datetime.datetime):
        raise TypeError("datetime state was not preserved")
    if not isinstance(state.vector, np.ndarray) or not nx.is_tree(state.graph):
        raise TypeError("numeric or graph state was not preserved")
    if state.count != 1 or not state.seen:
        raise TypeError("mutable state was not preserved")
    # Deliberately mutate query-local state. The worker must discard these
    # changes after each search; the next search should see the original state.
    state.ids.reverse()
    chosen = list(state.ids)
    state.ids.clear()
    state.seen.clear()
    return chosen
'''


class PreflightFailure(RuntimeError):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


def _validate_worker() -> dict[str, str]:
    from rsi.policy import GeneratedPolicy

    chunk = {
        "id": SYNTHETIC_ID,
        "persona_id": 0,
        "session_id": 1,
        "text": SYNTHETIC_TEXT,
    }
    policy = GeneratedPolicy(POLICY_SOURCE, {"top_k": 1}, timeout_s=60.0)
    policy.ingest(chunk)
    snapshot = policy.snapshot()

    first = policy.search("synthetic probe", {"persona_id": 0, "checkpoint": 1})
    second = policy.search("synthetic probe", {"persona_id": 0, "checkpoint": 1})
    if first != second:
        raise PreflightFailure("worker_mutation_isolation_failed")
    expected = [{"id": SYNTHETIC_ID, "text": SYNTHETIC_TEXT, "session_id": 1}]
    if first != expected:
        raise PreflightFailure("worker_evidence_contract_failed")

    restored = GeneratedPolicy(POLICY_SOURCE, {"top_k": 1}, timeout_s=60.0)
    restored.restore(snapshot)
    after_restore = restored.search("synthetic probe", {"persona_id": 0, "checkpoint": 1})
    if after_restore != expected:
        raise PreflightFailure("worker_snapshot_restore_failed")
    return {
        "object_state": "ok",
        "repeat_search_mutation_isolation": "ok",
        "snapshot_restore": "ok",
    }


def _validate_dataset(dataset_root: str | Path) -> dict[str, Any]:
    # Read only the requested metadata column; no dialogue/query content or
    # answer annotations are loaded by this check.
    import pandas as pd

    from rsi.data import SPLIT_VERSION, patient_splits

    path = Path(dataset_root) / "data" / "zh" / "dialogues.parquet"
    frame = pd.read_parquet(path, columns=["persona_id"])
    patients = sorted({int(value) for value in frame["persona_id"].dropna().unique()})
    if len(patients) != 20:
        raise PreflightFailure("patient_count_mismatch")
    splits = patient_splits(patients, seed=SPLIT_SEED)
    counts = {name: len(splits[name]) for name in ("train", "validation", "test")}
    if counts != {"train": 8, "validation": 4, "test": 8}:
        raise PreflightFailure("patient_split_count_mismatch")
    all_ids = [item for group in splits.values() for item in group]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(patients):
        raise PreflightFailure("patient_split_partition_invalid")
    return {
        "split_version": SPLIT_VERSION,
        "seed": SPLIT_SEED,
        "patient_count": len(patients),
        "split_counts": counts,
    }


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def _emit_error(category: str, error_type: str | None = None) -> int:
    body = {"status": "error", "category": category}
    if error_type:
        body["error_type"] = error_type
    print(json.dumps(body, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset root containing data/zh/dialogues.parquet")
    args = parser.parse_args(argv)

    from rsi import sandbox

    if not sandbox.isolation_available():
        return _emit_error("isolation_unavailable")

    try:
        worker = _validate_worker()
    except PreflightFailure as exc:
        return _emit_error(exc.category)
    except Exception as exc:
        return _emit_error("worker_validation_failed", type(exc).__name__)

    try:
        dataset = _validate_dataset(args.dataset)
    except PreflightFailure as exc:
        return _emit_error(exc.category)
    except Exception as exc:
        return _emit_error("dataset_validation_failed", type(exc).__name__)

    try:
        import cloudpickle

        result = {
            "status": "ok",
            "kernel": platform.release(),
            "python": platform.python_version(),
            "landlock_abi": sandbox._landlock_abi(),
            "cloudpickle_version": cloudpickle.__version__,
            "numpy_version": _version("numpy"),
            "networkx_version": _version("networkx"),
            **dataset,
            "worker": worker,
        }
    except Exception as exc:
        return _emit_error("runtime_metadata_unavailable", type(exc).__name__)

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
