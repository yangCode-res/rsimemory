"""Memory baselines and host wrapper for ordinary Python object-state policies."""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .sandbox import ContractError, SandboxError, run_candidate


def _grams(text: str) -> list[str]:
    """Return character bigrams; works for Chinese without a tokenizer."""
    normalized = re.sub(r"\s+", " ", str(text).casefold()).strip()
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return [normalized[i : i + 2] for i in range(len(normalized) - 1)]


class PlainPolicy:
    """Lexical BM25 policy with optional, explicitly local sentence embeddings."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = dict(config or {})
        self.top_k = max(1, int(config.get("top_k", 5)))
        self.k1 = float(config.get("bm25_k1", 1.5))
        self.b = float(config.get("bm25_b", 0.75))
        self.bm25_weight = float(config.get("bm25_weight", 1.0))
        self.semantic_model_path = config.get("semantic_model_path")
        self.semantic_weight = float(config.get("semantic_weight", 0.0))
        self._model = None
        # Per-source-chunk, normalized vectors are immutable tuples shared by
        # host snapshots; they are an accelerator, not an alternate memory store.
        self._embedding_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._chunks: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._visible_ids: set[str] = set()
        self.cost = {"ingest_calls": 0, "search_calls": 0, "ingest_seconds": 0.0,
                     "search_seconds": 0.0, "storage_bytes": 0,
                     "peak_storage_bytes": 0, "embedding_cache_bytes": 0}
        if self.semantic_weight:
            # Explicit local path only. SentenceTransformer's network fallback is
            # disabled. A configured semantic branch must be available; it never
            # silently changes the requested policy to lexical-only.
            if not self.semantic_model_path:
                raise ValueError("semantic_weight requires semantic_model_path")
            local_path = Path(self.semantic_model_path).expanduser()
            if not local_path.is_dir():
                raise FileNotFoundError(f"local sentence-transformer model directory not found: {local_path}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(str(local_path), device="cpu", local_files_only=True)
            except Exception as exc:
                raise RuntimeError(f"could not load local sentence-transformer model: {local_path}") from exc

    def ingest(self, chunk: dict[str, Any]) -> None:
        start = time.perf_counter()
        item = copy.deepcopy(chunk)
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError("chunk must contain id and text")
        item["id"] = str(item["id"])
        if item["id"] in self._by_id:
            raise ValueError("duplicate chunk id")
        item["text"] = str(item["text"])
        item.setdefault("persona_id", None)
        item.setdefault("session_id", None)
        self._chunks.append(item)
        self._by_id[item["id"]] = item
        self.cost["ingest_calls"] += 1
        self.cost["ingest_seconds"] += time.perf_counter() - start
        self._update_storage_cost()

    def _update_storage_cost(self) -> None:
        source_bytes = len(json.dumps(self._chunks, ensure_ascii=False).encode("utf-8"))
        cache_bytes = sum(len(json.dumps(vector).encode("utf-8"))
                          for vector in self._embedding_cache.values())
        self.cost["embedding_cache_bytes"] = cache_bytes
        self.cost["storage_bytes"] = source_bytes + cache_bytes
        self.cost["peak_storage_bytes"] = max(self.cost["peak_storage_bytes"], self.cost["storage_bytes"])

    def __deepcopy__(self, memo):
        """Copy per-question records/cost while sharing the heavyweight encoder/cache."""
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in {"_model", "_embedding_cache"}:
                setattr(clone, name, value)
            else:
                setattr(clone, name, copy.deepcopy(value, memo))
        return clone

    def _eligible(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        result = []
        for chunk in self._chunks:
            sid = chunk.get("session_id")
            if filters.get("persona_id") is not None and chunk.get("persona_id") != filters["persona_id"]:
                continue
            if filters.get("session_id") is not None and sid != filters["session_id"]:
                continue
            # checkpoint is the maximum visible session. It applies even when a
            # candidate has retained or synthesized state from future sessions.
            checkpoint = filters.get("checkpoint", filters.get("max_session_id"))
            if checkpoint is not None and (sid is None or sid > checkpoint):
                continue
            if filters.get("min_session_id") is not None and (sid is None or sid < filters["min_session_id"]):
                continue
            result.append(chunk)
        return result

    def _lexical_scores(self, query: str, chunks: list[dict[str, Any]]) -> dict[str, float]:
        qterms = _grams(query)
        if not qterms:
            return {c["id"]: 0.0 for c in chunks}
        docs = [_grams(c["text"]) for c in chunks]
        n = len(docs)
        avgdl = sum(map(len, docs)) / n if n else 0.0
        df = Counter(term for doc in docs for term in set(doc))
        qtf = Counter(qterms)
        scores = {}
        for chunk, doc in zip(chunks, docs):
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term, query_count in qtf.items():
                freq = tf[term]
                if not freq:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl) if avgdl else freq + self.k1
                score += idf * (freq * (self.k1 + 1) / denom) * min(query_count, 2)
            scores[chunk["id"]] = score
        return scores

    def search(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        start = time.perf_counter()
        self._visible_ids = set()
        filters = dict(filters or {})
        eligible = self._eligible(filters)
        scores = {key: value * self.bm25_weight
                  for key, value in self._lexical_scores(str(query), eligible).items()}
        if self._model is not None and self.semantic_weight and eligible:
            try:
                missing = [c for c in eligible if (c["id"], c["text"]) not in self._embedding_cache]
                encode_inputs = [str(query)] + [c["text"] for c in missing]
                vectors = self._model.encode(encode_inputs, batch_size=32, normalize_embeddings=True)
                query_vector = tuple(float(x) for x in vectors[0])
                for chunk, vector in zip(missing, vectors[1:]):
                    self._embedding_cache[(chunk["id"], chunk["text"])] = tuple(float(x) for x in vector)
                semantic = {}
                for chunk in eligible:
                    vector = self._embedding_cache[(chunk["id"], chunk["text"])]
                    semantic[chunk["id"]] = sum(a * b for a, b in zip(query_vector, vector))
                scores = {key: val + self.semantic_weight * semantic[key] for key, val in scores.items()}
                self._update_storage_cost()
            except Exception as exc:
                raise RuntimeError("local semantic encoder failed during search") from exc
        def session_order(chunk):
            value = chunk.get("session_id")
            return (0, value) if isinstance(value, (int, float)) else (1, str(value))

        ranked = sorted(eligible, key=lambda c: (scores.get(c["id"], 0.0), session_order(c), c["id"]), reverse=True)
        chosen = ranked[: self.top_k]
        self._visible_ids = {c["id"] for c in chosen}
        self.cost["search_calls"] += 1
        self.cost["search_seconds"] += time.perf_counter() - start
        return [{"id": c["id"], "text": c["text"], "session_id": c.get("session_id")} for c in chosen]

    def read(self, ids: list[str]) -> list[dict[str, Any]]:
        requested = [str(i) for i in ids]
        if any(i not in self._visible_ids for i in requested):
            raise PermissionError("read is restricted to IDs visible in the latest search")
        return [{"id": self._by_id[i]["id"], "text": self._by_id[i]["text"],
                 "session_id": self._by_id[i].get("session_id")} for i in requested]

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated, deep-copied snapshot for a single question."""
        return copy.deepcopy({"chunks": self._chunks, "cost": self.cost})

    def restore(self, snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("chunks"), list):
            raise ValueError("invalid PlainPolicy snapshot")
        self._chunks = copy.deepcopy(snapshot["chunks"])
        self._by_id = {str(c["id"]): c for c in self._chunks}
        self.cost = copy.deepcopy(snapshot.get("cost", self.cost))
        self._visible_ids = set()


class GeneratedPolicy:
    """Host wrapper for ordinary Python candidates and opaque object snapshots.

    The host retains original chunks and validates every candidate ID against
    the eligible, visible registry before returning evidence.
    """

    serializable = True  # Host snapshot capability, not a promise about every Python object.

    def __init__(self, source: str, config: dict[str, Any] | None = None,
                 *, timeout_s: float = 30.0, memory_bytes: int = 2 * 1024 * 1024 * 1024):
        self.source = source
        self.config = copy.deepcopy(config or {})
        self.timeout_s = timeout_s
        self.memory_bytes = memory_bytes
        self.top_k = max(1, int(self.config.get("top_k", 5)))
        self._state = None
        self._registry: dict[str, dict[str, Any]] = {}
        self._visible_ids: set[str] = set()
        self.cost = {"sandbox_calls": 0, "sandbox_seconds": 0.0,
                     "initialize_calls": 0, "ingest_calls": 0, "search_calls": 0,
                     "ingest_seconds": 0.0, "search_seconds": 0.0,
                     "source_storage_bytes": 0, "candidate_state_bytes": 0,
                     "storage_bytes": 0, "peak_storage_bytes": 0}
        result = self._call("initialize", {"config": self.config})
        self._state = result["state"]
        self.cost["source_storage_bytes"] = 0
        self.cost["candidate_state_bytes"] = self._snapshot_size(self._state)
        self.cost["storage_bytes"] = self.cost["source_storage_bytes"] + self.cost["candidate_state_bytes"]
        self.cost["peak_storage_bytes"] = self.cost["storage_bytes"]

    def _call(self, operation: str, payload: dict[str, Any]):
        start = time.perf_counter()
        error = None
        try:
            return run_candidate(self.source, operation, payload, timeout_s=self.timeout_s,
                                 memory_bytes=self.memory_bytes)
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.cost["sandbox_calls"] += 1
            self.cost["sandbox_seconds"] += elapsed
            if operation == "initialize":
                self.cost["initialize_calls"] += 1
            elif operation in {"ingest", "search"}:
                self.cost[f"{operation}_calls"] += 1
                self.cost[f"{operation}_seconds"] += elapsed
            if error is not None: error.policy_cost = copy.deepcopy(self.cost)

    @staticmethod
    def _json_size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _snapshot_size(self, snapshot):
        if not isinstance(snapshot,dict) or snapshot.get("serializer") != "cloudpickle-base64-v1":
            raise ContractError("invalid opaque state snapshot")
        blob = base64.b64decode(snapshot["data"], validate=True)
        if snapshot.get("sha256") != hashlib.sha256(blob).hexdigest():
            raise ContractError("state snapshot checksum mismatch")
        if snapshot.get("source_sha256") != hashlib.sha256(self.source.encode()).hexdigest():
            raise ContractError("state snapshot belongs to a different policy version")
        return len(blob)  # No pickle deserialization ever occurs in the host.

    def ingest(self, chunk: dict[str, Any]) -> None:
        original = copy.deepcopy(chunk)
        if "id" not in original or "text" not in original:
            raise ValueError("chunk must contain id and text")
        cid = str(original["id"])
        if cid in self._registry:
            raise ValueError("duplicate chunk id")
        original["id"] = cid
        original["text"] = str(original["text"])
        original.setdefault("persona_id", None)
        original.setdefault("session_id", None)
        result = self._call("ingest", {"state": self._state, "session": copy.deepcopy(original)})
        self._state = result["state"]
        self._registry[cid] = original
        self.cost["source_storage_bytes"] = self._json_size(list(self._registry.values()))
        self.cost["candidate_state_bytes"] = self._snapshot_size(self._state)
        self.cost["storage_bytes"] = self.cost["source_storage_bytes"] + self.cost["candidate_state_bytes"]
        self.cost["peak_storage_bytes"] = max(self.cost["peak_storage_bytes"], self.cost["storage_bytes"])

    @staticmethod
    def _matches(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        sid = chunk.get("session_id")
        if filters.get("persona_id") is not None and chunk.get("persona_id") != filters["persona_id"]:
            return False
        if filters.get("session_id") is not None and sid != filters["session_id"]:
            return False
        checkpoint = filters.get("checkpoint", filters.get("max_session_id"))
        if checkpoint is not None and (sid is None or sid > checkpoint):
            return False
        if filters.get("min_session_id") is not None and (sid is None or sid < filters["min_session_id"]):
            return False
        return True

    def search(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._visible_ids = set()
        filters = copy.deepcopy(filters or {})
        eligible_ids = {cid for cid, c in self._registry.items() if self._matches(c, filters)}
        result = self._call("search", {"state": self._state, "query": str(query), "filters": filters})
        candidate_ids = result.get("ids")
        if not isinstance(candidate_ids, list):
            raise ContractError("candidate search must return a list of IDs")
        if any(not isinstance(candidate_id, (str, int, float)) for candidate_id in candidate_ids):
            raise ContractError("candidate search returned a non-scalar ID")
        allowed = []
        for candidate_id in candidate_ids:
            cid = str(candidate_id)
            if cid not in eligible_ids:
                raise ContractError("candidate search returned an unknown or out-of-scope ID")
            if cid in allowed:
                raise ContractError("candidate search returned a duplicate ID")
            allowed.append(cid)
        allowed = allowed[: self.top_k]
        self._visible_ids = set(allowed)
        return [{"id": cid, "text": self._registry[cid]["text"],
                 "session_id": self._registry[cid].get("session_id")} for cid in allowed]

    def read(self, ids: list[str]) -> list[dict[str, Any]]:
        requested = [str(i) for i in ids]
        if any(i not in self._visible_ids for i in requested):
            raise PermissionError("read is restricted to IDs visible in the latest search")
        return [{"id": cid, "text": self._registry[cid]["text"],
                 "session_id": self._registry[cid].get("session_id")} for cid in requested]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy({"state": self._state, "registry": self._registry, "cost": self.cost})

    def restore(self, snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict) or "state" not in snapshot or not isinstance(snapshot.get("registry"), dict):
            raise ValueError("invalid GeneratedPolicy snapshot")
        self._snapshot_size(snapshot["state"])
        self._state = copy.deepcopy(snapshot["state"])
        self._registry = copy.deepcopy(snapshot["registry"])
        self.cost = copy.deepcopy(snapshot.get("cost", self.cost))
        self._visible_ids = set()
