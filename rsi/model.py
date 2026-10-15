"""Small OpenAI-compatible chat client with per-role call accounting."""
from __future__ import annotations

import json
import errno
import http.client
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any
from .tracing import trace_span


class ModelServiceUnavailable(RuntimeError):
    """The configured model endpoint could not be reached reliably."""


_NETWORK_ERRNOS = {errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED,
                   errno.ETIMEDOUT, errno.ENETUNREACH, errno.EHOSTUNREACH,
                   errno.EPIPE, errno.EIO}


def _transport_error(exc: Exception) -> str | None:
    """Return a safe exception class for infrastructure failures only."""
    if isinstance(exc, urllib.error.HTTPError):
        return None
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return type(reason).__name__ if isinstance(reason, BaseException) else type(exc).__name__
    if isinstance(exc, (http.client.RemoteDisconnected, ConnectionError,
                        TimeoutError, socket.timeout)):
        return type(exc).__name__
    if isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS:
        return type(exc).__name__
    return None


class ModelClient:
    """Call a chat-completions endpoint and return normalized call metadata.

    Fake clients can be injected into :class:`rsi.actor.Actor` and
    :func:`rsi.evaluation.evaluate`; no dataset-backed offline answer is
    produced by this class.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:7860/v1", model: str = "",
                 api_key: str | None = None, timeout_s: float = 180.0,
                 temperature: float = 0.0, max_tokens: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("RSI_API_KEY")
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], role: str = "actor",
                 json_mode: bool = False, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        with trace_span('llm.'+role, kind='generation', input={'messages': messages},
                        metadata={'role': role, 'temperature': self.temperature,
                                  'max_tokens': self.max_tokens}) as observation:
            return self._complete(messages, role, json_mode, json_schema, observation)

    def _complete(self, messages, role, json_mode, json_schema, observation):
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema":
                                       {"name": role, "schema": json_schema, "strict": True}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        payload: Any = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
            elapsed = time.perf_counter() - started
            # Retain the actual response before protocol/length validation, so
            # failed code generation remains inspectable in the private trace.
            observation.update(output=payload, model=self.model,
                               usage=payload.get('usage', {}) if isinstance(payload, dict) else {},
                               metadata={'latency_s': elapsed})
            if not isinstance(payload, dict):
                raise ValueError("chat completion response must be a JSON object")
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("chat completion response has no valid choices")
            choice = choices[0]
            if choice.get("finish_reason") == "length":
                raise ValueError("chat completion was truncated (finish_reason=length)")
            message = choice.get("message", {})
            if not isinstance(message, dict):
                raise ValueError("chat completion choice has no valid message")
            content = message.get("content")
            if content is None:
                raise ValueError("chat completion returned null content")
            # Some OpenAI-compatible endpoints return multipart content blocks.
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content
                                  if isinstance(part, dict))
            result = {"text": str(content), "usage": payload.get("usage") or {},
                      "latency_s": elapsed, "role": role}
        except Exception as exc:
            transport_class = _transport_error(exc)
            if transport_class:
                safe_error = f"Model service unavailable ({transport_class})"
                self.calls.append({"role": role,
                                   "usage": payload.get("usage") or {} if isinstance(payload, dict) else {},
                                   "latency_s": time.perf_counter() - started,
                                   "error": safe_error})
                raise ModelServiceUnavailable(safe_error) from None
            self.calls.append({"role": role,
                               "usage": payload.get("usage") or {} if isinstance(payload, dict) else {},
                               "latency_s": time.perf_counter() - started,
                               "error": f"{type(exc).__name__}: {exc}"})
            raise
        self.calls.append({"role": role, "usage": result["usage"],
                           "latency_s": result["latency_s"]})
        return result
