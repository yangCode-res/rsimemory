"""Small OpenAI-compatible chat client with per-role call accounting."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any


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
                 json_mode: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
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
            self.calls.append({"role": role,
                               "usage": payload.get("usage") or {} if isinstance(payload, dict) else {},
                               "latency_s": time.perf_counter() - started,
                               "error": f"{type(exc).__name__}: {exc}"})
            raise
        self.calls.append({"role": role, "usage": result["usage"],
                           "latency_s": result["latency_s"]})
        return result
