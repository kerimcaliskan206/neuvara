"""
Ollama provider.

Talks to a running Ollama server over its HTTP API.  Connection details
(base URL, model, timeout, keep_alive) come from ``OllamaConfig`` so the
provider is fully configurable.

Endpoints used:

  * ``POST /api/chat``  — chat completion
  * ``GET  /api/tags``  — list available models (also used for health probe)
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.modules.ai.config import OllamaConfig
from app.modules.ai.providers.base import (
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    ProviderError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(ChatProvider):
    """Async chat-completion provider backed by a local Ollama server."""

    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    # ── ChatProvider ─────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": _build_options(temperature, top_p, max_tokens),
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama request timed out after {self.config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama HTTP error: {exc}") from exc

        data = response.json()
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise ProviderError("Ollama returned an empty response.")

        return ChatCompletion(
            content=content,
            model=data.get("model", payload["model"]),
            finish_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=_safe_add(
                data.get("prompt_eval_count"), data.get("eval_count")
            ),
            raw=data,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens one by one using Ollama's streaming API."""
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "options": _build_options(temperature, top_p, max_tokens),
        }

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (data.get("message") or {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama streaming timed out after {self.config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama HTTP error: {exc}") from exc

    async def health(self) -> dict:
        """Probe ``/api/tags`` to verify the server is reachable and the model is loaded."""
        result: dict[str, Any] = {
            "ok": False,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "model_loaded": False,
            "available_models": [],
            "error": None,
        }
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            result["error"] = f"timeout after {self.config.timeout_seconds:.1f}s"
            logger.warning("Ollama health: %s", result["error"])
            return result
        except httpx.HTTPError as exc:
            result["error"] = str(exc)
            logger.warning("Ollama health failed: %s", exc)
            return result

        data = response.json()
        tags = [tag.get("name") for tag in data.get("models", []) if tag.get("name")]
        result["available_models"] = tags
        result["ok"] = True
        result["model_loaded"] = any(
            t == self.config.model or t.startswith(self.config.model + ":")
            for t in tags
        )
        return result

    async def close(self) -> None:
        await self._client.aclose()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_options(
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Map provider-neutral knobs onto Ollama's option keys."""
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    return options


def _safe_add(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)