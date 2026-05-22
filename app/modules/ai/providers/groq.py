"""
Groq provider.

Calls the Groq chat-completions API (OpenAI-compatible wire format).
Uses httpx directly — no groq SDK required.

Endpoint:  POST https://api.groq.com/openai/v1/chat/completions
Health:    GET  https://api.groq.com/openai/v1/models
Auth:      Authorization: Bearer {api_key}
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from app.modules.ai.config import GroqConfig
from app.modules.ai.providers.base import (
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    ProviderError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(ChatProvider):
    """Async chat-completion provider backed by the Groq API."""

    def __init__(self, config: GroqConfig) -> None:
        self.config = config
        if not config.api_key:
            raise ProviderError(
                "GROQ_API_KEY is not set — AI assistant cannot start. "
                "Add GROQ_API_KEY to your environment variables."
            )
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(config.timeout_seconds),
        )

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
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Groq request timed out after {self.config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Groq API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq HTTP error: {exc}") from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Groq returned no choices in response.")

        content = (choices[0].get("message") or {}).get("content", "").strip()
        if not content:
            raise ProviderError("Groq returned an empty response.")

        usage = data.get("usage") or {}
        return ChatCompletion(
            content=content,
            model=data.get("model", payload["model"]),
            finish_reason=choices[0].get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
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
        """Yield response tokens using Groq's SSE streaming."""
        import json as _json

        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    token = delta.get("content", "")
                    if token:
                        yield token
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Groq streaming timed out after {self.config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Groq streaming returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq streaming HTTP error: {exc}") from exc

    async def health(self) -> dict:
        """Probe /models to verify the API key is valid and the service is reachable."""
        result: dict[str, Any] = {
            "ok": False,
            "model": self.config.model,
            "error": None,
        }
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            available = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
            result["ok"] = True
            result["available_models"] = available
            result["model_loaded"] = self.config.model in available
        except httpx.TimeoutException as exc:
            result["error"] = f"timeout after {self.config.timeout_seconds:.1f}s"
            logger.warning("Groq health: %s", result["error"])
        except httpx.HTTPStatusError as exc:
            result["error"] = f"HTTP {exc.response.status_code}"
            logger.warning("Groq health: API key rejected or service error: %s", exc)
        except httpx.HTTPError as exc:
            result["error"] = str(exc)
            logger.warning("Groq health failed: %s", exc)
        return result

    async def close(self) -> None:
        await self._client.aclose()
