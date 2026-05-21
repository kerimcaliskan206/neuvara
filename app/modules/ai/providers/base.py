"""
Provider abstraction.

Every chat provider (Ollama, OpenAI-compatible, future LiteLLM, etc.)
implements the same async ``chat`` + ``health`` interface so the rest of
the AI module never depends on a concrete backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


class ProviderError(RuntimeError):
    """Raised when a provider fails (network, timeout, bad response)."""


@dataclass
class ChatMessage:
    """Wire-level chat message — one of system / user / assistant."""

    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletion:
    """Provider-agnostic chat completion result."""

    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict | None = None


class ChatProvider(ABC):
    """Async chat-completion provider."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        """Send a chat request and return the completion."""

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens as they arrive. Raises NotImplementedError if unsupported."""
        raise NotImplementedError("This provider does not support streaming.")
        yield  # makes this an async generator

    @abstractmethod
    async def health(self) -> dict:
        """
        Return a health snapshot. Implementations should not raise — they
        should serialize the failure into the returned dict (``ok=False``).
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources (HTTP clients, sessions)."""