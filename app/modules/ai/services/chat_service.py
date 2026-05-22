"""
AIChatService — orchestrates user-facing chat.

Responsibilities (in order):
  1. Validate input length / shape.
  2. Detect prompt-injection attempts and emit a refusal if found.
  3. Classify intent (deterministic keyword routing).
  4. Build the message list (system prompt + memory + user turn).
  5. Call the provider with generation defaults.
  6. Sanitize the response.
  7. Update conversation memory.
  8. Emit structured log line.

The service depends only on abstractions (``ChatProvider``, memory store,
filters), so it is fully testable with a fake provider.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.modules.ai.config import AIConfig
from app.modules.ai.memory.conversation import ConversationMemory
from app.modules.ai.prompts.system import (
    INJECTION_REFUSAL_TR,
    REFUSAL_TR,
    SYSTEM_PROMPT_TR,
)
from app.modules.ai.providers.base import (
    ChatMessage,
    ChatProvider,
    ProviderError,
)
from app.modules.ai.routing.intent import Intent, detect_intent
from app.modules.ai.safety.filters import (
    PromptInjectionFilter,
    ResponseSanitizer,
)
from app.modules.ai.safety.validators import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """User-facing chat result + metadata."""

    content: str
    intent: str
    refused: bool
    refusal_reason: str | None
    model: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    timestamp: str


class AIChatService:
    """
    Chat orchestration. Construct once at startup and reuse — both the
    provider's HTTP client and the conversation memory live on the service.
    """

    def __init__(
        self,
        *,
        provider: ChatProvider,
        config: AIConfig,
        memory: ConversationMemory | None = None,
        input_validator: InputValidator | None = None,
        injection_filter: PromptInjectionFilter | None = None,
        sanitizer: ResponseSanitizer | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.memory = memory or ConversationMemory(
            max_turns=config.safety.max_conversation_turns
        )
        self.input_validator = input_validator or InputValidator(config.safety)
        self.injection_filter = injection_filter or PromptInjectionFilter(config.safety)
        self.sanitizer = sanitizer or ResponseSanitizer(config.safety)

    # ── Public API ───────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        message: str,
        session_id: str = "default",
        has_ml_context: bool = False,
        has_vision_context: bool = False,
    ) -> ChatResult:
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "AI chat: received session=%s len=%d ml_ctx=%s vision_ctx=%s",
            session_id, len(message or ""), has_ml_context, has_vision_context,
        )

        # 1. Input validation
        validation = self.input_validator.validate(message)
        if not validation.ok:
            return self._refusal_result(
                content=validation.reason or "Mesaj geçersiz.",
                intent=Intent.OFF_TOPIC.value,
                refusal_reason="invalid_input",
                start=start,
                timestamp=timestamp,
            )

        # 2. Prompt-injection scan
        detection = self.injection_filter.scan(message)
        if detection.detected and self.config.safety.refuse_on_injection:
            logger.warning(
                "AI chat: refused session=%s reason=injection pattern=%s",
                session_id, detection.pattern,
            )
            return self._refusal_result(
                content=INJECTION_REFUSAL_TR,
                intent=Intent.OFF_TOPIC.value,
                refusal_reason="prompt_injection",
                start=start,
                timestamp=timestamp,
            )

        # 3. Intent routing
        intent = detect_intent(
            message,
            has_ml_context=has_ml_context,
            has_vision_context=has_vision_context,
        )
        if intent == Intent.OFF_TOPIC:
            logger.info(
                "AI chat: refused session=%s reason=off_topic", session_id,
            )
            return self._refusal_result(
                content=REFUSAL_TR,
                intent=intent.value,
                refusal_reason="off_topic",
                start=start,
                timestamp=timestamp,
            )

        # 4. Build the message stack
        messages = self._build_messages(message=message, session_id=session_id)

        # 5. Provider call
        try:
            completion = await self.provider.chat(
                messages,
                model=self.config.groq.model,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                max_tokens=self.config.generation.max_tokens,
            )
        except ProviderError as exc:
            logger.error("AI chat: provider error session=%s err=%s", session_id, exc)
            raise

        # 6. Sanitize + 7. Persist turn
        clean = self.sanitizer.sanitize(completion.content)
        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "assistant", clean)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "AI chat: ok session=%s intent=%s duration_ms=%.1f model=%s "
            "tokens prompt=%s completion=%s",
            session_id, intent.value, duration_ms, completion.model,
            completion.prompt_tokens, completion.completion_tokens,
        )

        return ChatResult(
            content=clean,
            intent=intent.value,
            refused=False,
            refusal_reason=None,
            model=completion.model,
            duration_ms=round(duration_ms, 2),
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            timestamp=timestamp,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_messages(self, *, message: str, session_id: str) -> list[ChatMessage]:
        history = self.memory.recent(session_id)
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT_TR.render()),
            *(ChatMessage(role=t.role, content=t.content) for t in history),
            ChatMessage(role="user", content=message),
        ]

    def _refusal_result(
        self,
        *,
        content: str,
        intent: str,
        refusal_reason: str,
        start: float,
        timestamp: str,
    ) -> ChatResult:
        return ChatResult(
            content=content,
            intent=intent,
            refused=True,
            refusal_reason=refusal_reason,
            model=self.config.groq.model,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            timestamp=timestamp,
        )