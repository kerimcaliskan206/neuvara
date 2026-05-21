"""
MedicalAssistantService — Phase 26.

Context-aware AI assistant for medical analysis explanations.
Knows the current UnifiedAnalysisSession (curated fields) and can answer
questions about THIS specific encounter in Turkish clinical language.

Safety contract:
  - No diagnosis, no medication, no treatment plan
  - Reuses existing PromptInjectionFilter + ResponseSanitizer
  - Conversation memory isolated per session_id
  - Backend internals (ECE, T*, fusion scores) never exposed
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from app.modules.ai.config import AIConfig
from app.modules.ai.memory.conversation import ConversationMemory
from app.modules.ai.prompts.medical_system import (
    build_medical_system_prompt,
    needs_disclaimer,
)
from app.modules.ai.providers.base import (
    ChatMessage,
    ChatProvider,
    ProviderError,
)
from app.modules.ai.safety.filters import PromptInjectionFilter, ResponseSanitizer
from app.modules.ai.safety.validators import InputValidator
from app.schemas.medical_assistant import MedicalAnalysisContext

logger = logging.getLogger(__name__)

_INJECTION_REFUSAL = (
    "Bu istek sistem talimatlarını değiştirmeye yönelik göründüğü için "
    "yanıtlanamadı. Mevcut analiz sonucu hakkında bir soru sormak ister misiniz?"
)

_INVALID_INPUT_REFUSAL = (
    "Mesajınız işlenemedi. Lütfen sorunuzu tekrar yazın."
)


@dataclass
class MedicalAssistantResult:
    content: str
    refused: bool
    refusal_reason: str | None
    model: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    timestamp: str


class MedicalAssistantService:
    """
    Stateless per-request reasoning, stateful per-session memory.

    Construct once at startup; reuse across requests.
    Each session_id gets its own conversation memory window.
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

    async def ask(
        self,
        *,
        message: str,
        session_id: str,
        analysis_context: MedicalAnalysisContext,
    ) -> MedicalAssistantResult:
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Medical assistant: session=%s len=%d risk=%s",
            session_id, len(message or ""), analysis_context.risk_tier,
        )

        # 1. Input length / shape validation
        validation = self.input_validator.validate(message)
        if not validation.ok:
            return self._refusal(
                content=_INVALID_INPUT_REFUSAL,
                reason="invalid_input",
                start=start,
                timestamp=timestamp,
            )

        # 2. Prompt-injection scan
        detection = self.injection_filter.scan(message)
        if detection.detected and self.config.safety.refuse_on_injection:
            logger.warning(
                "Medical assistant: refused session=%s reason=injection",
                session_id,
            )
            return self._refusal(
                content=_INJECTION_REFUSAL,
                reason="prompt_injection",
                start=start,
                timestamp=timestamp,
            )

        # 3. Build message stack (system = context-injected prompt + history + user)
        messages = self._build_messages(
            message=message,
            session_id=session_id,
            analysis_context=analysis_context,
        )

        # 4. LLM call
        try:
            completion = await self.provider.chat(
                messages,
                model=self.config.ollama.model,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                max_tokens=self.config.generation.max_tokens,
            )
        except ProviderError as exc:
            logger.error(
                "Medical assistant: provider error session=%s err=%s", session_id, exc
            )
            raise

        # 5. Sanitize — strip any debug/internal leakage
        clean = self.sanitizer.sanitize(completion.content)

        # 6. Append disclaimer when response touches risk/treatment territory
        if needs_disclaimer(message.lower()) or needs_disclaimer(clean.lower()):
            if not clean.endswith("*"):
                clean += "\n\n*Bu yanıt tıbbi teşhis veya tedavi tavsiyesi değildir.*"

        # 7. Persist turn for multi-turn continuity
        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "assistant", clean)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "Medical assistant: ok session=%s duration_ms=%.1f model=%s "
            "tokens prompt=%s completion=%s",
            session_id, duration_ms, completion.model,
            completion.prompt_tokens, completion.completion_tokens,
        )

        return MedicalAssistantResult(
            content=clean,
            refused=False,
            refusal_reason=None,
            model=completion.model,
            duration_ms=duration_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            timestamp=timestamp,
        )

    async def ask_stream(
        self,
        *,
        message: str,
        session_id: str,
        analysis_context: MedicalAnalysisContext,
    ) -> AsyncIterator[str]:
        """Stream response tokens. Safety checks run synchronously before the first yield."""
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Medical assistant stream: session=%s len=%d risk=%s",
            session_id, len(message or ""), analysis_context.risk_tier,
        )

        validation = self.input_validator.validate(message)
        if not validation.ok:
            yield _INVALID_INPUT_REFUSAL
            return

        detection = self.injection_filter.scan(message)
        if detection.detected and self.config.safety.refuse_on_injection:
            logger.warning("Medical assistant stream: refused session=%s reason=injection", session_id)
            yield _INJECTION_REFUSAL
            return

        messages = self._build_messages(
            message=message,
            session_id=session_id,
            analysis_context=analysis_context,
        )

        full_response = ""
        try:
            async for token in self.provider.chat_stream(
                messages,
                model=self.config.ollama.model,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                max_tokens=self.config.generation.max_tokens,
            ):
                full_response += token
                yield token
        except ProviderError:
            raise

        clean = self.sanitizer.sanitize(full_response)

        if needs_disclaimer(message.lower()) or needs_disclaimer(clean.lower()):
            if not clean.endswith("*"):
                disclaimer = "\n\n*Bu yanıt tıbbi teşhis veya tedavi tavsiyesi değildir.*"
                yield disclaimer
                clean += disclaimer

        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "assistant", clean)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Medical assistant stream: ok session=%s duration_ms=%.1f",
            session_id, duration_ms,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        *,
        message: str,
        session_id: str,
        analysis_context: MedicalAnalysisContext,
    ) -> list[ChatMessage]:
        system_content = build_medical_system_prompt(analysis_context)
        history = self.memory.recent(session_id)
        return [
            ChatMessage(role="system", content=system_content),
            *(ChatMessage(role=t.role, content=t.content) for t in history),
            ChatMessage(role="user", content=message),
        ]

    def _refusal(
        self,
        *,
        content: str,
        reason: str,
        start: float,
        timestamp: str,
    ) -> MedicalAssistantResult:
        return MedicalAssistantResult(
            content=content,
            refused=True,
            refusal_reason=reason,
            model=self.config.ollama.model,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            prompt_tokens=None,
            completion_tokens=None,
            timestamp=timestamp,
        )
