"""
Prediction interpretation services.

Wrap structured prediction results (from the ML or vision modules) into
Turkish, user-facing explanations. Each service builds its prompt from a
versioned template, calls the same shared ``ChatProvider`` used by the
chat service, and returns sanitized text.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.modules.ai.config import AIConfig
from app.modules.ai.prompts.fusion_interpretation import FUSION_INTERPRETATION_TR
from app.modules.ai.prompts.ml_interpretation import ML_INTERPRETATION_TR
from app.modules.ai.prompts.system import SYSTEM_PROMPT_TR
from app.modules.ai.prompts.vision_interpretation import VISION_INTERPRETATION_TR
from app.modules.ai.providers.base import ChatMessage, ChatProvider, ProviderError
from app.modules.ai.safety.filters import ResponseSanitizer

logger = logging.getLogger(__name__)


@dataclass
class InterpretationResult:
    content: str
    model: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    timestamp: str


class _BaseInterpretation:
    """Shared plumbing — both interpretation services follow the same shape."""

    def __init__(
        self,
        provider: ChatProvider,
        config: AIConfig,
        sanitizer: ResponseSanitizer | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.sanitizer = sanitizer or ResponseSanitizer(config.safety)

    async def _call(self, user_prompt: str) -> InterpretationResult:
        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT_TR.render()),
            ChatMessage(role="user", content=user_prompt),
        ]
        try:
            completion = await self.provider.chat(
                messages,
                model=self.config.groq.model,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                max_tokens=self.config.generation.max_tokens,
            )
        except ProviderError as exc:
            logger.error("AI interpretation: provider error: %s", exc)
            raise

        clean = self.sanitizer.sanitize(completion.content)
        duration_ms = (time.perf_counter() - start) * 1000

        return InterpretationResult(
            content=clean,
            model=completion.model,
            duration_ms=round(duration_ms, 2),
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            timestamp=timestamp,
        )


class MLInterpretationService(_BaseInterpretation):
    """Explains a single-patient ML prediction."""

    async def explain(self, prediction: dict) -> InterpretationResult:
        features = prediction.get("feature_summary") or "(belirtilmemiş)"
        user_prompt = ML_INTERPRETATION_TR.render(
            label=prediction.get("label", "?"),
            prediction=prediction.get("prediction", "?"),
            probability=_fmt_prob(prediction.get("probability")),
            confidence=prediction.get("confidence", "?"),
            model_name=prediction.get("model_name", "?"),
            model_version=prediction.get("model_version", "?"),
            feature_summary=features,
        )
        return await self._call(user_prompt)


class VisionInterpretationService(_BaseInterpretation):
    """Explains a single vision prediction (including rejections)."""

    async def explain(self, prediction: dict) -> InterpretationResult:
        gate = prediction.get("gate") or {}
        user_prompt = VISION_INTERPRETATION_TR.render(
            accepted=_yesno_tr(prediction.get("accepted", False)),
            predicted_class=prediction.get("predicted_class") or "(yok)",
            confidence=_fmt_prob(prediction.get("confidence")),
            threshold=_fmt_prob(prediction.get("threshold")),
            rejection_reason=prediction.get("rejection_reason") or "(yok)",
            gate_enabled=_yesno_tr(gate.get("enabled", False)),
            gate_predicted_class=gate.get("predicted_class") or "(yok)",
            model_name=prediction.get("model_name", "?"),
            model_version=prediction.get("model_version", "?"),
        )
        return await self._call(user_prompt)


class FusionInterpretationService(_BaseInterpretation):
    """Explains a multimodal fusion result in Turkish."""

    async def explain(self, fusion: dict) -> InterpretationResult:
        ep = fusion.get("explanation_payload") or {}
        flags = ep.get("uncertainty_flags") or []
        flags_str = ", ".join(flags) if flags else "(yok)"
        vision_used = ep.get("vision_used", False)
        vision_conf = ep.get("vision_confidence")
        dominant = ep.get("dominant_signal", "ml_only")
        dominant_str = (
            "ML semptom modeli + görüntü analizi"
            if dominant == "ml_and_vision"
            else "Yalnızca ML semptom modeli"
        )
        user_prompt = FUSION_INTERPRETATION_TR.render(
            final_risk_score=ep.get("final_risk_score", 0.0),
            risk_level=ep.get("risk_level", "?"),
            fusion_confidence=fusion.get("fusion_confidence", "?"),
            ml_label=ep.get("ml_label", "?"),
            ml_probability=ep.get("ml_probability", 0.0),
            ml_confidence=ep.get("ml_confidence", "?"),
            vision_used=_yesno_tr(vision_used),
            vision_status=ep.get("vision_status", "unavailable"),
            vision_class=ep.get("vision_class") or "(yok)",
            vision_confidence=_fmt_prob(vision_conf),
            vision_rejection_reason=ep.get("vision_rejection_reason") or "(yok)",
            uncertainty_flags=flags_str,
            dominant_signal=dominant_str,
        )
        return await self._call(user_prompt)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_prob(value) -> str:
    if value is None:
        return "(bilinmiyor)"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _yesno_tr(flag: bool) -> str:
    return "evet" if flag else "hayır"