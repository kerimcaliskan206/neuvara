"""
Prompt injection detection + response sanitisation.

These are *defense in depth* layers — they cannot guarantee security on
their own, but they catch the obvious attempts before they reach the
provider, and they keep raw model output from leaking dangerous content
to clients.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.modules.ai.config import SafetyConfig

logger = logging.getLogger(__name__)


# Patterns are intentionally broad. False positives are acceptable because
# the user can rephrase; false negatives are not.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.I),
    re.compile(r"forget\s+(everything|all\s+previous|your\s+instructions)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"\bact\s+as\s+(a|an|if)\b", re.I),
    re.compile(r"pretend\s+(to\s+be|you\s+are)", re.I),
    re.compile(r"new\s+(system|role)\s*[:\-]", re.I),
    re.compile(r"^\s*system\s*[:\-]", re.I | re.M),
    re.compile(r"^\s*assistant\s*[:\-]", re.I | re.M),
    re.compile(r"<\s*\|?\s*(system|assistant|user)\s*\|?\s*>", re.I),
    re.compile(r"###\s*(system|new\s+instructions)", re.I),
    # Turkish variants — same intent, different language
    re.compile(r"önceki\s+talimatları\s+(yok\s+say|unut|göz\s+ardı\s+et)", re.I),
    re.compile(r"sen\s+artık\s+\w+sin", re.I),
    re.compile(r"şu\s+andan\s+itibaren\s+sen", re.I),
    re.compile(r"kuralları\s+(unut|yok\s+say|göz\s+ardı\s+et)", re.I),
]


@dataclass
class InjectionDetection:
    detected: bool
    pattern: str | None = None


class PromptInjectionFilter:
    """Detect common prompt-injection attempts in user messages."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def scan(self, message: str) -> InjectionDetection:
        if not message:
            return InjectionDetection(detected=False)

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(message):
                logger.warning(
                    "PromptInjectionFilter: blocked message matching pattern %s",
                    pattern.pattern,
                )
                return InjectionDetection(detected=True, pattern=pattern.pattern)

        return InjectionDetection(detected=False)


class ResponseSanitizer:
    """Strip dangerous content and enforce length caps on model output."""

    # Tags we never want appearing in a downstream client.
    _STRIP_HTML = re.compile(r"<\s*(script|iframe|object|embed)[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
    _STRIP_EVENT_HANDLERS = re.compile(r"\bon\w+\s*=\s*\"[^\"]*\"", re.I)

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def sanitize(self, text: str) -> str:
        if not text:
            return ""

        cleaned = self._STRIP_HTML.sub("", text)
        cleaned = self._STRIP_EVENT_HANDLERS.sub("", cleaned)
        cleaned = cleaned.strip()

        if len(cleaned) > self.config.max_output_chars:
            cleaned = cleaned[: self.config.max_output_chars].rstrip() + "…"

        return cleaned