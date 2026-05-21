"""
Input validation for AI requests.

Catches obviously bad inputs at the boundary before they reach the
provider or the safety filters. Each check returns a structured
``ValidationOutcome`` so routes can map failures to specific HTTP codes.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.modules.ai.config import SafetyConfig


@dataclass
class ValidationOutcome:
    ok: bool
    reason: str | None = None


class InputValidator:
    """Length and shape checks for user-supplied chat messages."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def validate(self, message: str) -> ValidationOutcome:
        if message is None:
            return ValidationOutcome(ok=False, reason="Mesaj boş olamaz.")

        stripped = message.strip()
        if not stripped:
            return ValidationOutcome(ok=False, reason="Mesaj boş olamaz.")

        if len(stripped) > self.config.max_input_chars:
            return ValidationOutcome(
                ok=False,
                reason=(
                    f"Mesaj çok uzun (maksimum {self.config.max_input_chars} karakter)."
                ),
            )

        return ValidationOutcome(ok=True)