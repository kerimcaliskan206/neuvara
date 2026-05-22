"""
AI health service.

Wraps the provider's health probe with a small timeout fence and produces
a structured, route-friendly dict. Designed to fail fast — if the provider
hangs or errors out, the API still returns within a few seconds.
"""
from __future__ import annotations

import asyncio
import logging

from app.modules.ai.config import AIConfig
from app.modules.ai.providers.base import ChatProvider

logger = logging.getLogger(__name__)


class AIHealthService:
    def __init__(self, provider: ChatProvider, config: AIConfig) -> None:
        self.provider = provider
        self.config = config

    async def check(self) -> dict:
        if not self.config.enabled:
            return {
                "ok": False,
                "enabled": False,
                "reason": "AI module is disabled via config.",
            }

        # Cap the health probe so a slow network never blocks the endpoint.
        probe_timeout = min(self.config.groq.timeout_seconds, 10.0)
        try:
            result = await asyncio.wait_for(self.provider.health(), timeout=probe_timeout)
        except asyncio.TimeoutError:
            logger.warning("AI health: provider probe timed out after %.1fs", probe_timeout)
            return {
                "ok": False,
                "enabled": True,
                "model": self.config.groq.model,
                "reason": f"Provider probe timed out after {probe_timeout:.1f}s",
            }

        return {
            "ok": bool(result.get("ok")),
            "enabled": True,
            "model": result.get("model", self.config.groq.model),
            "model_loaded": result.get("model_loaded", False),
            "available_models": result.get("available_models", []),
            "error": result.get("error"),
        }