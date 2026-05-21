"""
AI assistant configuration.

All values can be overridden via environment variables (see app.core.config.Settings)
so the AI layer is fully configurable without code changes.
"""
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Connection settings for the Ollama HTTP API."""

    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3.1:8b")
    timeout_seconds: float = Field(default=60.0, gt=0)
    keep_alive: str = Field(
        default="5m",
        description="Ollama keep_alive setting — keeps the model resident between calls.",
    )


class GenerationConfig(BaseModel):
    """Per-call generation defaults."""

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=300, gt=0)


class SafetyConfig(BaseModel):
    """Input / output safety limits."""

    max_input_chars: int = Field(default=2000, gt=0)
    max_output_chars: int = Field(default=4000, gt=0)
    max_conversation_turns: int = Field(default=8, gt=0)
    refuse_on_injection: bool = Field(default=True)


class AIConfig(BaseModel):
    """Top-level AI module config."""

    ollama: OllamaConfig = OllamaConfig()
    generation: GenerationConfig = GenerationConfig()
    safety: SafetyConfig = SafetyConfig()
    enabled: bool = Field(
        default=True,
        description="Master switch — when False, /ai endpoints return 503.",
    )
    default_language: str = Field(default="tr")


def build_ai_config_from_settings() -> "AIConfig":
    """Hydrate AIConfig from the application Settings (env-driven)."""
    from app.core.config import settings  # local import avoids circular

    return AIConfig(
        enabled=settings.AI_ENABLED,
        ollama=OllamaConfig(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout_seconds=settings.OLLAMA_TIMEOUT_SECONDS,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
        ),
        generation=GenerationConfig(
            temperature=settings.AI_TEMPERATURE,
            top_p=settings.AI_TOP_P,
            max_tokens=settings.AI_MAX_TOKENS,
        ),
        safety=SafetyConfig(
            max_input_chars=settings.AI_MAX_INPUT_CHARS,
            max_output_chars=settings.AI_MAX_OUTPUT_CHARS,
            max_conversation_turns=settings.AI_MAX_CONVERSATION_TURNS,
        ),
    )


ai_config = build_ai_config_from_settings()