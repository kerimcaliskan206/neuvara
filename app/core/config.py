from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    APP_NAME: str = "HantaProject"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    TESTING: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Database — Neon/Railway provides a single DATABASE_URL
    DATABASE_URL: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Email (Resend)
    RESEND_API_KEY: str = ""
    MAIL_FROM: str = "NEURAVA <noreply@neurava.ai>"
    FRONTEND_URL: str = "http://localhost:3000"

    # Startup model loading — set false to skip warm-up when wheels mismatch
    # the persisted model's ABI. Endpoints return 503 until a fresh model
    # is trained and the flag is flipped back on.
    ML_AUTO_LOAD_ON_STARTUP: bool = True
    VISION_AUTO_LOAD_ON_STARTUP: bool = True

    # AI assistant (Groq)
    AI_ENABLED: bool = True
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TIMEOUT_SECONDS: float = 30.0
    AI_TEMPERATURE: float = 0.2
    AI_TOP_P: float = 0.9
    AI_MAX_TOKENS: int = 512
    AI_MAX_INPUT_CHARS: int = 2000
    AI_MAX_OUTPUT_CHARS: int = 4000
    AI_MAX_CONVERSATION_TURNS: int = 8

    @property
    def database_url(self) -> str:
        """asyncpg-compatible URL for SQLAlchemy async engine."""
        url = self.DATABASE_URL
        url = url.replace("postgres://", "postgresql://", 1)
        if not url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def database_url_sync(self) -> str:
        """Standard psycopg2-compatible URL for Alembic offline mode."""
        url = self.DATABASE_URL
        url = url.replace("postgres://", "postgresql://", 1)
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return url

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def docs_enabled(self) -> bool:
        return self.DEBUG and not self.is_production


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
