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

    # Database
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

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

    # AI assistant (Ollama)
    AI_ENABLED: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_KEEP_ALIVE: str = "5m"
    AI_TEMPERATURE: float = 0.2
    AI_TOP_P: float = 0.9
    AI_MAX_TOKENS: int = 512
    AI_MAX_INPUT_CHARS: int = 2000
    AI_MAX_OUTPUT_CHARS: int = 4000
    AI_MAX_CONVERSATION_TURNS: int = 8

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

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
