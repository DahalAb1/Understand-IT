import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str = "gemini"
    gemini_api_key: str | None = None
    allowed_origin: str = "http://localhost:5173"
    cache_path: str = "cache.db"


def load_settings() -> Settings:
    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "gemini").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        allowed_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        cache_path=os.getenv("CACHE_DB_PATH", "cache.db"),
    )
