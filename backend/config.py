import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_provider: str = "cloudflare"
    allowed_origin: str = "http://localhost:5173"
    cache_path: str = "cache.db"
    model_max_retries: int = 2
    enable_trace: bool = False


# Provider-specific keys/models are no longer named fields here. The adapter registry
# (adapters/outbound/registry.py) resolves them generically from {PROVIDER}_API_KEY /
# {PROVIDER}_MODEL env vars, so adding a provider never touches this file.
def load_settings() -> Settings:
    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "cloudflare").strip().lower(),
        allowed_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        cache_path=os.getenv("CACHE_DB_PATH", "cache.db"),
        model_max_retries=max(0, int(os.getenv("MODEL_MAX_RETRIES", "2"))),
        enable_trace=_as_bool(os.getenv("ENABLE_TRACE"), default=False),
    )
