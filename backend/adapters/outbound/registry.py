import logging
import os

from ...config import Settings
from ...domain.ports import ModelPort
from .cloudflare_adapter import CloudflareAdapter
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


# Provider name -> adapter class. Adding a provider means writing one adapter class
# (exposing provider_name / default_model) and adding a single line here — no changes
# to config.py, server.py, or the eval harness.
ADAPTER_REGISTRY: dict[str, type[ModelPort]] = {
    OpenAIAdapter.provider_name: OpenAIAdapter,
    GeminiAdapter.provider_name: GeminiAdapter,
    CloudflareAdapter.provider_name: CloudflareAdapter,
}


def build_model_adapter(settings: Settings, model_override: str | None = None) -> ModelPort:
    adapter_cls = ADAPTER_REGISTRY.get(settings.model_provider)
    if adapter_cls is None:
        raise RuntimeError(
            f"Unsupported MODEL_PROVIDER '{settings.model_provider}'. "
            f"Expected one of: {', '.join(sorted(ADAPTER_REGISTRY))}."
        )

    prefix = adapter_cls.provider_name.upper()
    model_name = model_override or os.getenv(f"{prefix}_MODEL", adapter_cls.default_model)
    adapter = adapter_cls(
        api_key=os.getenv(f"{prefix}_API_KEY"),
        model_name=model_name,
        max_retries=settings.model_max_retries,
    )

    if not adapter.is_available():
        logger.warning(
            "Model provider '%s' is configured but not available (missing API key or SDK) "
            "— clause extraction will fall back to the heuristic extractor.",
            settings.model_provider,
        )

    return adapter
