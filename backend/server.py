from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .adapters.inbound.api import create_router
from .adapters.outbound.gemini_adapter import GeminiAdapter
from .adapters.outbound.openai_adapter import OpenAIAdapter
from .adapters.outbound.pypdf_reader import PypdfReaderAdapter
from .adapters.outbound.sqlite_cache import SqliteCacheAdapter
from .config import load_settings
from .domain.simplifier import SimplifierService
from .domain.ports import ModelPort


def build_model_adapter() -> ModelPort:
    if settings.model_provider == "openai":
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            max_retries=settings.model_max_retries,
        )

    if settings.model_provider == "gemini":
        return GeminiAdapter(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            max_retries=settings.model_max_retries,
        )

    raise RuntimeError(
        f"Unsupported MODEL_PROVIDER '{settings.model_provider}'. Expected one of: gemini, openai."
    )

load_dotenv()
settings = load_settings()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

simplifier = SimplifierService(
    reader=PypdfReaderAdapter(),
    model=build_model_adapter(),
    cache=SqliteCacheAdapter(db_path=settings.cache_path),
)

app.include_router(create_router(simplifier, enable_trace=settings.enable_trace))
