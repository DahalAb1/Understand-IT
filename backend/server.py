from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .adapters.inbound.api import create_router
from .adapters.outbound.gemini_adapter import GeminiAdapter
from .adapters.outbound.pypdf_reader import PypdfReaderAdapter
from .adapters.outbound.sqlite_cache import SqliteCacheAdapter
from .config import load_settings
from .domain.simplifier import SimplifierService

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
    model=GeminiAdapter(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        max_retries=settings.model_max_retries,
    ),
    cache=SqliteCacheAdapter(db_path=settings.cache_path),
)

app.include_router(create_router(simplifier, enable_trace=settings.enable_trace))
