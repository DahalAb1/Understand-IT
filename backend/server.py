from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .domain.simplifier import SimplifierService
from .adapters.outbound.pypdf_reader import PypdfReaderAdapter
from .adapters.outbound.gemini_adapter import GeminiAdapter
from .adapters.outbound.sqlite_cache import SqliteCacheAdapter
from .adapters.inbound.api import create_router

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

simplifier = SimplifierService(
    reader=PypdfReaderAdapter(),
    model=GeminiAdapter(),
    cache=SqliteCacheAdapter(),
)

app.include_router(create_router(simplifier))
