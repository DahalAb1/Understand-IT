from typing import Protocol
from .models import SimplificationRequest, SimplificationResult


class PdfReaderPort(Protocol):
    def extract(self, pdf_bytes: bytes) -> str:
        ...


class ModelPort(Protocol):
    max_input_length: int

    def simplify(self, text: str) -> str:
        ...


class CachePort(Protocol):
    def get(self, key: str) -> str | None:
        ...

    def set(self, key: str, value: str) -> None:
        ...


class SimplifierPort(Protocol):
    def simplify(self, request: SimplificationRequest) -> SimplificationResult:
        ...
