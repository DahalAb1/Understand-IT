from typing import Protocol

from .models import (
    ClauseContext,
    ClauseExtraction,
    DocumentMetadata,
    ExtractedDocument,
    SimplificationRequest,
    SimplificationResult,
)


class PdfReaderPort(Protocol):
    def extract(self, pdf_bytes: bytes) -> ExtractedDocument:
        ...


class ModelPort(Protocol):
    max_input_length: int

    def extract_clause(
        self,
        text: str,
        metadata: DocumentMetadata,
        source_location: str,
        context: ClauseContext,
    ) -> ClauseExtraction:
        ...

    def is_available(self) -> bool:
        ...


class CachePort(Protocol):
    def get(self, key: str) -> str | None:
        ...

    def set(self, key: str, value: str) -> None:
        ...


class SimplifierPort(Protocol):
    def simplify(self, request: SimplificationRequest) -> SimplificationResult:
        ...
