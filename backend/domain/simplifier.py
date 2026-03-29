import hashlib
import json
from .models import Clause, RiskLevel, SimplificationRequest, SimplificationResult
from .ports import PdfReaderPort, ModelPort, CachePort


class SimplifierService:
    def __init__(self, reader: PdfReaderPort, model: ModelPort, cache: CachePort):
        self.reader = reader
        self.model = model
        self.cache = cache

    def simplify(self, request: SimplificationRequest) -> SimplificationResult:
        text = self.reader.extract(request.pdf_bytes)
        chunks = self._chunk(text, self.model.max_input_length)

        clauses = []
        for chunk in chunks:
            key = hashlib.sha256(chunk.encode()).hexdigest()

            cached = self.cache.get(key)
            if cached:
                data = json.loads(cached)
                clause = Clause(
                    title=data["title"],
                    original=chunk,
                    simplified=data["simplified"],
                    risk_level=RiskLevel(data["risk_level"]),
                    risk_reason=data["risk_reason"],
                )
            else:
                output = self.model.simplify(chunk)
                clause = Clause(
                    title=output.title,
                    original=chunk,
                    simplified=output.simplified,
                    risk_level=output.risk_level,
                    risk_reason=output.risk_reason,
                )
                self.cache.set(key, json.dumps({
                    "title": output.title,
                    "simplified": output.simplified,
                    "risk_level": output.risk_level.value,
                    "risk_reason": output.risk_reason,
                }))

            clauses.append(clause)

        return SimplificationResult(clauses=clauses)

    def _chunk(self, text: str, max_length: int) -> list[str]:
        words = text.split()
        chunks = []
        current = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(" ".join(current))
                current = [word]
                current_length = len(word)
            else:
                current.append(word)
                current_length += len(word) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks
