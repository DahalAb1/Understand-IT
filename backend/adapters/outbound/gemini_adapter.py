import json
import os
from typing import Any

import google.generativeai as genai

from ...domain.models import ClauseExtraction, DocumentMetadata, RiskLevel


class GeminiAdapter:
    max_input_length = 4000

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)

    def extract_clause(self, text: str, metadata: DocumentMetadata, source_location: str) -> ClauseExtraction:
        if self.model is None:
            raise RuntimeError("The configured model provider is unavailable. Set GEMINI_API_KEY to enable clause extraction.")

        prompt = self._build_prompt(text, metadata, source_location)
        response = self.model.generate_content(prompt)
        payload = self._load_json(response.text)

        return ClauseExtraction(
            title=self._as_string(payload, "title", fallback="Untitled Clause"),
            clause_type=self._as_string(payload, "clause_type", fallback="general"),
            source_text=text,
            source_location=source_location,
            defined_terms_used=self._as_list(payload, "defined_terms_used"),
            obligations=self._as_list(payload, "obligations"),
            rights=self._as_list(payload, "rights"),
            conditions=self._as_list(payload, "conditions"),
            exceptions=self._as_list(payload, "exceptions"),
            deadlines=self._as_list(payload, "deadlines"),
            money_terms=self._as_list(payload, "money_terms"),
            plain_english=self._as_string(payload, "plain_english", fallback="This clause requires review."),
            legal_precision_note=self._optional_string(payload, "legal_precision_note"),
            questions_to_ask=self._as_list(payload, "questions_to_ask"),
            risk_level=self._risk_level(payload.get("risk_level")),
            risk_reason=self._optional_string(payload, "risk_reason"),
            confidence=self._as_confidence(payload.get("confidence")),
            missing_context=self._as_list(payload, "missing_context"),
        )

    def _build_prompt(self, text: str, metadata: DocumentMetadata, source_location: str) -> str:
        return (
            "You are a legal document accessibility assistant. "
            "Your job is to explain legal clauses in plain language while preserving legal nuance. "
            "Do not provide legal advice. Extract obligations, rights, conditions, exceptions, deadlines, and money terms. "
            "If context is missing, say so explicitly instead of guessing.\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "title": string,\n'
            '  "clause_type": string,\n'
            '  "defined_terms_used": string[],\n'
            '  "obligations": string[],\n'
            '  "rights": string[],\n'
            '  "conditions": string[],\n'
            '  "exceptions": string[],\n'
            '  "deadlines": string[],\n'
            '  "money_terms": string[],\n'
            '  "plain_english": string,\n'
            '  "legal_precision_note": string | null,\n'
            '  "questions_to_ask": string[],\n'
            '  "risk_level": "low" | "medium" | "high",\n'
            '  "risk_reason": string | null,\n'
            '  "confidence": number,\n'
            '  "missing_context": string[]\n'
            "}\n\n"
            f"Document type: {metadata.document_type}\n"
            f"Governing law: {metadata.governing_law or 'unknown'}\n"
            f"OCR quality: {metadata.ocr_quality}\n"
            f"Document warnings: {metadata.warnings or ['none']}\n"
            f"Source location: {source_location}\n\n"
            "Preserve legal nuance. Pay close attention to negations, conditions, exceptions, deadlines, money, and one-sided rights.\n\n"
            f"Clause:\n{text}"
        )

    def _load_json(self, text: str) -> dict[str, Any]:
        payload = text.strip()
        if payload.startswith("```"):
            payload = payload.split("```", 2)[1]
            if payload.startswith("json"):
                payload = payload[4:]
        payload = payload.strip()

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("The model returned invalid structured output.") from exc

    def _risk_level(self, value: Any) -> RiskLevel:
        try:
            return RiskLevel(str(value).lower())
        except ValueError:
            return RiskLevel.MEDIUM

    def _as_list(self, payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _as_string(self, payload: dict[str, Any], key: str, fallback: str) -> str:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback

    def _optional_string(self, payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _as_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.65
