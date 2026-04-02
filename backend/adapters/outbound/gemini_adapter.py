import json
import os
import time
from typing import Any

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None

from ...domain.models import ClauseContext, ClauseExtraction, DocumentMetadata, RiskLevel
from ...domain.policies import get_clause_policy


CLAUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "propertyOrdering": [
        "title",
        "clause_type",
        "defined_terms_used",
        "obligations",
        "rights",
        "conditions",
        "exceptions",
        "deadlines",
        "money_terms",
        "plain_english",
        "legal_precision_note",
        "questions_to_ask",
        "risk_level",
        "risk_reason",
        "confidence",
        "missing_context",
    ],
    "properties": {
        "title": {"type": "string"},
        "clause_type": {"type": "string"},
        "defined_terms_used": {"type": "array", "items": {"type": "string"}},
        "obligations": {"type": "array", "items": {"type": "string"}},
        "rights": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "exceptions": {"type": "array", "items": {"type": "string"}},
        "deadlines": {"type": "array", "items": {"type": "string"}},
        "money_terms": {"type": "array", "items": {"type": "string"}},
        "plain_english": {"type": "string"},
        "legal_precision_note": {"type": ["string", "null"]},
        "questions_to_ask": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "risk_reason": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "missing_context": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "clause_type",
        "defined_terms_used",
        "obligations",
        "rights",
        "conditions",
        "exceptions",
        "deadlines",
        "money_terms",
        "plain_english",
        "legal_precision_note",
        "questions_to_ask",
        "risk_level",
        "risk_reason",
        "confidence",
        "missing_context",
    ],
}


class GeminiAdapter:
    max_input_length = 4000

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = None
        self.legacy_model = None

        if not self.api_key:
            return

        if google_genai is not None:
            self.client = google_genai.Client(api_key=self.api_key)
            return

        if legacy_genai is not None:
            legacy_genai.configure(api_key=self.api_key)
            self.legacy_model = legacy_genai.GenerativeModel(model_name)

    def is_available(self) -> bool:
        return self.client is not None or self.legacy_model is not None

    def extract_clause(
        self,
        text: str,
        metadata: DocumentMetadata,
        source_location: str,
        context: ClauseContext,
    ) -> ClauseExtraction:
        if not self.is_available():
            raise RuntimeError("The configured model provider is unavailable. Set GEMINI_API_KEY and install a supported Gemini SDK to enable clause extraction.")

        prompt = self._build_prompt(text, metadata, source_location, context)
        payload = self._generate_payload(prompt)

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

    def _generate_payload(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.client is not None:
                    return self._generate_with_current_sdk(prompt)
                if self.legacy_model is not None:
                    return self._generate_with_legacy_sdk(prompt)
                raise RuntimeError("No Gemini SDK is available.")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError("The model could not produce a reliable structured extraction.") from last_error

    def _generate_with_current_sdk(self, prompt: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=google_genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=CLAUSE_SCHEMA,
                temperature=0.2,
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("The model returned an empty response.")
        return self._load_json(text)

    def _generate_with_legacy_sdk(self, prompt: str) -> dict[str, Any]:
        response = self.legacy_model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("The model returned an empty response.")
        return self._load_json(self._strip_code_fence(text))

    def _build_prompt(
        self,
        text: str,
        metadata: DocumentMetadata,
        source_location: str,
        context: ClauseContext,
    ) -> str:
        policy = get_clause_policy(self._detect_policy_type(text))
        definition_lines = [
            f'- "{definition.term}": {definition.definition} ({definition.source_location})'
            for definition in context.relevant_definitions
        ]
        referenced_lines = [f"- {entry}" for entry in context.referenced_texts]
        policy_focus = ", ".join(policy.required_focus) if policy else "general legal meaning"
        policy_questions = "\n".join(f"- {question}" for question in policy.review_questions) if policy else "- none"
        policy_triggers = ", ".join(policy.review_triggers) if policy else "none"

        return (
            "You are a legal document accessibility assistant. "
            "Explain legal clauses in plain language while preserving legal nuance. "
            "Do not provide legal advice. Extract obligations, rights, conditions, exceptions, deadlines, money terms, and missing context. "
            "If information is uncertain or depends on other sections, say so explicitly.\n\n"
            "Return JSON only matching the requested extraction fields.\n\n"
            f"Document type: {metadata.document_type}\n"
            f"Governing law: {metadata.governing_law or 'unknown'}\n"
            f"OCR quality: {metadata.ocr_quality}\n"
            f"Document warnings: {metadata.warnings or ['none']}\n"
            f"Source location: {source_location}\n"
            f"Parent heading: {context.parent_heading}\n"
            f"Referenced sections: {context.referenced_sections or ['none']}\n"
            f"Relevant definitions:\n{chr(10).join(definition_lines) if definition_lines else '- none'}\n"
            f"Referenced section text:\n{chr(10).join(referenced_lines) if referenced_lines else '- none'}\n"
            f"Policy focus: {policy_focus}\n"
            f"Policy trigger terms to watch for: {policy_triggers}\n"
            f"Policy review questions:\n{policy_questions}\n\n"
            "Preserve negations, exceptions, deadlines, cross-references, one-sided rights, payment details, and any waiver of remedies or procedural rights.\n\n"
            f"Clause:\n{text}"
        )

    def _detect_policy_type(self, text: str) -> str:
        lowered = text.lower()
        if "indemn" in lowered:
            return "indemnity"
        if "liability" in lowered or "damages" in lowered:
            return "liability"
        if "arbitration" in lowered or "jury trial" in lowered:
            return "arbitration"
        if "terminat" in lowered:
            return "termination"
        if "intellectual property" in lowered or "assign" in lowered or "license" in lowered:
            return "ip"
        if "renew" in lowered:
            return "renewal"
        if "confidential" in lowered:
            return "confidentiality"
        if "payment" in lowered or "fee" in lowered or "invoice" in lowered:
            return "payment"
        return "general"

    def _strip_code_fence(self, text: str) -> str:
        payload = text.strip()
        if payload.startswith("```"):
            payload = payload.split("```", 2)[1]
            if payload.startswith("json"):
                payload = payload[4:]
        return payload.strip()

    def _load_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
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
