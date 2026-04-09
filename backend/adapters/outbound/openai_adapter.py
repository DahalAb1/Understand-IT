import json
import os
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ...domain.models import ClauseContext, ClauseExtraction, DocumentMetadata, RiskLevel
from .structured_clause_extraction import CLAUSE_SCHEMA, build_clause_prompt


class OpenAIAdapter:
    max_input_length = 4000

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = None

        if self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)

    def is_available(self) -> bool:
        return self.client is not None

    def extract_clause(
        self,
        text: str,
        metadata: DocumentMetadata,
        source_location: str,
        context: ClauseContext,
    ) -> ClauseExtraction:
        if not self.is_available():
            raise RuntimeError(
                "The configured model provider is unavailable. Set OPENAI_API_KEY and install the OpenAI SDK to enable clause extraction."
            )

        prompt = build_clause_prompt(text, metadata, source_location, context)
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
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    temperature=0.2,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "clause_extraction",
                            "schema": CLAUSE_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                payload = getattr(response, "output_text", None)
                if not payload:
                    raise RuntimeError("The model returned an empty response.")
                return self._load_json(payload)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError("The model could not produce a reliable structured extraction.") from last_error

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
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))
