import re

from .models import ClauseExtraction


NEGATION_RE = re.compile(r"\b(not|no|except|unless|subject to|provided that|notwithstanding)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
MONEY_RE = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d{2})?")


class ClauseVerifier:
    def verify(self, extraction: ClauseExtraction) -> tuple[float, list[str]]:
        confidence = extraction.confidence
        warnings: list[str] = []

        source = extraction.source_text
        rendered = " ".join(
            [
                extraction.plain_english,
                extraction.legal_precision_note or "",
                " ".join(extraction.conditions),
                " ".join(extraction.exceptions),
                " ".join(extraction.deadlines),
                " ".join(extraction.money_terms),
            ]
        )

        if NEGATION_RE.search(source) and not NEGATION_RE.search(rendered):
            confidence -= 0.15
            warnings.append("Important legal limitations may have been flattened.")

        if DATE_RE.search(source) and not extraction.deadlines:
            confidence -= 0.1
            warnings.append("The clause may contain dates or timing details that were not extracted.")

        if MONEY_RE.search(source) and not extraction.money_terms:
            confidence -= 0.1
            warnings.append("The clause may contain payment terms that were not extracted.")

        if len(source.split()) > 120 and not extraction.missing_context and not extraction.exceptions and not extraction.conditions:
            confidence -= 0.1
            warnings.append("This clause is dense and may need extra review for hidden conditions.")

        return max(0.0, min(confidence, 1.0)), warnings
