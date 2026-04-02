import re

from .models import ClauseExtraction, DocumentMetadata, RiskLevel
from .segmenter import extract_defined_terms


CLAUSE_TYPE_PATTERNS = {
    "confidentiality": ("confidential", "non-disclosure", "proprietary"),
    "termination": ("terminate", "termination", "end this agreement"),
    "payment": ("fee", "fees", "payment", "invoice", "late charge"),
    "liability": ("liability", "damages", "indirect", "consequential"),
    "indemnity": ("indemnify", "indemnification", "hold harmless"),
    "governing_law": ("governing law", "laws of", "jurisdiction"),
    "ip": ("intellectual property", "ownership", "assign", "license"),
    "renewal": ("renewal", "auto-renew", "renew automatically"),
}

DEADLINE_RE = re.compile(
    r"\b(?:within|before|after|no later than|at least)\s+\d+\s+(?:day|days|business days|calendar days|months|years)\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d{2})?")
CONDITION_RE = re.compile(r"\b(if|provided that|subject to|unless|conditioned on)\b", re.IGNORECASE)
EXCEPTION_RE = re.compile(r"\b(except|excluding|other than|notwithstanding)\b", re.IGNORECASE)
RIGHT_RE = re.compile(r"\b(may|can|reserve the right|is entitled to)\b", re.IGNORECASE)
OBLIGATION_RE = re.compile(r"\b(shall|must|agrees to|is required to)\b", re.IGNORECASE)


class HeuristicClauseExtractor:
    max_input_length = 6000

    def extract_clause(self, text: str, metadata: DocumentMetadata, source_location: str) -> ClauseExtraction:
        clause_type = self._classify_clause_type(text)
        obligations = self._collect_sentences(text, OBLIGATION_RE)
        rights = self._collect_sentences(text, RIGHT_RE)
        conditions = self._collect_sentences(text, CONDITION_RE)
        exceptions = self._collect_sentences(text, EXCEPTION_RE)
        deadlines = DEADLINE_RE.findall(text)
        money_terms = MONEY_RE.findall(text)
        defined_terms = extract_defined_terms(text)
        risk_level, risk_reason = self._risk(text, clause_type, obligations, exceptions, money_terms)

        missing_context: list[str] = []
        if defined_terms:
            missing_context.append("Defined terms may require review against the document definitions section.")
        if metadata.is_partial:
            missing_context.append("The uploaded document may be incomplete.")
        if metadata.ocr_quality != "good":
            missing_context.append("OCR quality may affect the accuracy of this summary.")

        title = self._title_from_text(text, clause_type)
        plain_english = self._plain_english(title, obligations, rights, conditions, exceptions, deadlines, money_terms)
        legal_precision_note = None
        if conditions or exceptions:
            legal_precision_note = "Important conditions or exceptions apply to this clause and should be read with the original text."

        questions_to_ask = self._questions(clause_type, money_terms, deadlines, exceptions)

        return ClauseExtraction(
            title=title,
            clause_type=clause_type,
            source_text=text,
            source_location=source_location,
            defined_terms_used=defined_terms,
            obligations=obligations,
            rights=rights,
            conditions=conditions,
            exceptions=exceptions,
            deadlines=deadlines,
            money_terms=money_terms,
            plain_english=plain_english,
            legal_precision_note=legal_precision_note,
            questions_to_ask=questions_to_ask,
            risk_level=risk_level,
            risk_reason=risk_reason,
            confidence=0.42,
            missing_context=missing_context,
        )

    def _classify_clause_type(self, text: str) -> str:
        lowered = text.lower()
        best_type = "general"
        best_score = 0
        for clause_type, markers in CLAUSE_TYPE_PATTERNS.items():
            score = sum(1 for marker in markers if marker in lowered)
            if score > best_score:
                best_type = clause_type
                best_score = score
        return best_type

    def _collect_sentences(self, text: str, pattern: re.Pattern[str]) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if pattern.search(sentence)]

    def _risk(
        self,
        text: str,
        clause_type: str,
        obligations: list[str],
        exceptions: list[str],
        money_terms: list[str],
    ) -> tuple[RiskLevel, str | None]:
        lowered = text.lower()
        if clause_type in {"liability", "indemnity", "termination", "ip"}:
            return RiskLevel.HIGH, "This is a high-impact clause that can materially affect rights, liability, or exit terms."
        if money_terms or "automatic renewal" in lowered or obligations:
            return RiskLevel.MEDIUM, "This clause contains obligations, payments, or business terms that should be reviewed carefully."
        if exceptions:
            return RiskLevel.MEDIUM, "This clause includes exceptions that may narrow or change the main rule."
        return RiskLevel.LOW, None

    def _title_from_text(self, text: str, clause_type: str) -> str:
        first_line = text.splitlines()[0].strip()
        if len(first_line) <= 80:
            return first_line.rstrip(".")
        return clause_type.replace("_", " ").title()

    def _plain_english(
        self,
        title: str,
        obligations: list[str],
        rights: list[str],
        conditions: list[str],
        exceptions: list[str],
        deadlines: list[str],
        money_terms: list[str],
    ) -> str:
        parts = [f"This clause is about {title.lower()}."]
        if obligations:
            parts.append("It creates one or more obligations.")
        if rights:
            parts.append("It gives one side one or more rights or options.")
        if deadlines:
            parts.append("It includes timing requirements.")
        if money_terms:
            parts.append("It includes payment or financial terms.")
        if conditions or exceptions:
            parts.append("Its meaning depends on stated conditions or exceptions.")
        return " ".join(parts)

    def _questions(
        self,
        clause_type: str,
        money_terms: list[str],
        deadlines: list[str],
        exceptions: list[str],
    ) -> list[str]:
        questions: list[str] = []
        if clause_type in {"liability", "indemnity", "termination", "ip"}:
            questions.append("Does this clause shift unusually high risk to one side?")
        if money_terms:
            questions.append("Are the payment amounts, triggers, and penalties acceptable?")
        if deadlines:
            questions.append("Can all deadlines in this clause realistically be met?")
        if exceptions:
            questions.append("Do the exceptions significantly narrow the protection or right described here?")
        return questions
