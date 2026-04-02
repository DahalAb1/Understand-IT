import re

from .models import ClauseContext, ClauseExtraction, DocumentMetadata, RiskLevel
from .policies import get_clause_policy
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
    "arbitration": ("arbitration", "jury trial", "class action", "dispute"),
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

    def is_available(self) -> bool:
        return True

    def extract_clause(
        self,
        text: str,
        metadata: DocumentMetadata,
        source_location: str,
        context: ClauseContext,
    ) -> ClauseExtraction:
        clause_type = self._classify_clause_type(text)
        policy = get_clause_policy(clause_type)
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
        if context.parent_text:
            missing_context.append("This clause may depend on a parent section for its full meaning.")
        if context.referenced_sections:
            missing_context.append("This clause refers to other sections that may change its effect.")
        if metadata.is_partial:
            missing_context.append("The uploaded document may be incomplete.")
        if metadata.ocr_quality != "good":
            missing_context.append("OCR quality may affect the accuracy of this summary.")
        if policy:
            missing_context.extend(self._policy_warnings(text, policy.review_triggers))

        title = self._title_from_text(text, clause_type, context.parent_heading)
        plain_english = self._plain_english(title, obligations, rights, conditions, exceptions, deadlines, money_terms, context, policy)
        legal_precision_note = None
        if policy and policy.precision_note:
            legal_precision_note = policy.precision_note
        elif conditions or exceptions or context.referenced_sections or context.parent_text:
            legal_precision_note = "Important conditions, hierarchy, exceptions, or cross-references apply to this clause and should be read with the original text."

        questions_to_ask = self._questions(clause_type, money_terms, deadlines, exceptions, context.referenced_sections, policy)

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
        policy = get_clause_policy(clause_type)
        lowered = text.lower()

        if policy:
            level = RiskLevel(policy.risk_level)
            reason = policy.base_reason
            matched = [trigger for trigger in policy.review_triggers if trigger in lowered]
            if matched:
                reason = f"{reason} Trigger terms detected: {', '.join(matched[:3])}."
            return level, reason

        if money_terms or "automatic renewal" in lowered or obligations:
            return RiskLevel.MEDIUM, "This clause contains obligations, payments, or business terms that should be reviewed carefully."
        if exceptions:
            return RiskLevel.MEDIUM, "This clause includes exceptions that may narrow or change the main rule."
        return RiskLevel.LOW, None

    def _title_from_text(self, text: str, clause_type: str, parent_heading: str) -> str:
        first_line = text.splitlines()[0].strip()
        if len(first_line) <= 80:
            return first_line.rstrip(".")
        if parent_heading:
            return parent_heading
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
        context: ClauseContext,
        policy,
    ) -> str:
        parts = [f"This clause is about {title.lower()}."]
        if context.parent_source_location:
            parts.append(f"It sits under {context.parent_source_location}, so it may qualify or narrow a broader parent rule.")
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
        if context.relevant_definitions:
            parts.append("Some defined terms in this clause may change how it should be read.")
        if context.referenced_sections:
            parts.append("It refers to other sections that may affect the full legal meaning.")
        if policy and policy.required_focus:
            parts.append(f"Review should focus on {', '.join(policy.required_focus[:3])}.")
        return " ".join(parts)

    def _questions(
        self,
        clause_type: str,
        money_terms: list[str],
        deadlines: list[str],
        exceptions: list[str],
        referenced_sections: list[str],
        policy,
    ) -> list[str]:
        questions: list[str] = []
        if policy:
            questions.extend(policy.review_questions)
        if money_terms and clause_type != "payment":
            questions.append("Are the payment amounts, triggers, and penalties acceptable?")
        if deadlines:
            questions.append("Can all deadlines in this clause realistically be met?")
        if exceptions:
            questions.append("Do the exceptions significantly narrow the protection or right described here?")
        if referenced_sections:
            questions.append("Do the referenced sections change the meaning of this clause in an important way?")
        return self._unique(questions)

    def _policy_warnings(self, text: str, triggers: tuple[str, ...]) -> list[str]:
        lowered = text.lower()
        matched = [trigger for trigger in triggers if trigger in lowered]
        if not matched:
            return []
        return [f"Review trigger detected: {trigger}." for trigger in matched[:3]]

    def _unique(self, values: list[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            if value and value not in output:
                output.append(value)
        return output
