from typing import Any

from ...domain.models import ClauseContext, DocumentMetadata
from ...domain.policies import get_clause_policy


CLAUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
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
    "additionalProperties": False,
}


def detect_policy_type(text: str) -> str:
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


def build_clause_prompt(
    text: str,
    metadata: DocumentMetadata,
    source_location: str,
    context: ClauseContext,
) -> str:
    policy = get_clause_policy(detect_policy_type(text))
    definition_lines = [
        f'- "{definition.term}": {definition.definition} ({definition.source_location})'
        for definition in context.relevant_definitions
    ]
    referenced_lines = [f"- {entry}" for entry in context.referenced_texts]
    hierarchy_line = " > ".join(context.hierarchy_path) if context.hierarchy_path else source_location
    policy_focus = ", ".join(policy.required_focus) if policy else "general legal meaning"
    policy_questions = "\n".join(f"- {question}" for question in policy.review_questions) if policy else "- none"
    policy_triggers = ", ".join(policy.review_triggers) if policy else "none"

    return (
        "You are a legal document accessibility assistant. "
        "Explain legal clauses in plain language while preserving legal nuance. "
        "Do not provide legal advice. Extract obligations, rights, conditions, exceptions, deadlines, money terms, and missing context. "
        "If information is uncertain or depends on parent sections, sibling clauses, or other sections, say so explicitly.\n\n"
        "Return JSON only matching the requested extraction fields.\n\n"
        f"Document type: {metadata.document_type}\n"
        f"Governing law: {metadata.governing_law or 'unknown'}\n"
        f"OCR quality: {metadata.ocr_quality}\n"
        f"Document warnings: {metadata.warnings or ['none']}\n"
        f"Source location: {source_location}\n"
        f"Hierarchy path: {hierarchy_line}\n"
        f"Parent heading: {context.parent_heading}\n"
        f"Parent source location: {context.parent_source_location or 'none'}\n"
        f"Parent clause text:\n{context.parent_text or '- none'}\n"
        f"Referenced sections: {context.referenced_sections or ['none']}\n"
        f"Relevant definitions:\n{chr(10).join(definition_lines) if definition_lines else '- none'}\n"
        f"Referenced section text:\n{chr(10).join(referenced_lines) if referenced_lines else '- none'}\n"
        f"Policy focus: {policy_focus}\n"
        f"Policy trigger terms to watch for: {policy_triggers}\n"
        f"Policy review questions:\n{policy_questions}\n\n"
        "Preserve negations, exceptions, deadlines, parent-child hierarchy, cross-references, one-sided rights, payment details, and any waiver of remedies or procedural rights.\n\n"
        f"Clause:\n{text}"
    )
