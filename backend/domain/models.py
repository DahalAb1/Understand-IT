from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DocumentMetadata:
    document_type: str
    governing_law: str | None = None
    is_partial: bool = False
    ocr_quality: str = "good"
    warnings: list[str] = field(default_factory=list)


@dataclass
class DefinedTerm:
    term: str
    definition: str
    source_location: str


@dataclass
class ClauseContext:
    parent_heading: str
    referenced_sections: list[str] = field(default_factory=list)
    referenced_texts: list[str] = field(default_factory=list)
    relevant_definitions: list[DefinedTerm] = field(default_factory=list)


@dataclass
class DocumentContext:
    definitions: dict[str, DefinedTerm] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    cross_references: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ClauseSegment:
    id: str
    heading: str
    text: str
    source_location: str
    section_number: str | None = None
    parent_heading: str | None = None
    referenced_sections: list[str] = field(default_factory=list)


@dataclass
class ClauseExtraction:
    title: str
    clause_type: str
    source_text: str
    source_location: str
    defined_terms_used: list[str]
    obligations: list[str]
    rights: list[str]
    conditions: list[str]
    exceptions: list[str]
    deadlines: list[str]
    money_terms: list[str]
    plain_english: str
    legal_precision_note: str | None
    questions_to_ask: list[str]
    risk_level: RiskLevel
    risk_reason: str | None
    confidence: float
    missing_context: list[str]


@dataclass
class Clause:
    title: str
    original: str
    simplified: str
    risk_level: RiskLevel
    risk_reason: str | None
    source_location: str | None = None
    clause_type: str | None = None
    confidence: float | None = None
    plain_english: str | None = None
    legal_precision_note: str | None = None
    what_you_must_do: list[str] = field(default_factory=list)
    what_the_other_side_can_do: list[str] = field(default_factory=list)
    important_exceptions: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    money_terms: list[str] = field(default_factory=list)
    defined_terms_used: list[str] = field(default_factory=list)
    questions_to_ask: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    referenced_sections: list[str] = field(default_factory=list)


@dataclass
class SimplificationRequest:
    pdf_bytes: bytes


@dataclass
class SimplificationResult:
    clauses: list[Clause]
    metadata: DocumentMetadata
