import re

from .models import ClauseContext, ClauseSegment, DefinedTerm, DocumentContext
from .segmenter import extract_cross_references, extract_defined_terms


DEFINITION_PATTERNS = [
    re.compile(r'"(?P<term>[A-Z][A-Za-z0-9\s-]{1,60})"\s+means\s+(?P<definition>.+?)(?:\.|;)', re.IGNORECASE),
    re.compile(r'"(?P<term>[A-Z][A-Za-z0-9\s-]{1,60})"\s+shall mean\s+(?P<definition>.+?)(?:\.|;)', re.IGNORECASE),
]


class DocumentContextBuilder:
    def build(self, segments: list[ClauseSegment]) -> DocumentContext:
        definitions: dict[str, DefinedTerm] = {}
        sections: dict[str, str] = {}
        cross_references: dict[str, list[str]] = {}

        for segment in segments:
            if segment.section_number:
                sections[segment.section_number] = segment.text

            cross_references[segment.id] = extract_cross_references(segment.text)

            for term in extract_defined_terms(segment.text):
                if term not in definitions:
                    definition = self._find_definition(segment.text, term)
                    if definition:
                        definitions[term] = DefinedTerm(
                            term=term,
                            definition=definition,
                            source_location=segment.source_location,
                        )

        return DocumentContext(
            definitions=definitions,
            sections=sections,
            cross_references=cross_references,
        )

    def context_for(self, segment: ClauseSegment, document_context: DocumentContext) -> ClauseContext:
        referenced_sections = document_context.cross_references.get(segment.id, [])
        referenced_texts = [
            f"Section {section}: {document_context.sections[section]}"
            for section in referenced_sections
            if section in document_context.sections
        ]

        terms = extract_defined_terms(segment.text)
        relevant_definitions = [
            document_context.definitions[term]
            for term in terms
            if term in document_context.definitions
        ]

        return ClauseContext(
            parent_heading=segment.parent_heading or segment.heading,
            referenced_sections=referenced_sections,
            referenced_texts=referenced_texts,
            relevant_definitions=relevant_definitions,
        )

    def _find_definition(self, text: str, term: str) -> str | None:
        for pattern in DEFINITION_PATTERNS:
            for match in pattern.finditer(text):
                if match.group("term").strip() == term:
                    return match.group("definition").strip()
        return None
