import re

from .models import ClauseContext, ClauseSegment, DefinedTerm, DocumentContext
from .segmenter import extract_cross_references, extract_defined_terms


DEFINITION_PATTERNS = [
    re.compile(r'"(?P<term>[A-Z][A-Za-z0-9\s-]{1,60})"\s+means\s+(?P<definition>.+?)(?:\.|;)', re.IGNORECASE),
    re.compile(r'"(?P<term>[A-Z][A-Za-z0-9\s-]{1,60})"\s+shall mean\s+(?P<definition>.+?)(?:\.|;)', re.IGNORECASE),
]
RELATIVE_SECTION_RE = re.compile(r"\b(this section|this subsection|above|below|herein)\b", re.IGNORECASE)


class DocumentContextBuilder:
    def build(self, segments: list[ClauseSegment]) -> DocumentContext:
        definitions: dict[str, DefinedTerm] = {}
        sections: dict[str, str] = {}
        segments_by_id = {segment.id: segment for segment in segments}
        segment_order = [segment.id for segment in segments]
        cross_references: dict[str, list[str]] = {}

        for index, segment in enumerate(segments):
            if segment.section_number:
                sections[segment.section_number] = segment.text

            cross_references[segment.id] = self._resolve_references(segment, segments, index)

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
            segments_by_id=segments_by_id,
            segment_order=segment_order,
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

        parent_segment = document_context.segments_by_id.get(segment.parent_id) if segment.parent_id else None
        hierarchy_path = self._hierarchy_path(segment, document_context)

        return ClauseContext(
            parent_heading=segment.parent_heading or segment.heading,
            parent_source_location=parent_segment.source_location if parent_segment else None,
            parent_text=parent_segment.text if parent_segment else None,
            hierarchy_path=hierarchy_path,
            referenced_sections=referenced_sections,
            referenced_texts=referenced_texts,
            relevant_definitions=relevant_definitions,
        )

    def _resolve_references(self, segment: ClauseSegment, segments: list[ClauseSegment], index: int) -> list[str]:
        refs = set(extract_cross_references(segment.text))
        lowered = segment.text.lower()

        if "this section" in lowered or "this subsection" in lowered:
            if segment.section_number:
                refs.add(segment.section_number)
            elif segment.parent_section_number:
                refs.add(segment.parent_section_number)

        if "above" in lowered and index > 0:
            previous = segments[index - 1]
            if previous.section_number:
                refs.add(previous.section_number)

        if "below" in lowered and index + 1 < len(segments):
            next_segment = segments[index + 1]
            if next_segment.section_number:
                refs.add(next_segment.section_number)

        if "herein" in lowered and segment.parent_section_number:
            refs.add(segment.parent_section_number)

        return sorted(ref for ref in refs if ref)

    def _hierarchy_path(self, segment: ClauseSegment, document_context: DocumentContext) -> list[str]:
        path: list[str] = []
        current = segment
        while current.parent_id:
            parent = document_context.segments_by_id.get(current.parent_id)
            if not parent:
                break
            path.insert(0, parent.source_location)
            current = parent
        if segment.source_location:
            path.append(segment.source_location)
        return path

    def _find_definition(self, text: str, term: str) -> str | None:
        for pattern in DEFINITION_PATTERNS:
            for match in pattern.finditer(text):
                if match.group("term").strip() == term:
                    return match.group("definition").strip()
        return None
