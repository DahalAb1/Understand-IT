import hashlib
import json
import re

from .context_builder import DocumentContextBuilder
from .heuristics import HeuristicClauseExtractor
from .models import Clause, ClauseExtraction, DocumentContext, DocumentMetadata, RiskLevel, SimplificationRequest, SimplificationResult
from .ports import CachePort, ModelPort, PdfReaderPort
from .segmenter import ClauseSegment, ClauseSegmenter, extract_defined_terms
from .verifier import ClauseVerifier


DOCUMENT_PATTERNS = {
    "nda": ("confidential information", "non-disclosure", "receiving party"),
    "employment_agreement": ("employee", "employer", "employment", "termination"),
    "lease": ("landlord", "tenant", "premises", "rent"),
    "saas_terms": ("service", "subscription", "license", "uptime"),
    "contractor_agreement": ("contractor", "independent contractor", "statement of work"),
    "privacy_policy": ("personal data", "privacy policy", "cookies", "data subject"),
}

HIGH_RISK_KEYWORDS = (
    "indemnify",
    "limitation of liability",
    "consequential damages",
    "arbitration",
    "terminate",
    "termination",
    "governing law",
    "auto-renew",
    "intellectual property",
    "assign all right",
    "personal guarantee",
)

GOVERNING_LAW_RE = re.compile(r"governed by the laws of ([A-Za-z ,]+)", re.IGNORECASE)


class SimplifierService:
    def __init__(self, reader: PdfReaderPort, model: ModelPort, cache: CachePort):
        self.reader = reader
        self.model = model
        self.cache = cache
        self.segmenter = ClauseSegmenter()
        self.context_builder = DocumentContextBuilder()
        self.verifier = ClauseVerifier()
        self.fallback_model = HeuristicClauseExtractor()

    def simplify(self, request: SimplificationRequest) -> SimplificationResult:
        text = self.reader.extract(request.pdf_bytes)
        if not text.strip():
            raise ValueError("No readable text was found in the uploaded PDF.")

        metadata = self._classify_document(text)
        segments = self.segmenter.segment(text, self.model.max_input_length)
        document_context = self.context_builder.build(segments)

        clauses: list[Clause] = []
        for segment in segments:
            cache_key = self._cache_key(segment.text, metadata.document_type)
            cached = self.cache.get(cache_key)

            if cached:
                clause = self._clause_from_cache(json.loads(cached), segment.text)
            else:
                extraction = self._extract_segment(segment, metadata, document_context)
                extraction.defined_terms_used = extraction.defined_terms_used or extract_defined_terms(segment.text)
                confidence, warnings = self.verifier.verify(extraction)
                extraction.confidence = confidence
                extraction.missing_context = self._merge_unique(extraction.missing_context, warnings)
                clause = self._render_clause(extraction, segment)
                self.cache.set(cache_key, json.dumps(self._clause_to_cache_payload(clause)))

            clauses.append(clause)

        return SimplificationResult(clauses=clauses, metadata=metadata)

    def _extract_segment(
        self,
        segment: ClauseSegment,
        metadata: DocumentMetadata,
        document_context: DocumentContext,
    ) -> ClauseExtraction:
        context = self.context_builder.context_for(segment, document_context)
        prefer_primary = self.model.is_available() and self._should_use_primary_model(segment.text)

        if prefer_primary:
            try:
                extraction = self.model.extract_clause(segment.text, metadata, segment.source_location, context)
                extraction.missing_context = self._merge_unique(
                    extraction.missing_context,
                    self._context_warnings(metadata, extraction.defined_terms_used, context.referenced_sections),
                )
                return extraction
            except RuntimeError:
                pass

        extraction = self.fallback_model.extract_clause(segment.text, metadata, segment.source_location, context)
        extraction.missing_context = self._merge_unique(
            extraction.missing_context,
            self._context_warnings(metadata, extraction.defined_terms_used, context.referenced_sections),
        )
        return extraction

    def _should_use_primary_model(self, text: str) -> bool:
        lowered = text.lower()
        if any(keyword in lowered for keyword in HIGH_RISK_KEYWORDS):
            return True
        if len(text) > 900:
            return True
        if lowered.count("provided that") + lowered.count("subject to") + lowered.count("unless") >= 2:
            return True
        return False

    def _context_warnings(
        self,
        metadata: DocumentMetadata,
        defined_terms: list[str],
        referenced_sections: list[str],
    ) -> list[str]:
        warnings = list(metadata.warnings)
        if defined_terms:
            warnings.append("Defined terms may depend on other sections of the document.")
        if referenced_sections:
            warnings.append("This clause refers to other sections that may materially affect its meaning.")
        return warnings

    def _classify_document(self, text: str) -> DocumentMetadata:
        lowered = text.lower()
        document_type = "other"
        best_score = 0

        for candidate, markers in DOCUMENT_PATTERNS.items():
            score = sum(1 for marker in markers if marker in lowered)
            if score > best_score:
                best_score = score
                document_type = candidate

        warnings: list[str] = []
        is_partial = False

        if "page 1 of" in lowered and "signature" not in lowered:
            warnings.append("The document may be incomplete.")
            is_partial = True

        if len(text.split()) < 80:
            warnings.append("Very little text was extracted from the PDF.")

        ocr_quality = "good"
        if any(token in text for token in ["�", "  ", "...."]) or len(re.findall(r"[A-Za-z]", text)) < len(text) * 0.5:
            ocr_quality = "needs_review"
            warnings.append("The extracted text may contain OCR issues.")

        governing_law_match = GOVERNING_LAW_RE.search(text)
        governing_law = governing_law_match.group(1).strip(" .,") if governing_law_match else None

        return DocumentMetadata(
            document_type=document_type,
            governing_law=governing_law,
            is_partial=is_partial,
            ocr_quality=ocr_quality,
            warnings=warnings,
        )

    def _render_clause(self, extraction: ClauseExtraction, segment: ClauseSegment) -> Clause:
        simplified = extraction.plain_english
        if extraction.legal_precision_note:
            simplified = f"{simplified}\n\nPrecision note: {extraction.legal_precision_note}"

        risk_reason = extraction.risk_reason
        if extraction.missing_context:
            note = " ".join(extraction.missing_context)
            risk_reason = f"{risk_reason} {note}".strip() if risk_reason else note

        return Clause(
            title=extraction.title,
            original=extraction.source_text,
            simplified=simplified,
            risk_level=extraction.risk_level,
            risk_reason=risk_reason,
            source_location=extraction.source_location,
            clause_type=extraction.clause_type,
            confidence=extraction.confidence,
            plain_english=extraction.plain_english,
            legal_precision_note=extraction.legal_precision_note,
            what_you_must_do=extraction.obligations,
            what_the_other_side_can_do=extraction.rights,
            important_exceptions=extraction.exceptions,
            deadlines=extraction.deadlines,
            money_terms=extraction.money_terms,
            defined_terms_used=extraction.defined_terms_used,
            questions_to_ask=extraction.questions_to_ask,
            missing_context=extraction.missing_context,
            referenced_sections=segment.referenced_sections,
        )

    def _cache_key(self, text: str, document_type: str) -> str:
        return hashlib.sha256(f"{document_type}:{text}".encode()).hexdigest()

    def _clause_to_cache_payload(self, clause: Clause) -> dict:
        return {
            "title": clause.title,
            "original": clause.original,
            "simplified": clause.simplified,
            "risk_level": clause.risk_level.value,
            "risk_reason": clause.risk_reason,
            "source_location": clause.source_location,
            "clause_type": clause.clause_type,
            "confidence": clause.confidence,
            "plain_english": clause.plain_english,
            "legal_precision_note": clause.legal_precision_note,
            "what_you_must_do": clause.what_you_must_do,
            "what_the_other_side_can_do": clause.what_the_other_side_can_do,
            "important_exceptions": clause.important_exceptions,
            "deadlines": clause.deadlines,
            "money_terms": clause.money_terms,
            "defined_terms_used": clause.defined_terms_used,
            "questions_to_ask": clause.questions_to_ask,
            "missing_context": clause.missing_context,
            "referenced_sections": clause.referenced_sections,
        }

    def _clause_from_cache(self, data: dict, fallback_original: str) -> Clause:
        return Clause(
            title=data["title"],
            original=data.get("original", fallback_original),
            simplified=data["simplified"],
            risk_level=RiskLevel(data["risk_level"]),
            risk_reason=data.get("risk_reason"),
            source_location=data.get("source_location"),
            clause_type=data.get("clause_type"),
            confidence=data.get("confidence"),
            plain_english=data.get("plain_english"),
            legal_precision_note=data.get("legal_precision_note"),
            what_you_must_do=data.get("what_you_must_do", []),
            what_the_other_side_can_do=data.get("what_the_other_side_can_do", []),
            important_exceptions=data.get("important_exceptions", []),
            deadlines=data.get("deadlines", []),
            money_terms=data.get("money_terms", []),
            defined_terms_used=data.get("defined_terms_used", []),
            questions_to_ask=data.get("questions_to_ask", []),
            missing_context=data.get("missing_context", []),
            referenced_sections=data.get("referenced_sections", []),
        )

    def _merge_unique(self, base: list[str], extra: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*base, *extra]:
            cleaned = item.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged
