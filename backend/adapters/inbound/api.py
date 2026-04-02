from fastapi import APIRouter, File, HTTPException, UploadFile

from ...domain.models import SimplificationRequest
from ...domain.ports import SimplifierPort

router = APIRouter()


def create_router(simplifier: SimplifierPort) -> APIRouter:
    @router.post("/simplify")
    async def simplify(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        pdf_bytes = await file.read()
        request = SimplificationRequest(pdf_bytes=pdf_bytes)

        try:
            result = simplifier.simplify(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="The document could not be processed.") from exc

        return {
            "metadata": {
                "document_type": result.metadata.document_type,
                "governing_law": result.metadata.governing_law,
                "is_partial": result.metadata.is_partial,
                "ocr_quality": result.metadata.ocr_quality,
                "warnings": result.metadata.warnings,
            },
            "summary": {
                "plain_language_overview": result.summary.plain_language_overview,
                "total_clauses": result.summary.total_clauses,
                "risk_counts": result.summary.risk_counts,
                "top_risks": result.summary.top_risks,
                "key_obligations": result.summary.key_obligations,
                "key_deadlines": result.summary.key_deadlines,
                "key_money_terms": result.summary.key_money_terms,
                "sections_requiring_review": result.summary.sections_requiring_review,
            },
            "clauses": [
                {
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
                for clause in result.clauses
            ],
        }

    return router
