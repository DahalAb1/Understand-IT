from fastapi import APIRouter, UploadFile, File, HTTPException
from ...domain.models import SimplificationRequest
from ...domain.ports import SimplifierPort

router = APIRouter()


def create_router(simplifier: SimplifierPort) -> APIRouter:
    @router.post("/simplify")
    async def simplify(file: UploadFile = File(...)):
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        pdf_bytes = await file.read()
        request = SimplificationRequest(pdf_bytes=pdf_bytes)
        result = simplifier.simplify(request)

        return {
            "clauses": [
                {
                    "title": clause.title,
                    "original": clause.original,
                    "simplified": clause.simplified,
                    "risk_level": clause.risk_level.value,
                    "risk_reason": clause.risk_reason,
                }
                for clause in result.clauses
            ]
        }

    return router
