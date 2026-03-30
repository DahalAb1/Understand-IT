import io
from pypdf import PdfReader


class PypdfReaderAdapter:
    def extract(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return " ".join(
            page.extract_text() or "" for page in reader.pages
        )
