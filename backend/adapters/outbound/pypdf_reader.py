import io

from pypdf import PdfReader


class PypdfReaderAdapter:
    def extract(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            content = page.extract_text() or ""
            if content.strip():
                pages.append(content.strip())
        return "\n\n".join(pages)
