import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

from ...domain.models import ExtractedDocument


class PypdfReaderAdapter:
    def extract(self, pdf_bytes: bytes) -> ExtractedDocument:
        text_result = self._extract_text(pdf_bytes)
        if self._looks_usable(text_result.text):
            return text_result

        ocr_result = self._extract_with_ocr(pdf_bytes)
        if ocr_result and self._looks_usable(ocr_result.text):
            merged_warnings = [*text_result.warnings, *ocr_result.warnings]
            return ExtractedDocument(
                text=ocr_result.text,
                extraction_method=ocr_result.extraction_method,
                ocr_attempted=True,
                ocr_available=True,
                warnings=merged_warnings,
            )

        warnings = list(text_result.warnings)
        if ocr_result is None:
            warnings.append("OCR fallback is unavailable in this environment. Install tesseract and pdftoppm for scanned PDFs.")
        else:
            warnings.extend(ocr_result.warnings)

        return ExtractedDocument(
            text=text_result.text,
            extraction_method=text_result.extraction_method,
            ocr_attempted=True,
            ocr_available=ocr_result is not None,
            warnings=warnings,
        )

    def _extract_text(self, pdf_bytes: bytes) -> ExtractedDocument:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            content = page.extract_text() or ""
            if content.strip():
                pages.append(content.strip())

        text = "\n\n".join(pages)
        warnings: list[str] = []
        if not text.strip():
            warnings.append("No embedded PDF text was extracted.")
        elif len(text.split()) < 40:
            warnings.append("Very little embedded PDF text was extracted.")

        return ExtractedDocument(
            text=text,
            extraction_method="embedded_text",
            ocr_attempted=False,
            ocr_available=self._ocr_available(),
            warnings=warnings,
        )

    def _extract_with_ocr(self, pdf_bytes: bytes) -> ExtractedDocument | None:
        if not self._ocr_available():
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            image_prefix = str(Path(tmpdir) / "page")

            raster = subprocess.run(
                ["pdftoppm", "-png", str(pdf_path), image_prefix],
                capture_output=True,
                text=True,
            )
            if raster.returncode != 0:
                return ExtractedDocument(
                    text="",
                    extraction_method="ocr",
                    ocr_attempted=True,
                    ocr_available=True,
                    warnings=["OCR rasterization failed."],
                )

            pages = sorted(Path(tmpdir).glob("page-*.png"))
            texts: list[str] = []
            for page in pages:
                ocr = subprocess.run(
                    ["tesseract", str(page), "stdout"],
                    capture_output=True,
                    text=True,
                )
                if ocr.returncode == 0 and ocr.stdout.strip():
                    texts.append(ocr.stdout.strip())

            extracted = "\n\n".join(texts).strip()
            warnings: list[str] = []
            if not extracted:
                warnings.append("OCR did not recover readable text from the PDF.")

            return ExtractedDocument(
                text=extracted,
                extraction_method="ocr",
                ocr_attempted=True,
                ocr_available=True,
                warnings=warnings,
            )

    def _ocr_available(self) -> bool:
        return shutil.which("tesseract") is not None and shutil.which("pdftoppm") is not None

    def _looks_usable(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped.split()) < 40:
            return False
        alpha_chars = sum(1 for char in stripped if char.isalpha())
        return alpha_chars >= max(20, int(len(stripped) * 0.35))
