import re

from .models import ClauseSegment


HEADING_RE = re.compile(
    r"^(?P<number>(?:section\s+)?\d+(?:\.\d+)*|(?:\([a-zA-Z0-9]+\))+|[A-Z])[\).\s-]+(?P<title>.+)$",
    re.IGNORECASE,
)
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s,&/-]{4,}$")
DEFINED_TERM_RE = re.compile(r'"([A-Z][A-Za-z0-9\s-]{1,60})"')


class ClauseSegmenter:
    def segment(self, text: str, max_length: int) -> list[ClauseSegment]:
        normalized = self._normalize(text)
        lines = [line.strip() for line in normalized.splitlines()]

        segments: list[ClauseSegment] = []
        current_heading = "Introduction"
        current_lines: list[str] = []
        index = 1

        for line in lines:
            if not line:
                continue

            if self._is_heading(line) and current_lines:
                segments.append(self._build_segment(index, current_heading, current_lines, max_length))
                index += 1
                current_heading = self._clean_heading(line)
                current_lines = []
                continue

            if self._is_heading(line):
                current_heading = self._clean_heading(line)
                continue

            current_lines.append(line)

        if current_lines:
            segments.append(self._build_segment(index, current_heading, current_lines, max_length))

        return self._split_oversized_segments(segments, max_length)

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_heading(self, line: str) -> bool:
        if len(line) > 110:
            return False
        if HEADING_RE.match(line):
            return True
        return bool(ALL_CAPS_RE.match(line))

    def _clean_heading(self, line: str) -> str:
        match = HEADING_RE.match(line)
        if match:
            return match.group("title").strip().title()
        return line.title()

    def _build_segment(self, index: int, heading: str, lines: list[str], max_length: int) -> ClauseSegment:
        text = " ".join(lines).strip()
        if len(text) > max_length * 2:
            text = text[: max_length * 2].rsplit(" ", 1)[0].strip()
        return ClauseSegment(
            id=f"clause-{index}",
            heading=heading,
            text=text,
            source_location=f"Clause {index}",
        )

    def _split_oversized_segments(self, segments: list[ClauseSegment], max_length: int) -> list[ClauseSegment]:
        output: list[ClauseSegment] = []
        for segment in segments:
            if len(segment.text) <= max_length:
                output.append(segment)
                continue

            paragraphs = [part.strip() for part in re.split(r"(?<=\.)\s+", segment.text) if part.strip()]
            chunk: list[str] = []
            chunk_index = 1

            for paragraph in paragraphs:
                candidate = " ".join(chunk + [paragraph]).strip()
                if chunk and len(candidate) > max_length:
                    output.append(
                        ClauseSegment(
                            id=f"{segment.id}-{chunk_index}",
                            heading=segment.heading,
                            text=" ".join(chunk).strip(),
                            source_location=f"{segment.source_location}.{chunk_index}",
                        )
                    )
                    chunk = [paragraph]
                    chunk_index += 1
                else:
                    chunk.append(paragraph)

            if chunk:
                output.append(
                    ClauseSegment(
                        id=f"{segment.id}-{chunk_index}",
                        heading=segment.heading,
                        text=" ".join(chunk).strip(),
                        source_location=f"{segment.source_location}.{chunk_index}",
                    )
                )

        return output


def extract_defined_terms(text: str) -> list[str]:
    terms = {term.strip() for term in DEFINED_TERM_RE.findall(text)}
    return sorted(term for term in terms if term)
