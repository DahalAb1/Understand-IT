import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.adapters.outbound.registry import ADAPTER_REGISTRY, build_model_adapter
from backend.config import Settings, load_settings
from backend.domain.models import ClauseContext, DefinedTerm, DocumentMetadata


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class EvaluationResult:
    fixture_id: str
    passed: bool
    checks_passed: int
    checks_total: int
    details: list[str]
    output: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable clause extraction checks against the configured model provider."
    )
    parser.add_argument(
        "--fixtures-dir",
        default=str(FIXTURES_DIR),
        help="Directory containing fixture JSON files.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(ADAPTER_REGISTRY),
        help="Override MODEL_PROVIDER for this run.",
    )
    parser.add_argument(
        "--model",
        help="Override the provider model for this run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON.",
    )
    return parser.parse_args()


def build_adapter(settings: Settings, provider_override: str | None, model_override: str | None):
    if provider_override:
        settings = replace(settings, model_provider=provider_override)
    return build_model_adapter(settings, model_override=model_override)


def load_fixtures(fixtures_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted(fixtures_dir.glob("*.json"))
    ]


def build_metadata(payload: dict[str, Any]) -> DocumentMetadata:
    metadata = payload.get("metadata", {})
    return DocumentMetadata(
        document_type=metadata.get("document_type", "other"),
        governing_law=metadata.get("governing_law"),
        is_partial=metadata.get("is_partial", False),
        ocr_quality=metadata.get("ocr_quality", "good"),
        extraction_method=metadata.get("extraction_method", "text"),
        ocr_attempted=metadata.get("ocr_attempted", False),
        ocr_available=metadata.get("ocr_available", False),
        warnings=metadata.get("warnings", []),
    )


def build_context(payload: dict[str, Any]) -> ClauseContext:
    context = payload.get("context", {})
    definitions = [
        DefinedTerm(
            term=entry["term"],
            definition=entry["definition"],
            source_location=entry.get("source_location", "Definitions"),
        )
        for entry in context.get("relevant_definitions", [])
    ]
    return ClauseContext(
        parent_heading=context.get("parent_heading", "General"),
        parent_source_location=context.get("parent_source_location"),
        parent_text=context.get("parent_text"),
        hierarchy_path=context.get("hierarchy_path", []),
        referenced_sections=context.get("referenced_sections", []),
        referenced_texts=context.get("referenced_texts", []),
        relevant_definitions=definitions,
    )


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def output_payload(extraction) -> dict[str, Any]:
    return {
        "title": extraction.title,
        "clause_type": extraction.clause_type,
        "risk_level": extraction.risk_level.value,
        "risk_reason": extraction.risk_reason or "",
        "plain_english": extraction.plain_english,
        "legal_precision_note": extraction.legal_precision_note or "",
        "defined_terms_used": extraction.defined_terms_used,
        "obligations": extraction.obligations,
        "rights": extraction.rights,
        "conditions": extraction.conditions,
        "exceptions": extraction.exceptions,
        "deadlines": extraction.deadlines,
        "money_terms": extraction.money_terms,
        "questions_to_ask": extraction.questions_to_ask,
        "missing_context": extraction.missing_context,
        "confidence": extraction.confidence,
    }


def evaluate_fixture(adapter, fixture: dict[str, Any]) -> EvaluationResult:
    metadata = build_metadata(fixture)
    context = build_context(fixture)
    extraction = adapter.extract_clause(
        text=fixture["text"],
        metadata=metadata,
        source_location=fixture.get("source_location", "Unknown"),
        context=context,
    )
    output = output_payload(extraction)
    expected = fixture.get("expected", {})

    checks_passed = 0
    checks_total = 0
    details: list[str] = []

    if "risk_level" in expected:
        checks_total += 1
        actual = output["risk_level"]
        wanted = expected["risk_level"]
        passed = actual == wanted
        checks_passed += int(passed)
        details.append(f"risk_level expected={wanted} actual={actual} {'PASS' if passed else 'FAIL'}")

    if "clause_type" in expected:
        checks_total += 1
        actual = output["clause_type"]
        wanted = expected["clause_type"]
        passed = actual == wanted
        checks_passed += int(passed)
        details.append(f"clause_type expected={wanted} actual={actual} {'PASS' if passed else 'FAIL'}")

    searchable_sections = {
        "plain_english": normalize_text(output["plain_english"]),
        "risk_reason": normalize_text(output["risk_reason"]),
        "legal_precision_note": normalize_text(output["legal_precision_note"]),
        "missing_context": normalize_text(" ".join(output["missing_context"])),
        "obligations": normalize_text(" ".join(output["obligations"])),
        "rights": normalize_text(" ".join(output["rights"])),
        "deadlines": normalize_text(" ".join(output["deadlines"])),
        "money_terms": normalize_text(" ".join(output["money_terms"])),
        "questions_to_ask": normalize_text(" ".join(output["questions_to_ask"])),
        "defined_terms_used": normalize_text(" ".join(output["defined_terms_used"])),
    }

    for rule in expected.get("contains", []):
        checks_total += 1
        field = rule["field"]
        phrase = normalize_text(rule["phrase"])
        haystack = searchable_sections.get(field, "")
        passed = phrase in haystack
        checks_passed += int(passed)
        details.append(f"{field} contains='{rule['phrase']}' {'PASS' if passed else 'FAIL'}")

    return EvaluationResult(
        fixture_id=fixture["id"],
        passed=checks_passed == checks_total,
        checks_passed=checks_passed,
        checks_total=checks_total,
        details=details,
        output=output,
    )


def main() -> int:
    load_dotenv()
    args = parse_args()
    settings = load_settings()
    adapter = build_adapter(settings, args.provider, args.model)

    if not adapter.is_available():
        print("Configured model provider is unavailable. Check your API key and installed SDK.", file=sys.stderr)
        return 1

    fixtures = load_fixtures(Path(args.fixtures_dir))
    if not fixtures:
        print("No evaluation fixtures found.", file=sys.stderr)
        return 1

    provider_name = args.provider or settings.model_provider
    model_name = getattr(adapter, "model_name", args.model or "unknown")

    results = [evaluate_fixture(adapter, fixture) for fixture in fixtures]
    passed = sum(1 for result in results if result.passed)
    total_checks = sum(result.checks_total for result in results)
    passed_checks = sum(result.checks_passed for result in results)

    report = {
        "provider": provider_name,
        "model": model_name,
        "fixtures_passed": passed,
        "fixtures_total": len(results),
        "checks_passed": passed_checks,
        "checks_total": total_checks,
        "results": [
            {
                "fixture_id": result.fixture_id,
                "passed": result.passed,
                "checks_passed": result.checks_passed,
                "checks_total": result.checks_total,
                "details": result.details,
                "output": result.output,
            }
            for result in results
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if passed == len(results) else 2

    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print(f"Fixtures passed: {passed}/{len(results)}")
    print(f"Checks passed: {passed_checks}/{total_checks}")
    print("")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.fixture_id} ({result.checks_passed}/{result.checks_total})")
        for detail in result.details:
            print(f"  - {detail}")
        print("")

    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
