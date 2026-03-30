from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Clause:
    title: str
    original: str
    simplified: str
    risk_level: RiskLevel
    risk_reason: str | None


@dataclass
class ModelOutput:
    title: str
    simplified: str
    risk_level: RiskLevel
    risk_reason: str | None


@dataclass
class SimplificationRequest:
    pdf_bytes: bytes


@dataclass
class SimplificationResult:
    clauses: list[Clause]
