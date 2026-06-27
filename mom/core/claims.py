from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtomicClaim:
    id: str
    text: str
    source_role: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class RefutationResult:
    claim_id: str
    refuter_role: str
    refuted: bool
    reason: str
