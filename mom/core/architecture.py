from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mom.core.claims import AtomicClaim, RefutationResult
from mom.core.telemetry import RunTelemetry
from mom.core.tool_calls import ToolCall

InternalRoleKind = Literal["proposer", "refuter", "synthesizer"]


@dataclass(frozen=True)
class InternalRole:
    name: str
    kind: InternalRoleKind
    instructions: str = ""


@dataclass(frozen=True)
class MoMResult:
    final_answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    claims: list[AtomicClaim] = field(default_factory=list)
    refutations: list[RefutationResult] = field(default_factory=list)
    surviving_claims: list[AtomicClaim] = field(default_factory=list)
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)


DEFAULT_ARCHITECTURE: tuple[InternalRole, ...] = (
    InternalRole(
        name="proposer",
        kind="proposer",
        instructions="Produce atomic, falsifiable claims that answer the user request.",
    ),
    InternalRole(
        name="cross_checker",
        kind="proposer",
        instructions="Produce independent claims and flag uncertainty.",
    ),
    InternalRole(
        name="alternate_proposer",
        kind="proposer",
        instructions="Produce a third independent pass focused on missing considerations.",
    ),
    InternalRole(
        name="adversarial_refuter",
        kind="refuter",
        instructions="Reject unsupported, contradictory, or low-confidence claims.",
    ),
    InternalRole(
        name="final_synthesizer",
        kind="synthesizer",
        instructions="Answer using only claims that survived refutation.",
    ),
)
