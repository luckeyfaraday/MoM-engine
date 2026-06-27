from __future__ import annotations

import json
from typing import Any

from mom.api.messages import messages_to_text
from mom.api.schemas import ChatMessage
from mom.core.architecture import InternalRole
from mom.core.claims import AtomicClaim, RefutationResult
from mom.core.synthesis import synthesize_survivors
from mom.core.tool_calls import first_function_tool_name, forced_tool_name


class MockProvider:
    """Deterministic provider used by tests and local development."""

    async def generate_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        messages: list[ChatMessage],
    ) -> list[AtomicClaim]:
        user_text = messages_to_text(messages, role="user")
        topic = user_text.strip() or "the request"
        return [
            AtomicClaim(
                id=f"{role.name}-survives",
                text=f"Supported answer material: {topic[:80]}",
                source_role=role.name,
                evidence=["mock-input"],
                confidence=0.8,
            ),
            AtomicClaim(
                id=f"{role.name}-refuted",
                text="unsupported mock claim",
                source_role=role.name,
                evidence=[],
                confidence=0.2,
            ),
        ]

    async def refute_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        claims: list[AtomicClaim],
        messages: list[ChatMessage],
    ) -> list[RefutationResult]:
        return [
            RefutationResult(
                claim_id=claim.id,
                refuter_role=role.name,
                refuted="unsupported" in claim.text.lower() or claim.confidence < 0.5,
                reason="Mock refuter rejects unsupported or low-confidence claims.",
            )
            for claim in claims
        ]

    async def synthesize(
        self,
        *,
        model: str,
        role: InternalRole | None,
        surviving_claims: list[AtomicClaim],
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> str:
        if tools and tool_choice != "none":
            tool_name = forced_tool_name(tool_choice) or first_function_tool_name(tools)
            if tool_name and (tool_choice == "required" or forced_tool_name(tool_choice)):
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": tool_name,
                                "arguments": {"request": messages_to_text(messages, role="user")},
                            }
                        ]
                    },
                    separators=(",", ":"),
                )
        return synthesize_survivors(surviving_claims)
