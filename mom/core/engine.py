from __future__ import annotations

from typing import Any

from mom.api.schemas import ChatMessage
from mom.core.architecture import DEFAULT_ARCHITECTURE, InternalRole, MoMResult
from mom.core.refutation import surviving_claims
from mom.core.synthesis import synthesize_survivors
from mom.core.telemetry import RunTelemetry
from mom.core.tool_calls import parse_tool_decision
from mom.providers.base import Provider, ProviderCallError


class MoMEngine:
    """Single internal Mixture-of-Models execution path for chat requests."""

    def __init__(
        self,
        *,
        provider: Provider,
        architecture: tuple[InternalRole, ...] = DEFAULT_ARCHITECTURE,
    ) -> None:
        self.provider = provider
        self.architecture = architecture

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> MoMResult:
        direct_chat = getattr(self.provider, "chat_turn", None)
        if tools and tool_choice != "none" and direct_chat:
            final_answer = await direct_chat(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            final_answer, tool_calls = parse_tool_decision(
                final_answer,
                tools=tools,
                tool_choice=tool_choice,
            )
            return MoMResult(
                final_answer=final_answer,
                tool_calls=tool_calls,
                telemetry=RunTelemetry(),
            )

        claims = []
        refutations = []

        # A single throttled/failing leg must not sink the whole response. Skip
        # proposer and refuter legs that error (e.g. an upstream rate limit) and
        # carry on with whatever the surviving legs produced.
        for role in self._roles("proposer"):
            try:
                claims.extend(await self.provider.generate_claims(model=model, role=role, messages=messages))
            except ProviderCallError:
                continue

        for role in self._roles("refuter"):
            try:
                refutations.extend(
                    await self.provider.refute_claims(
                        model=model,
                        role=role,
                        claims=claims,
                        messages=messages,
                    )
                )
            except ProviderCallError:
                continue

        survivors = surviving_claims(claims=claims, refutations=refutations)
        final_answer = await self.provider.synthesize(
            model=model,
            role=self._synthesizer_role(),
            surviving_claims=survivors,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        final_answer, tool_calls = parse_tool_decision(
            final_answer,
            tools=tools,
            tool_choice=tool_choice,
        )

        if not final_answer and not tool_calls:
            final_answer = synthesize_survivors(survivors)

        return MoMResult(
            final_answer=final_answer,
            tool_calls=tool_calls,
            claims=claims,
            refutations=refutations,
            surviving_claims=survivors,
            telemetry=RunTelemetry(),
        )

    def _roles(self, kind: str) -> list[InternalRole]:
        return [role for role in self.architecture if role.kind == kind]

    def _synthesizer_role(self) -> InternalRole | None:
        return next((role for role in self.architecture if role.kind == "synthesizer"), None)
