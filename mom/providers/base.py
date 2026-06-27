from __future__ import annotations

from typing import Protocol
from typing import Any

from mom.api.schemas import ChatMessage
from mom.core.architecture import InternalRole
from mom.core.claims import AtomicClaim, RefutationResult


class ProviderCallError(RuntimeError):
    """Raised when an upstream model provider cannot complete a request."""


class Provider(Protocol):
    async def generate_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        messages: list[ChatMessage],
    ) -> list[AtomicClaim]:
        ...

    async def refute_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        claims: list[AtomicClaim],
        messages: list[ChatMessage],
    ) -> list[RefutationResult]:
        ...

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
        ...
