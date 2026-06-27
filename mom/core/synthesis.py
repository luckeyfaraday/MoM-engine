from __future__ import annotations

from mom.core.claims import AtomicClaim


def synthesize_survivors(claims: list[AtomicClaim]) -> str:
    if not claims:
        return "No claims survived refutation."
    return "\n".join(f"- {claim.text}" for claim in claims)
