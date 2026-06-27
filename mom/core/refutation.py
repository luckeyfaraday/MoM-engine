from __future__ import annotations

from collections.abc import Iterable

from mom.core.claims import AtomicClaim, RefutationResult


def is_refuted(claim: AtomicClaim, refutations: Iterable[RefutationResult]) -> bool:
    return any(result.claim_id == claim.id and result.refuted for result in refutations)


def surviving_claims(
    *,
    claims: Iterable[AtomicClaim],
    refutations: Iterable[RefutationResult],
) -> list[AtomicClaim]:
    refutation_list = list(refutations)
    return [claim for claim in claims if not is_refuted(claim, refutation_list)]
