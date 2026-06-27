import asyncio

from mom.api.schemas import ChatMessage
from mom.core.engine import MoMEngine
from mom.providers.mock import MockProvider


def test_claim_lifecycle_generates_refutes_and_survives() -> None:
    result = asyncio.run(_run_engine())

    assert result.claims
    assert any(refutation.refuted for refutation in result.refutations)
    assert result.surviving_claims
    assert "unsupported mock claim" not in result.final_answer
    assert result.final_answer.startswith("- Supported answer material:")


async def _run_engine():
    engine = MoMEngine(provider=MockProvider())
    return await engine.chat(
        model="mom-chat",
        messages=[ChatMessage(role="user", content="Check error handling around file parsing.")],
    )
