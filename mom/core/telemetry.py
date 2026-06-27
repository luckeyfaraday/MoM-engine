from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderCallTelemetry:
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class RunTelemetry:
    calls: list[ProviderCallTelemetry] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)
