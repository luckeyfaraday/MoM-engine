from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from mom.api.messages import content_to_text
from mom.api.schemas import ChatMessage
from mom.core.architecture import InternalRole
from mom.core.claims import AtomicClaim, RefutationResult
from mom.providers.base import ProviderCallError

DEFAULT_TOOL_JUDGE_TEMPLATE = """\
You are the judge in a multi-model fusion pipeline operating inside a
tool-using agent. The same conversation and the same set of tools were given to
a panel of {response_count} independent models. Each panelist independently
proposed the next step: a tool call or a direct answer.

Decide the single best next step and produce it yourself:
- If acting is warranted, call exactly one tool with complete arguments.
- If no tool is needed, write the final answer directly.

Judge by correctness and the conversation's actual goal, not by majority vote.
Do not mention the panel, the other models, or that synthesis took place.
"""


class OpenAICompatibleProvider:
    """Provider for upstream APIs with OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        proposer_models: Sequence[str] = ("~openai/gpt-latest",),
        refuter_model: str | None = None,
        synthesizer_model: str | None = None,
        app_referer: str | None = None,
        app_title: str | None = None,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not proposer_models:
            raise ValueError("at least one proposer model is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proposer_models = tuple(proposer_models)
        self.refuter_model = refuter_model or self.proposer_models[0]
        self.synthesizer_model = synthesizer_model or self.proposer_models[0]
        self.app_referer = app_referer
        self.app_title = app_title
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        proposer_models = _csv(os.getenv("MOM_PROPOSER_MODELS")) or ("~openai/gpt-latest",)
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            proposer_models=proposer_models,
            refuter_model=os.getenv("MOM_REFUTER_MODEL") or None,
            synthesizer_model=os.getenv("MOM_SYNTHESIZER_MODEL") or None,
            app_referer=os.getenv("OPENROUTER_HTTP_REFERER") or None,
            app_title=os.getenv("OPENROUTER_APP_TITLE") or "Mixture of Models",
            timeout_seconds=float(os.getenv("MOM_PROVIDER_TIMEOUT_SECONDS", "60")),
        )

    async def chat_turn(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> str:
        """Run a fusion-style chat turn for tool-using agent requests."""
        serialized_messages = self._serialize_messages(messages)
        use_tools = bool(tools) and tool_choice != "none"
        panel_responses = await asyncio.gather(
            *(
                self._complete_chat_model(
                    upstream_model=panel_model,
                    messages=serialized_messages,
                    tools=tools if use_tools else None,
                    tool_choice=tool_choice if use_tools else None,
                )
                for panel_model in self.proposer_models
            )
        )

        if not any(response.ok for response in panel_responses):
            raise ProviderCallError("All panel models failed")

        panel_block = _format_tool_panel_block(panel_responses)
        judge_messages = [
            *serialized_messages,
            {
                "role": "user",
                "content": (
                    DEFAULT_TOOL_JUDGE_TEMPLATE.format(response_count=len(panel_responses)).rstrip()
                    + "\n\n## Panel proposals\n"
                    + panel_block
                ),
            },
        ]
        message = await self._chat_raw(
            upstream_model=self.synthesizer_model,
            messages=judge_messages,
            tools=tools if use_tools else None,
            tool_choice=tool_choice if use_tools else None,
            max_tokens=16000,
            temperature=0.2,
        )
        return _message_decision_text(message, use_tools=use_tools)

    async def generate_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        messages: list[ChatMessage],
    ) -> list[AtomicClaim]:
        content = await self._chat_completion(
            upstream_model=self._proposer_model_for(role),
            messages=[
                self._system_message(
                    f"{role.instructions}\n"
                    "Return only JSON with this shape: "
                    '{"claims":[{"text":"...","evidence":["..."],"confidence":0.0}]}'
                ),
                *self._serialize_messages(messages),
            ],
            response_format={"type": "json_object"},
            max_tokens=900,
            temperature=0.2,
        )
        return self._parse_claims(role=role, content=content)

    async def refute_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        claims: list[AtomicClaim],
        messages: list[ChatMessage],
    ) -> list[RefutationResult]:
        content = await self._chat_completion(
            upstream_model=self.refuter_model,
            messages=[
                self._system_message(
                    f"{role.instructions}\n"
                    "Return only JSON with this shape: "
                    '{"refutations":[{"claim_id":"...","refuted":true,"reason":"..."}]}'
                ),
                *self._serialize_messages(messages),
                {
                    "role": "user",
                    "content": "Claims to evaluate:\n" + json.dumps([claim.__dict__ for claim in claims]),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.1,
        )
        return self._parse_refutations(role=role, claims=claims, content=content)

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
        role_instructions = role.instructions if role else "Synthesize the final answer."
        use_tools = bool(tools) and tool_choice != "none"
        system_content = (
            f"{role_instructions}\n"
            "Answer the user's request directly. When the caller supplied tools and an "
            "action is required to fulfill the request, call the appropriate tool with "
            "complete arguments. Do not expose internal claim IDs or orchestration details."
        )

        # Native tool calling: hand the synthesizer the real tools and let it emit
        # OpenAI-standard tool_calls. When it does, relay
        # them verbatim through the engine's tool-decision parser; otherwise return
        # the plain answer text.
        message = await self._chat_raw(
            upstream_model=self.synthesizer_model,
            messages=[
                self._system_message(system_content),
                *self._serialize_messages(messages),
                {
                    "role": "user",
                    "content": "Surviving internal claims:\n"
                    + json.dumps([claim.__dict__ for claim in surviving_claims]),
                },
            ],
            tools=tools if use_tools else None,
            tool_choice=tool_choice if use_tools else None,
            max_tokens=16000,
            temperature=0.3,
        )

        return _message_decision_text(message, use_tools=use_tools)

    async def _complete_chat_model(
        self,
        *,
        upstream_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> _PanelResponse:
        try:
            message = await self._chat_raw(
                upstream_model=upstream_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=16000,
                temperature=0.3,
            )
        except ProviderCallError as exc:
            return _PanelResponse(model=upstream_model, content="", error=str(exc))

        return _PanelResponse(
            model=upstream_model,
            content=_message_text(message),
            tool_calls=_message_tool_calls(message),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _chat_completion(
        self,
        *,
        upstream_model: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, str] | None = None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        message = await self._chat_raw(
            upstream_model=upstream_model,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _message_text(message)

    async def _chat_raw(
        self,
        *,
        upstream_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderCallError(f"Upstream provider returned {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderCallError("Upstream provider request failed") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError("Upstream provider response did not include a message") from exc

        return message if isinstance(message, Mapping) else {}

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_referer:
            headers["HTTP-Referer"] = self.app_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def _proposer_model_for(self, role: InternalRole) -> str:
        role_order = {
            "proposer": 0,
            "cross_checker": 1,
            "alternate_proposer": 2,
        }
        index = role_order.get(role.name, 0) % len(self.proposer_models)
        return self.proposer_models[index]

    def _serialize_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        serialized = []
        for message in messages:
            data = message.model_dump(exclude_none=True)
            role = data.get("role")
            if role == "developer":
                data["role"] = "system"
            elif role == "function":
                data["role"] = "tool"
            serialized.append(data)
        return serialized

    def _system_message(self, content: str) -> dict[str, str]:
        return {"role": "system", "content": content}

    def _parse_claims(self, *, role: InternalRole, content: str) -> list[AtomicClaim]:
        parsed = _parse_json(content)
        claim_items = _items_from(parsed, "claims")
        claims = []
        for index, item in enumerate(claim_items):
            text = _string_field(item, "text") or _string_field(item, "claim")
            if not text:
                continue
            evidence = _string_list(item.get("evidence")) if isinstance(item, Mapping) else []
            claims.append(
                AtomicClaim(
                    id=f"{role.name}-{index}",
                    text=text,
                    source_role=role.name,
                    evidence=evidence,
                    confidence=_confidence(item),
                )
            )
        return claims

    def _parse_refutations(
        self,
        *,
        role: InternalRole,
        claims: list[AtomicClaim],
        content: str,
    ) -> list[RefutationResult]:
        parsed = _parse_json(content)
        refutation_items = _items_from(parsed, "refutations")
        results = []
        for index, item in enumerate(refutation_items):
            claim_id = _string_field(item, "claim_id")
            if not claim_id and index < len(claims):
                claim_id = claims[index].id
            if not claim_id:
                continue
            results.append(
                RefutationResult(
                    claim_id=claim_id,
                    refuter_role=role.name,
                    refuted=_bool_field(item, "refuted"),
                    reason=_string_field(item, "reason") or "No reason provided.",
                )
            )
        return results


@dataclass(frozen=True)
class _PanelResponse:
    model: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _message_text(message: Any) -> str:
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return content_to_text(content).strip()
    return ""


def _message_tool_calls(message: Any) -> list[dict[str, Any]] | None:
    if not isinstance(message, Mapping):
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    valid_calls = [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]
    return valid_calls or None


def _message_decision_text(message: Any, *, use_tools: bool) -> str:
    tool_calls = _message_tool_calls(message)
    if use_tools and tool_calls:
        return json.dumps({"tool_calls": tool_calls}, separators=(",", ":"))
    return _message_text(message)


def _format_tool_panel_block(responses: Sequence[_PanelResponse]) -> str:
    chunks = []
    for index, response in enumerate(responses, start=1):
        header = f"### Model {index} - {response.model}"
        if not response.ok:
            chunks.append(f"{header}\n[no answer: this model failed - {response.error}]")
            continue

        if response.tool_calls:
            lines = []
            for tool_call in response.tool_calls:
                function = tool_call.get("function")
                if isinstance(function, Mapping):
                    name = function.get("name", "?")
                    arguments = function.get("arguments", "")
                else:
                    name = tool_call.get("name", "?")
                    arguments = tool_call.get("arguments", "")
                lines.append(f"- {name}({arguments})")
            extra = f"\nReasoning: {response.content}" if response.content else ""
            chunks.append(f"{header}\nProposed tool call(s):\n" + "\n".join(lines) + extra)
            continue

        chunks.append(f"{header}\nProposed answer:\n{response.content}")
    return "\n\n".join(chunks)


def _csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = min(_positive_index(content.find("{")), _positive_index(content.find("[")))
        end = max(content.rfind("}"), content.rfind("]"))
        if start < len(content) and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _positive_index(index: int) -> int:
    return index if index >= 0 else 10**9


def _items_from(parsed: Any, key: str) -> list[Any]:
    if isinstance(parsed, Mapping):
        items = parsed.get(key)
        if isinstance(items, list):
            return items
    if isinstance(parsed, list):
        return parsed
    return []


def _string_field(item: Any, key: str) -> str:
    if not isinstance(item, Mapping):
        return ""
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _confidence(item: Any) -> float:
    if not isinstance(item, Mapping):
        return 0.0
    value = item.get("confidence", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _bool_field(item: Any, key: str) -> bool:
    if not isinstance(item, Mapping):
        return False
    value = item.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "refuted"}
    return False
