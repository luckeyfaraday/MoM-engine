from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mom.api.messages import content_to_text
from mom.api.schemas import ChatMessage
from mom.core.architecture import InternalRole
from mom.core.claims import AtomicClaim, RefutationResult
from mom.core.tool_calls import tool_decision_instructions
from mom.providers.base import ProviderCallError

CommandRunner = Callable[[list[str], str, float], Awaitable[str]]

DEFAULT_OPENCODE_MODELS = (
    "opencode/deepseek-v4-flash-free",
    "opencode/mimo-v2.5-free",
    "opencode/nemotron-3-ultra-free",
)


class OpenCodeCliProvider:
    """Dev-only provider that uses `opencode run` as the upstream model caller."""

    def __init__(
        self,
        *,
        proposer_models: Sequence[str] = DEFAULT_OPENCODE_MODELS,
        refuter_model: str | None = None,
        synthesizer_model: str | None = None,
        opencode_bin: str = "opencode",
        agent: str = "mom-upstream",
        cwd: str | None = None,
        timeout_seconds: float = 180.0,
        runner: CommandRunner | None = None,
    ) -> None:
        if not proposer_models:
            raise ValueError("at least one proposer model is required")
        self.proposer_models = tuple(proposer_models)
        self.refuter_model = refuter_model or self.proposer_models[0]
        self.synthesizer_model = synthesizer_model or self.proposer_models[0]
        self.opencode_bin = opencode_bin
        self.agent = agent
        self.cwd = cwd or os.getcwd()
        self.timeout_seconds = timeout_seconds
        self.runner = runner or _run_command

    @classmethod
    def from_env(cls) -> OpenCodeCliProvider:
        proposer_models = _csv(os.getenv("MOM_PROPOSER_MODELS")) or DEFAULT_OPENCODE_MODELS
        return cls(
            proposer_models=proposer_models,
            refuter_model=os.getenv("MOM_REFUTER_MODEL") or proposer_models[0],
            synthesizer_model=os.getenv("MOM_SYNTHESIZER_MODEL") or proposer_models[0],
            opencode_bin=os.getenv("OPENCODE_BIN", "opencode"),
            agent=os.getenv("MOM_OPENCODE_AGENT", "mom-upstream"),
            cwd=os.getenv("MOM_OPENCODE_DIR") or _default_opencode_dir(),
            timeout_seconds=float(os.getenv("MOM_PROVIDER_TIMEOUT_SECONDS", "180")),
        )

    async def generate_claims(
        self,
        *,
        model: str,
        role: InternalRole,
        messages: list[ChatMessage],
    ) -> list[AtomicClaim]:
        content = await self._run_model(
            model=self._proposer_model_for(role),
            prompt=_claims_prompt(role=role, messages=messages),
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
        content = await self._run_model(
            model=self.refuter_model,
            prompt=_refutations_prompt(role=role, messages=messages, claims=claims),
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
        # Synthesis is the only leg whose output the caller actually sees, so it
        # must not die on a single throttled model. Try the configured
        # synthesizer first, then fall back through the proposer models.
        return await self._run_synthesis_with_fallback(
            models=self._synthesis_model_chain(),
            prompt=_synthesis_prompt(
                role=role,
                messages=messages,
                surviving_claims=surviving_claims,
                tools=tools,
                tool_choice=tool_choice,
            ),
            tools=tools,
            tool_choice=tool_choice,
        )

    def _synthesis_model_chain(self) -> list[str]:
        chain = [self.synthesizer_model, *self.proposer_models]
        seen: set[str] = set()
        return [m for m in chain if not (m in seen or seen.add(m))]

    async def _run_synthesis_with_fallback(
        self,
        *,
        models: list[str],
        prompt: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> str:
        last_error: ProviderCallError | None = None
        capture_tools = bool(tools) and tool_choice != "none"
        for model in models:
            try:
                stdout = await self._run_raw(model=model, prompt=prompt)
            except ProviderCallError as exc:
                last_error = exc
                continue
            # The upstream runs inside opencode's agent harness, which *executes*
            # tools (e.g. write) rather than returning a decision. When that
            # happens the real file content lives in the tool_use event, while
            # the trailing text is just narration ("Created index.html"). Relay
            # the executed call to our caller so the caller's agent performs the
            # write in its own workspace instead of losing it.
            if capture_tools:
                relayed = extract_opencode_tool_calls(stdout)
                if relayed:
                    return json.dumps({"tool_calls": relayed})
            text = extract_opencode_text(stdout)
            if text:
                return text
        raise last_error or ProviderCallError("OpenCode CLI did not return assistant text")

    async def _run_model(self, *, model: str, prompt: str) -> str:
        stdout = await self._run_raw(model=model, prompt=prompt)
        text = extract_opencode_text(stdout)
        if not text:
            raise ProviderCallError("OpenCode CLI did not return assistant text")
        return text

    async def _run_raw(self, *, model: str, prompt: str) -> str:
        command = [
            self.opencode_bin,
            "run",
            "--pure",
            "--model",
            model,
            "--agent",
            self.agent,
            "--format",
            "json",
            "--title",
            "MoM upstream",
            "--dir",
            self.cwd,
            prompt,
        ]
        return await self.runner(command, self.cwd, self.timeout_seconds)

    def _proposer_model_for(self, role: InternalRole) -> str:
        role_order = {
            "proposer": 0,
            "cross_checker": 1,
            "alternate_proposer": 2,
        }
        index = role_order.get(role.name, 0) % len(self.proposer_models)
        return self.proposer_models[index]

    def _parse_claims(self, *, role: InternalRole, content: str) -> list[AtomicClaim]:
        parsed = _parse_json(content)
        claims = []
        for index, item in enumerate(_items_from(parsed, "claims")):
            text = _string_field(item, "text") or _string_field(item, "claim")
            if not text:
                continue
            claims.append(
                AtomicClaim(
                    id=f"{role.name}-{index}",
                    text=text,
                    source_role=role.name,
                    evidence=_string_list(item.get("evidence")) if isinstance(item, Mapping) else [],
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
        refutations = []
        for index, item in enumerate(_items_from(parsed, "refutations")):
            claim_id = _string_field(item, "claim_id")
            if not claim_id and index < len(claims):
                claim_id = claims[index].id
            if not claim_id:
                continue
            refutations.append(
                RefutationResult(
                    claim_id=claim_id,
                    refuter_role=role.name,
                    refuted=_bool_field(item, "refuted"),
                    reason=_string_field(item, "reason") or "No reason provided.",
                )
            )
        return refutations


def extract_opencode_tool_calls(stdout: str) -> list[dict[str, Any]]:
    """Pull executed tool calls out of opencode's `--format json` event stream.

    opencode emits a ``tool_use`` event per executed tool, carrying the tool
    name and the exact input it was called with. We surface those as
    ``{"name", "arguments"}`` so the engine can relay them as OpenAI tool calls.
    """
    calls: list[dict[str, Any]] = []
    for item in _decode_opencode_output(stdout):
        if not isinstance(item, Mapping) or item.get("type") != "tool_use":
            continue
        part = item.get("part")
        if not isinstance(part, Mapping):
            continue
        name = part.get("tool")
        if not isinstance(name, str) or not name:
            continue
        state = part.get("state")
        arguments = state.get("input") if isinstance(state, Mapping) else None
        calls.append({"name": name, "arguments": _relocate_paths(arguments)})
    return calls


def _relocate_paths(arguments: Any) -> dict[str, Any]:
    """Make relayed file paths land in the caller's workspace, not the upstream's.

    The upstream runs in its own scratch dir and often picks an absolute temp
    path (e.g. ``/tmp/opencode/index.html``). The caller's agent should write
    into its own workspace, so collapse absolute paths to their basename while
    leaving genuine relative paths (``src/app.html``) untouched.
    """
    if not isinstance(arguments, Mapping):
        return {}
    relocated = dict(arguments)
    for key in ("filePath", "path", "file"):
        value = relocated.get(key)
        if isinstance(value, str) and os.path.isabs(value):
            relocated[key] = os.path.basename(value)
    return relocated


def extract_opencode_text(stdout: str) -> str:
    decoded = _decode_opencode_output(stdout)
    candidates = [_candidate_text(item) for item in decoded]
    candidates = [candidate for candidate in candidates if candidate]
    if candidates:
        return candidates[-1].strip()
    if decoded:
        return ""
    return stdout.strip()


async def _run_command(command: list[str], cwd: str, timeout_seconds: float) -> str:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise ProviderCallError("OpenCode CLI timed out") from exc

    if process.returncode != 0:
        error = stderr.decode(errors="replace").strip()
        raise ProviderCallError(f"OpenCode CLI failed: {error or process.returncode}")
    return stdout.decode(errors="replace")


def _claims_prompt(*, role: InternalRole, messages: list[ChatMessage]) -> str:
    return (
        f"{role.instructions}\n\n"
        "Return only JSON with this shape:\n"
        '{"claims":[{"text":"...","evidence":["..."],"confidence":0.0}]}\n\n'
        f"Conversation:\n{_conversation_text(messages)}"
    )


def _refutations_prompt(*, role: InternalRole, messages: list[ChatMessage], claims: list[AtomicClaim]) -> str:
    return (
        f"{role.instructions}\n\n"
        "Return only JSON with this shape:\n"
        '{"refutations":[{"claim_id":"...","refuted":true,"reason":"..."}]}\n\n'
        f"Conversation:\n{_conversation_text(messages)}\n\n"
        f"Claims:\n{json.dumps([claim.__dict__ for claim in claims])}"
    )


def _synthesis_prompt(
    *,
    role: InternalRole | None,
    messages: list[ChatMessage],
    surviving_claims: list[AtomicClaim],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    instructions = role.instructions if role else "Synthesize the final answer."
    response_instruction = "Answer the user's request directly."
    if tools and tool_choice != "none":
        instructions = f"{instructions}\n\n{tool_decision_instructions(tools=tools, tool_choice=tool_choice)}"
        response_instruction = "Return the JSON decision directly."
    return (
        f"{instructions}\n"
        f"{response_instruction} Do not expose internal claim IDs or orchestration details.\n\n"
        f"Conversation:\n{_conversation_text(messages)}\n\n"
        f"Surviving internal claims:\n{json.dumps([claim.__dict__ for claim in surviving_claims])}"
    )


def _conversation_text(messages: list[ChatMessage]) -> str:
    lines = []
    for message in messages:
        content = content_to_text(message.content)
        if content:
            lines.append(f"{message.role}: {content}")
    return "\n".join(lines)


def _default_opencode_dir() -> str:
    path = Path("/tmp/mom-opencode-provider")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _decode_opencode_output(stdout: str) -> list[Any]:
    stripped = stdout.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        decoded = []
        for line in stripped.splitlines():
            try:
                decoded.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return decoded


def _candidate_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("text", "content", "message", "output", "result"):
            value = item.get(key)
            text = _candidate_text(value)
            if text:
                return text
        for value in item.values():
            if isinstance(value, (Mapping, list)):
                text = _candidate_text(value)
                if text:
                    return text
    if isinstance(item, list):
        parts = [_candidate_text(value) for value in item]
        return "".join(part for part in parts if part)
    return ""


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
        return items if isinstance(items, list) else []
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
    try:
        return max(0.0, min(1.0, float(item.get("confidence", 0.0))))
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
