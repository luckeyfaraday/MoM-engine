from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


def parse_tool_decision(
    content: str,
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> tuple[str, list[ToolCall]]:
    if not _tool_use_allowed(tools=tools, tool_choice=tool_choice):
        return content, []

    parsed = _parse_json(content)
    if parsed is None:
        return content, []

    calls = _tool_calls_from(parsed, tools=tools or [], forced_name=forced_tool_name(tool_choice))
    if calls:
        return "", calls

    if isinstance(parsed, Mapping):
        direct_content = parsed.get("content") or parsed.get("answer") or parsed.get("final_answer")
        if isinstance(direct_content, str):
            return direct_content.strip(), []

    return content, []


def tool_decision_instructions(
    *,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> str:
    return (
        "The caller supplied tools. You are deciding whether the assistant should call a tool; "
        "do not execute tools yourself.\n"
        f"Tool choice: {json.dumps(tool_choice or 'auto', separators=(',', ':'))}\n"
        f"Available tools: {json.dumps(_function_tools(tools), separators=(',', ':'))}\n\n"
        "Return only JSON in one of these shapes:\n"
        '{"content":"final answer text"}\n'
        '{"tool_calls":[{"name":"tool_name","arguments":{"arg":"value"}}]}\n'
        "Use tool_calls when tool_choice is required or a tool is needed to answer correctly. "
        "Use only listed tool names. Arguments must match the tool schema as closely as possible."
    )


def forced_tool_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    if not isinstance(tool_choice, Mapping):
        return None
    if tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def first_function_tool_name(tools: list[dict[str, Any]]) -> str | None:
    for tool in tools:
        name = _function_tool_name(tool)
        if name:
            return name
    return None


def _tool_use_allowed(
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> bool:
    return bool(tools) and tool_choice != "none"


def _tool_calls_from(
    parsed: Any,
    *,
    tools: list[dict[str, Any]],
    forced_name: str | None,
) -> list[ToolCall]:
    raw_calls: list[Any] = []
    if isinstance(parsed, Mapping):
        tool_calls = parsed.get("tool_calls")
        if isinstance(tool_calls, list):
            raw_calls.extend(tool_calls)
        function_call = parsed.get("function_call")
        if isinstance(function_call, Mapping):
            raw_calls.append(function_call)
        if _looks_like_tool_call(parsed):
            raw_calls.append(parsed)
    elif isinstance(parsed, list):
        raw_calls.extend(parsed)

    valid_names = _function_tool_names(tools)
    calls = []
    for raw_call in raw_calls:
        call = _normalize_tool_call(raw_call, valid_names=valid_names, forced_name=forced_name)
        if call:
            calls.append(call)
    return calls


def _normalize_tool_call(
    raw_call: Any,
    *,
    valid_names: set[str],
    forced_name: str | None,
) -> ToolCall | None:
    if not isinstance(raw_call, Mapping):
        return None

    function = raw_call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")

    if not isinstance(name, str) or not name:
        return None
    if forced_name and name != forced_name:
        return None
    if valid_names and name not in valid_names:
        return None

    call_id = raw_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex[:24]}"

    return ToolCall(id=call_id, name=name, arguments=_arguments_json(arguments))


def _looks_like_tool_call(value: Mapping[str, Any]) -> bool:
    return "name" in value and "arguments" in value


def _arguments_json(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return json.dumps({"input": stripped}, separators=(",", ":"))
        return json.dumps(parsed, separators=(",", ":"))
    return json.dumps(value, separators=(",", ":"))


def _function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools if _function_tool_name(tool)]


def _function_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {name for tool in tools if (name := _function_tool_name(tool))}


def _function_tool_name(tool: dict[str, Any]) -> str | None:
    if tool.get("type") != "function":
        return None
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


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
