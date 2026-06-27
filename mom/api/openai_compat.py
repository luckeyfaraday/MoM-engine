from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from mom.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelListResponse,
    Usage,
)
from mom.core.tool_calls import ToolCall


def new_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def build_model_list(model_ids: list[str]) -> ModelListResponse:
    return ModelListResponse(data=[ModelInfo(id=model_id) for model_id in model_ids])


def build_chat_completion(
    *,
    model: str,
    content: str | None,
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletionResponse:
    message_kwargs: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message_kwargs["content"] = None
        message_kwargs["tool_calls"] = [tool_call.as_openai() for tool_call in tool_calls]
        finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=completion_id or new_chat_completion_id(),
        created=created or int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(**message_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def build_chat_completion_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def iter_chat_completion_sse(
    *,
    model: str,
    content: str,
    tool_calls: list[ToolCall] | None = None,
) -> Iterator[str]:
    completion_id = new_chat_completion_id()
    created = int(time.time())

    yield _sse(
        build_chat_completion_chunk(
            completion_id=completion_id,
            created=created,
            model=model,
            delta={"role": "assistant", "content": ""},
        )
    )

    if tool_calls:
        # Emit each tool call in a SINGLE delta with the complete function
        # (name + arguments together). Splitting name and arguments across two deltas confuses some
        # client parsers (e.g. opencode), so the relayed call never executed.
        yield _sse(
            build_chat_completion_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for index, tool_call in enumerate(tool_calls)
                    ]
                },
            )
        )

        yield _sse(
            build_chat_completion_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={},
                finish_reason="tool_calls",
            )
        )
        yield "data: [DONE]\n\n"
        return

    for chunk in _content_chunks(content):
        yield _sse(
            build_chat_completion_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"content": chunk},
            )
        )

    yield _sse(
        build_chat_completion_chunk(
            completion_id=completion_id,
            created=created,
            model=model,
            delta={},
            finish_reason="stop",
        )
    )
    yield "data: [DONE]\n\n"


def _content_chunks(content: str) -> Iterator[str]:
    words = content.split(" ")
    for index, word in enumerate(words):
        if not word:
            continue
        suffix = " " if index < len(words) - 1 else ""
        yield f"{word}{suffix}"


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
