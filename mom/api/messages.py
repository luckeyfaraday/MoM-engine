from __future__ import annotations

from typing import Any

from mom.api.schemas import ChatMessage


def content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif isinstance(part.get("content"), str):
            text_parts.append(part["content"])

    return " ".join(text_parts)


def messages_to_text(messages: list[ChatMessage], *, role: str | None = None) -> str:
    return " ".join(
        content_to_text(message.content)
        for message in messages
        if role is None or message.role == role
    ).strip()


def approximate_token_count(text: str) -> int:
    return len(text.split())
