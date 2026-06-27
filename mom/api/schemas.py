from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "mixtureofmodels"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Any | None = None
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
