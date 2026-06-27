from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from mom.api.messages import approximate_token_count, content_to_text
from mom.api.openai_compat import build_chat_completion, build_model_list, iter_chat_completion_sse
from mom.api.schemas import ChatCompletionRequest, ChatCompletionResponse, ModelListResponse
from mom.core.engine import MoMEngine
from mom.providers.base import ProviderCallError
from mom.providers.mock import MockProvider
from mom.providers.opencode_cli import OpenCodeCliProvider
from mom.providers.openai_compatible import OpenAICompatibleProvider

MODEL_IDS = ["mom-chat"]


def build_engine() -> MoMEngine:
    upstream = os.getenv("MOM_UPSTREAM", "").strip().lower()
    if upstream == "opencode-cli":
        return MoMEngine(provider=OpenCodeCliProvider.from_env())
    if upstream == "openrouter" or (not upstream and os.getenv("OPENROUTER_API_KEY")):
        return MoMEngine(provider=OpenAICompatibleProvider.from_env())
    return MoMEngine(provider=MockProvider())


MOM_ENGINE = build_engine()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        close_provider = getattr(MOM_ENGINE.provider, "aclose", None)
        if close_provider:
            await close_provider()


app = FastAPI(title="Mixture of Models API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    return build_model_list(MODEL_IDS)


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> ChatCompletionResponse | StreamingResponse:
    _authorize(authorization)
    if request.model not in MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    try:
        result = await MOM_ENGINE.chat(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )
    except ProviderCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    prompt_tokens = sum(approximate_token_count(content_to_text(message.content)) for message in request.messages)
    completion_tokens = len(result.final_answer.split()) + sum(
        len(tool_call.name.split()) + approximate_token_count(tool_call.arguments)
        for tool_call in result.tool_calls
    )
    if request.stream:
        return StreamingResponse(
            iter_chat_completion_sse(
                model=request.model,
                content=result.final_answer,
                tool_calls=result.tool_calls,
            ),
            media_type="text/event-stream",
        )

    return build_chat_completion(
        model=request.model,
        content=result.final_answer,
        tool_calls=result.tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _authorize(authorization: str | None) -> None:
    valid_keys = _configured_api_keys()
    if not valid_keys:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")

    token = authorization.removeprefix("Bearer ").strip()
    if token not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _configured_api_keys() -> set[str]:
    raw = os.getenv("MOM_API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}
