import asyncio
import json

import httpx

from mom.api.schemas import ChatMessage
from mom.core.engine import MoMEngine
from mom.providers.openai_compatible import OpenAICompatibleProvider


def test_openai_compatible_provider_calls_openrouter_shape() -> None:
    result, requests = asyncio.run(_run_provider())

    assert result.final_answer == "Final answer from synthesized claims."
    assert [request.url.path for request in requests] == [
        "/api/v1/chat/completions",
        "/api/v1/chat/completions",
        "/api/v1/chat/completions",
        "/api/v1/chat/completions",
        "/api/v1/chat/completions",
    ]
    assert all(request.headers["Authorization"] == "Bearer or_test_key" for request in requests)
    assert all(request.headers["HTTP-Referer"] == "https://example.test" for request in requests)
    assert all(request.headers["X-OpenRouter-Title"] == "MoM Test" for request in requests)

    payloads = [json.loads(request.content) for request in requests]
    assert payloads[3]["model"] == "openrouter/refuter"
    assert payloads[4]["model"] == "openrouter/synthesizer"
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[3]["response_format"] == {"type": "json_object"}
    assert result.claims
    assert result.refutations
    assert [claim.id for claim in result.surviving_claims] == ["proposer-0"]


def test_openai_compatible_provider_fuses_tool_chat_without_claim_refute() -> None:
    result, requests = asyncio.run(_run_tool_chat_provider())

    assert result.final_answer == ""
    assert not result.claims
    assert not result.refutations
    assert not result.surviving_claims
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "write"
    assert json.loads(result.tool_calls[0].arguments) == {
        "filePath": "index.html",
        "content": "<!doctype html><title>ok</title>",
    }

    assert len(requests) == 3
    payloads = [json.loads(request.content) for request in requests]
    assert [payload["model"] for payload in payloads] == [
        "openrouter/proposer-a",
        "openrouter/proposer-b",
        "openrouter/synthesizer",
    ]
    assert all("tools" in payload for payload in payloads)
    assert all("response_format" not in payload for payload in payloads)
    assert "Panel proposals" in payloads[2]["messages"][-1]["content"]


async def _run_provider():
    requests: list[httpx.Request] = []
    responses = [
        _chat_response(
            {
                "claims": [
                    {
                        "text": "The endpoint should hide internal orchestration.",
                        "evidence": ["user-request"],
                        "confidence": 0.9,
                    }
                ]
            }
        ),
        _chat_response(
            {
                "claims": [
                    {
                        "text": "This second claim should be rejected.",
                        "evidence": ["weak"],
                        "confidence": 0.4,
                    }
                ]
            }
        ),
        _chat_response(
            {
                "claims": [
                    {
                        "text": "This third claim should also be rejected.",
                        "evidence": ["weak"],
                        "confidence": 0.4,
                    }
                ]
            }
        ),
        _chat_response(
            {
                "refutations": [
                    {
                        "claim_id": "proposer-0",
                        "refuted": False,
                        "reason": "Supported.",
                    },
                    {
                        "claim_id": "cross_checker-0",
                        "refuted": True,
                        "reason": "Weak evidence.",
                    },
                    {
                        "claim_id": "alternate_proposer-0",
                        "refuted": True,
                        "reason": "Weak evidence.",
                    },
                ]
            }
        ),
        _chat_response("Final answer from synthesized claims."),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses[len(requests) - 1]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        api_key="or_test_key",
        base_url="https://openrouter.ai/api/v1",
        proposer_models=("openrouter/proposer-a", "openrouter/proposer-b"),
        refuter_model="openrouter/refuter",
        synthesizer_model="openrouter/synthesizer",
        app_referer="https://example.test",
        app_title="MoM Test",
        client=client,
    )
    try:
        engine = MoMEngine(provider=provider)
        result = await engine.chat(
            model="mom-chat",
            messages=[
                ChatMessage(role="developer", content="Stay concise."),
                ChatMessage(role="user", content="Explain the API contract."),
            ],
        )
    finally:
        await client.aclose()

    return result, requests


async def _run_tool_chat_provider():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        model = payload["model"]
        if model == "openrouter/proposer-a":
            return _chat_tool_response(
                "write",
                {"filePath": "index.html", "content": "<!doctype html><title>a</title>"},
                content="Use the write tool.",
            )
        if model == "openrouter/proposer-b":
            return _chat_response("The next step is to write index.html.")
        return _chat_tool_response(
            "write",
            {"filePath": "index.html", "content": "<!doctype html><title>ok</title>"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        api_key="or_test_key",
        base_url="https://openrouter.ai/api/v1",
        proposer_models=("openrouter/proposer-a", "openrouter/proposer-b"),
        refuter_model="openrouter/refuter",
        synthesizer_model="openrouter/synthesizer",
        client=client,
    )
    try:
        engine = MoMEngine(provider=provider)
        result = await engine.chat(
            model="mom-chat",
            messages=[ChatMessage(role="user", content="Create index.html.")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filePath": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filePath", "content"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
    finally:
        await client.aclose()

    return result, requests


def _chat_response(content: str | dict) -> httpx.Response:
    if isinstance(content, dict):
        message_content = json.dumps(content)
    else:
        message_content = content
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": message_content}}],
        },
    )


def _chat_tool_response(name: str, arguments: dict, *, content: str | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )
