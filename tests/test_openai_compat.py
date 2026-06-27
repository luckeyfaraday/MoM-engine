from fastapi.testclient import TestClient
import json

from mom.api.server import app


client = TestClient(app)


def test_models_returns_at_least_one_model() -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == ["mom-chat"]
    assert body["data"][0]["object"] == "model"


def test_chat_completions_returns_openai_compatible_shape() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Answer this normal chat request."}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "mom-chat"
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    content = body["choices"][0]["message"]["content"]
    assert content.startswith("- Supported answer material:")
    assert body["usage"]["total_tokens"] >= body["usage"]["completion_tokens"]


def test_chat_completions_streams_openai_compatible_chunks() -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Stream this response."}],
            "stream": True,
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in body
    assert 'data: [DONE]' in body


def test_chat_completions_returns_openai_tool_calls_when_required() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Read the project README."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file from the workspace.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    tool_call = message["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"])["request"] == "Read the project README."


def test_chat_completions_streams_openai_tool_call_chunks() -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Read the project README."}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert '"tool_calls"' in body
    assert '"name":"read_file"' in body
    assert '"finish_reason":"tool_calls"' in body
    assert "data: [DONE]" in body


def test_chat_completions_rejects_missing_api_key_when_keys_are_configured(monkeypatch) -> None:
    monkeypatch.setenv("MOM_API_KEYS", "mom_test_key")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 401


def test_chat_completions_accepts_configured_api_key(monkeypatch) -> None:
    monkeypatch.setenv("MOM_API_KEYS", "mom_test_key")

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer mom_test_key"},
        json={
            "model": "mom-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mom-chat"
