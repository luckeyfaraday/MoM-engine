import json

from mom.core.tool_calls import parse_tool_decision


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


def test_parse_tool_decision_normalizes_openai_shape() -> None:
    content = json.dumps(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    },
                }
            ]
        }
    )

    answer, calls = parse_tool_decision(content, tools=TOOLS, tool_choice="required")

    assert answer == ""
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert json.loads(calls[0].arguments) == {"path": "README.md"}


def test_parse_tool_decision_keeps_plain_text_as_content() -> None:
    answer, calls = parse_tool_decision("plain answer", tools=TOOLS, tool_choice="auto")

    assert answer == "plain answer"
    assert calls == []
