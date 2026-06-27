import asyncio
import json

from mom.api.schemas import ChatMessage
from mom.core.engine import MoMEngine
from mom.providers.opencode_cli import OpenCodeCliProvider, extract_opencode_text


def test_opencode_cli_provider_uses_free_model_order() -> None:
    result, commands = asyncio.run(_run_provider())

    assert result.final_answer == "Final OpenCode answer."
    assert [command[command.index("--model") + 1] for command in commands] == [
        "opencode/deepseek-v4-flash-free",
        "opencode/mimo-v2.5-free",
        "opencode/nemotron-3-ultra-free",
        "opencode/deepseek-v4-flash-free",
        "opencode/deepseek-v4-flash-free",
    ]
    assert all(command[command.index("--agent") + 1] == "mom-upstream" for command in commands)
    assert [claim.id for claim in result.surviving_claims] == ["proposer-0"]


def test_extract_opencode_text_handles_json_events() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "started"}),
            json.dumps({"type": "message", "message": {"role": "assistant", "content": "hello"}}),
        ]
    )

    assert extract_opencode_text(stdout) == "hello"


def test_extract_opencode_text_does_not_leak_tool_event_json() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start"}),
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "read",
                        "state": {
                            "status": "error",
                            "error": "The user rejected permission to use this specific tool call.",
                        },
                    },
                }
            ),
            json.dumps({"type": "step_finish", "part": {"reason": "tool-calls"}}),
        ]
    )

    assert extract_opencode_text(stdout) == ""


async def _run_provider():
    commands: list[list[str]] = []
    responses = [
        {"claims": [{"text": "Strong claim.", "evidence": ["input"], "confidence": 0.9}]},
        {"claims": [{"text": "Weak MiMo claim.", "evidence": [], "confidence": 0.3}]},
        {"claims": [{"text": "Weak Nemotron claim.", "evidence": [], "confidence": 0.3}]},
        {
            "refutations": [
                {"claim_id": "proposer-0", "refuted": False, "reason": "Supported."},
                {"claim_id": "cross_checker-0", "refuted": True, "reason": "Unsupported."},
                {"claim_id": "alternate_proposer-0", "refuted": True, "reason": "Unsupported."},
            ]
        },
        "Final OpenCode answer.",
    ]

    async def runner(command: list[str], cwd: str, timeout_seconds: float) -> str:
        commands.append(command)
        content = responses[len(commands) - 1]
        if isinstance(content, dict):
            content = json.dumps(content)
        return json.dumps({"message": {"role": "assistant", "content": content}})

    provider = OpenCodeCliProvider(runner=runner, cwd="/tmp")
    engine = MoMEngine(provider=provider)
    result = await engine.chat(
        model="mom-chat",
        messages=[ChatMessage(role="user", content="Test the architecture.")],
    )
    return result, commands
