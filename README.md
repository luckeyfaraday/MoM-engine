# Mixture of Models (MoM)

[![CI](https://github.com/luckeyfaraday/MoM-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/luckeyfaraday/MoM-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![API: OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-412991.svg)](#api-goal)
[![Built with FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **One OpenAI-compatible endpoint, many models.** MoM routes every chat completion through an internal proposer → refuter → synthesizer pipeline so several models propose, critique, and reconcile each answer.

Mixture of Models (MoM) is a Python/FastAPI service that exposes a standard `/v1/chat/completions` API and, behind it, runs every request through an internal multi-model pipeline. Any OpenAI-compatible client works unchanged — including native function/tool calling and streaming. Upstream providers are pluggable: [OpenRouter](#openrouter-upstream), the [OpenCode CLI](#opencode-cli-upstream), or a deterministic zero-cost [mock](#upstream-selection).

The user-facing contract is intentionally boring:

```text
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=mom_...
MODEL=mom-chat
```

Clients call `/v1/chat/completions` the same way they would call a normal chat model. The MoM machinery is internal to `mom-chat`. The endpoint supports normal text responses and OpenAI-style function tool calls.

The internal engine is not naive fusion. It runs one fixed architecture:

```text
OpenAI-compatible request
  -> MoM engine
  -> internal proposer/refuter/synthesizer passes
  -> OpenAI-compatible response
```

Bearer API-key enforcement is available through `MOM_API_KEYS` so OpenAI-compatible clients can be wired up early.

## Features

- **Drop-in OpenAI compatibility** — works with any client that speaks `/v1/chat/completions`; just change the base URL and model.
- **Multi-model architecture** — proposer, refuter, and synthesizer roles, each mappable to a different model.
- **Native tool calling** — returns OpenAI-style `tool_calls` with `finish_reason: "tool_calls"`, plus streaming (SSE).
- **Pluggable upstreams** — OpenRouter, OpenCode CLI, or a zero-cost deterministic mock for offline development and CI.
- **Optional bearer auth** — gate the public surface with `MOM_API_KEYS`.
- **No keys required to start** — the mock provider runs the full test suite and local server with zero configuration.

## Project Layout

- `mom/api/`: FastAPI app, OpenAI-compatible schemas, and response builders.
- `mom/core/`: MoM engine, internal architecture, claim, refutation, synthesis, tool-call parsing, and telemetry structures.
- `mom/providers/`: Pluggable provider interface, deterministic mock provider, OpenCode CLI provider, and OpenAI-compatible provider.
- `tests/`: API compatibility and claim lifecycle tests.

## Run Locally

Install in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run the OpenCode-backed dev server:

```bash
scripts/dev-opencode.sh
```

Prompt it from another terminal:

```bash
scripts/prompt.sh "Explain what this API does in one paragraph."
```

Use it as an OpenCode model by registering this provider in your OpenCode config:

```jsonc
"mom": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "Mixture of Models",
  "options": {
    "baseURL": "http://127.0.0.1:8000/v1",
    "apiKey": "sk-local-test"
  },
  "models": {
    "mom-chat": {
      "name": "MoM Chat",
      "tool_call": true
    }
  }
}
```

Then run:

```bash
opencode run -m mom/mom-chat "Inspect this repo and summarize the main API surface."
```

Check the API directly:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

To require bearer API keys in local/server mode, set `MOM_API_KEYS` to a comma-separated list:

```bash
MOM_API_KEYS=mom_dev_key uvicorn mom.api.server:app --reload
```

Then clients must send:

```http
Authorization: Bearer mom_dev_key
```

## Upstream Selection

The upstream provider is selected with `MOM_UPSTREAM`.

```text
MOM_UPSTREAM=mock          # zero-cost deterministic local provider
MOM_UPSTREAM=opencode-cli  # dev-only OpenCode CLI provider
MOM_UPSTREAM=openrouter    # direct OpenRouter API provider
```

If `MOM_UPSTREAM` is not set but `OPENROUTER_API_KEY` is present, the API uses OpenRouter. Otherwise it uses the deterministic mock provider.

## OpenCode CLI Upstream

Use this for budget-safe local architecture testing through your existing OpenCode setup. The default model order is:

```text
opencode/deepseek-v4-flash-free
opencode/mimo-v2.5-free
opencode/nemotron-3-ultra-free
```

The wrapper `scripts/dev-opencode.sh` sets these defaults for you. To override them manually:

```bash
MOM_UPSTREAM=opencode-cli \
MOM_PROPOSER_MODELS=opencode/deepseek-v4-flash-free,opencode/mimo-v2.5-free,opencode/nemotron-3-ultra-free \
MOM_REFUTER_MODEL=opencode/deepseek-v4-flash-free \
MOM_SYNTHESIZER_MODEL=opencode/deepseek-v4-flash-free \
uvicorn mom.api.server:app --reload
```

Optional OpenCode settings:

```bash
OPENCODE_BIN=opencode
MOM_OPENCODE_DIR=/tmp/mom-opencode-provider
MOM_PROVIDER_TIMEOUT_SECONDS=180
```

This path shells out to `opencode run --pure --format json --title "MoM upstream"`. It is intended for local development, not for a public hosted API. The default `MOM_OPENCODE_DIR` is outside the repo so internal model passes do not operate on project files.

## OpenRouter Upstream

Use this for direct API testing and production-like integration.

Copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`, then start the dev server:

```bash
cp .env.example .env
# edit .env to add your OPENROUTER_API_KEY
scripts/dev-openrouter.sh
```

`scripts/dev-openrouter.sh` loads `.env` and applies sensible model defaults. To run
uvicorn directly with an explicit configuration instead:

```bash
OPENROUTER_API_KEY=or_...
MOM_PROPOSER_MODELS=~openai/gpt-latest,~anthropic/claude-sonnet-latest
MOM_REFUTER_MODEL=~minimax/minimax-m3
MOM_SYNTHESIZER_MODEL=~openai/gpt-latest
uvicorn mom.api.server:app --reload
```

Optional OpenRouter settings:

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://github.com/luckeyfaraday/MoM-engine
OPENROUTER_APP_TITLE="Mixture of Models"
MOM_PROVIDER_TIMEOUT_SECONDS=60
```

## API Goal

The default response from `/v1/chat/completions` looks like a minimal OpenAI chat completion so existing clients can integrate with the service. When a client supplies `tools` and the internal synthesizer decides a tool is needed, the API returns assistant `tool_calls` with `finish_reason: "tool_calls"`. The client executes the tool and sends the result back as a normal tool message on the next request.

## FAQ

**What is a Mixture of Models?** Instead of returning one model's answer, MoM
asks several models to propose candidate answers, has a model critique
(refute) them, and has a synthesizer model produce a single reconciled reply —
all behind a normal OpenAI chat endpoint.

**How is this different from a router or load balancer?** A router picks *one*
model per request. MoM combines *multiple* models per request through a fixed
proposer/refuter/synthesizer architecture.

**How does this relate to other systems for combining AI models?** Combining
multiple AI models to produce better answers is an active area — for example
**Sakana AI's Fugu**, a Japanese model that takes a similar approach of blending
several models together. MoM does this as **inference-time fusion**: it combines
the *outputs* of several models per request (think "OpenRouter fusion" when they
are served via OpenRouter), keeping each model separate and reconciling their
answers live rather than merging weights into one offline model.

**Why "loops"?** The proposer → refuter → synthesizer passes form a short
refinement loop over a single request, so each answer is critiqued before it's
returned rather than emitted in one shot.

**Do I need API keys to try it?** No. With no environment variables set, MoM
uses a deterministic, zero-cost mock provider, so the server and tests run
offline.

**Which clients work with it?** Anything OpenAI-compatible — the OpenAI SDKs,
LangChain, OpenCode, `curl`, etc. Point them at `/v1` and use model `mom-chat`.

**Is it production-ready?** It's early-stage (alpha). The OpenRouter upstream is
the path for production-like integration.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security reports, see
[SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE).

---

<sub>**Keywords:** mixture of models · MoM · combining AI models · OpenRouter fusion · multi-model LLM API · OpenAI-compatible API · inference-time model fusion · model ensembling · LLM synthesis · proposer / refuter / synthesizer · agentic loops · Sakana Fugu · FastAPI LLM gateway · tool calling.</sub>
