# Mixture of Models

This repository is a Python skeleton for a Mixture-of-Models (MoM) API with an OpenAI-compatible public surface.

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
OPENROUTER_HTTP_REFERER=https://mixtureofmodels.com
OPENROUTER_APP_TITLE="Mixture of Models"
MOM_PROVIDER_TIMEOUT_SECONDS=60
```

## API Goal

The default response from `/v1/chat/completions` looks like a minimal OpenAI chat completion so existing clients can integrate with the service. When a client supplies `tools` and the internal synthesizer decides a tool is needed, the API returns assistant `tool_calls` with `finish_reason: "tool_calls"`. The client executes the tool and sends the result back as a normal tool message on the next request.

## License

Released under the [MIT License](LICENSE).
