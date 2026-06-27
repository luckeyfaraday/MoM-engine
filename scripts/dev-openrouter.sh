#!/usr/bin/env bash
set -euo pipefail

# Run the MoM API against OpenRouter with native tool calling.
# Reads configuration from a local .env file if present (see .env.example).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set. Add it to $ENV_FILE (see .env.example) or export it." >&2
  exit 1
fi

export MOM_UPSTREAM="${MOM_UPSTREAM:-openrouter}"
export MOM_PROPOSER_MODELS="${MOM_PROPOSER_MODELS:-deepseek/deepseek-v4-flash,xiaomi/mimo-v2.5,minimax/minimax-m3}"
export MOM_REFUTER_MODEL="${MOM_REFUTER_MODEL:-deepseek/deepseek-v4-flash}"
export MOM_SYNTHESIZER_MODEL="${MOM_SYNTHESIZER_MODEL:-minimax/minimax-m3}"
export MOM_PROVIDER_TIMEOUT_SECONDS="${MOM_PROVIDER_TIMEOUT_SECONDS:-180}"

exec uvicorn mom.api.server:app --reload --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
