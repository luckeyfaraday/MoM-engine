#!/usr/bin/env bash
set -euo pipefail

export MOM_UPSTREAM="${MOM_UPSTREAM:-opencode-cli}"
export MOM_PROPOSER_MODELS="${MOM_PROPOSER_MODELS:-opencode/deepseek-v4-flash-free,opencode/mimo-v2.5-free,opencode/nemotron-3-ultra-free}"
export MOM_REFUTER_MODEL="${MOM_REFUTER_MODEL:-opencode/deepseek-v4-flash-free}"
export MOM_SYNTHESIZER_MODEL="${MOM_SYNTHESIZER_MODEL:-opencode/deepseek-v4-flash-free}"
export MOM_OPENCODE_AGENT="${MOM_OPENCODE_AGENT:-mom-upstream}"
export MOM_OPENCODE_DIR="${MOM_OPENCODE_DIR:-/tmp/mom-opencode-provider}"
export MOM_PROVIDER_TIMEOUT_SECONDS="${MOM_PROVIDER_TIMEOUT_SECONDS:-180}"

mkdir -p "$MOM_OPENCODE_DIR"

exec uvicorn mom.api.server:app --reload --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
