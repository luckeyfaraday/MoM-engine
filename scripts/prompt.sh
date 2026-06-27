#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/prompt.sh 'your prompt here'" >&2
  exit 2
fi

prompt="$*"
base_url="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
model="${MODEL:-mom-chat}"

curl -sS "${base_url}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg model "$model" --arg prompt "$prompt" '{
    model: $model,
    messages: [{role: "user", content: $prompt}]
  }')" | jq -r '.choices[0].message.content // .'
