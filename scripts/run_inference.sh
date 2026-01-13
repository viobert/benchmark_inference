#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/models.json}"
MODEL_KEY="${2:-}"
PROMPT_FILE="${3:-prompt/simple_vuln_prompt.txt}"
OUTPUT_FILE="${4:-reports/out.jsonl}"
EXTRA_ARGS=("${@:5}")

ARGS=(
  --config "$CONFIG_PATH"
  --prompt_file "$PROMPT_FILE"
  --output "$OUTPUT_FILE"
)

if [[ -n "$MODEL_KEY" ]]; then
  ARGS+=(--model "$MODEL_KEY")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

python src/evaluation/run_inference.py "${ARGS[@]}"
