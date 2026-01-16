#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="config/models.json"
MODEL_KEY=""
PROMPT_FILE="prompt/simple_vuln_prompt.txt"
DATASET="/home/skl/mkx/data/big_vul/datasets/big_vul"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_KEY="${2:-}"
      shift 2
      ;;
    --prompt_file)
      PROMPT_FILE="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  --config "$CONFIG_PATH"
  --prompt_file "$PROMPT_FILE"
  --dataset "${DATASET:-}"
  --limit 10
)

if [[ -n "$MODEL_KEY" ]]; then
  ARGS+=(--model "$MODEL_KEY")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

python -m src.evaluation.run_inference "${ARGS[@]}"

PID=$!
echo "-----------------------------------------"
echo "Inference process started, PID=${PID}"
echo "-----------------------------------------"