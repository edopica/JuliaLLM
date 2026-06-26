#!/usr/bin/env bash
# benchmark_hf.sh — run Hugging Face benchmark and save results as JSON.
#
# Usage:
#   ./scripts/benchmark_hf.sh <model_id_or_path> <device> <dtype> [output.json]

set -euo pipefail

MODEL="${1:?Usage: $0 <model_id_or_path> <device> <dtype> [output.json]}"
DEVICE="${2:-cuda}"
DTYPE="${3:-float16}"
OUTPUT_PATH="${4:-}"

# Check if we are in a virtualenv/conda env, or use python3
PYTHON_BIN=$(which python3)

echo "=== Hugging Face benchmark ==="
echo "Model:  $MODEL"
echo "Device: $DEVICE"
echo "DType:  $DTYPE"

ARGS=(
    "$MODEL"
    "--device" "$DEVICE"
    "--dtype" "$DTYPE"
    "--n_prompt" "8"
    "--n_gen" "50"
    "--reps" "3"
)

if [[ -n "$OUTPUT_PATH" ]]; then
    ARGS+=("--output" "$OUTPUT_PATH")
fi

"$PYTHON_BIN" scripts/benchmark_hf.py "${ARGS[@]}"
