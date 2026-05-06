#!/usr/bin/env bash
# benchmark_llamacpp.sh — run llama.cpp benchmark for comparison.
#
# Usage:
#   ./scripts/benchmark_llamacpp.sh /path/to/llama.cpp /path/to/model.gguf
#
# Records:
#   - prompt processing speed (pp)
#   - token generation speed (tg)

set -euo pipefail

LLAMA_BIN="${1:?Usage: $0 <llama-cpp-dir> <model.gguf>}"
MODEL="${2:?Usage: $0 <llama-cpp-dir> <model.gguf>}"
PROMPT="Hello, my name is"
N_PROMPT=8
N_GEN=50

echo "=== llama.cpp benchmark ==="
echo "Model: $MODEL"
echo "Prompt tokens: $N_PROMPT"
echo "Generated tokens: $N_GEN"
echo ""

"${LLAMA_BIN}/llama-bench" \
    -m "$MODEL" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -r 3

echo ""
echo "Done. Copy results to README comparison table."
