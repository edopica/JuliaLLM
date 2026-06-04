#!/usr/bin/env bash
# benchmark_llamacpp.sh — run llama.cpp benchmark and save results as JSON.
#
# Usage:
#   ./scripts/benchmark_llamacpp.sh /path/to/llama.cpp /path/to/model.gguf [output.json]

set -euo pipefail

LLAMA_BIN_DIR="${1:?Usage: $0 <llama-cpp-bin-dir> <model.gguf> [output.json]}"
MODEL="${2:?Usage: $0 <llama-cpp-bin-dir> <model.gguf> [output.json]}"
OUTPUT_PATH="${3:-}"

N_PROMPT=8
N_GEN=50
REPS=3

echo "=== llama.cpp benchmark ==="
echo "Model: $MODEL"

# 1. Run benchmark and capture JSON output
# llama-bench -o json returns a JSON array of results
RAW_RESULTS=$( "${LLAMA_BIN_DIR}/llama-bench" -m "$MODEL" -p "$N_PROMPT" -n "$N_GEN" -r "$REPS" -o json )

# 2. Gather System Info
OS=$(uname -s)
MACHINE=$(uname -m)
CPU_MODEL=$(lscpu | grep "Model name" | head -n 1 | cut -d ':' -f 2 | xargs || echo "Unknown")
CORES=$(nproc)
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}' || echo "0")
TIMESTAMP=$(date -Iseconds)

# 3. Process with Python to match Julia's JSON structure
# llama-bench JSON format is a list of objects. 
# One entry for n_prompt > 0 (prefill) and one for n_gen > 0 (generation).
FINAL_JSON=$(python3 - <<EOF
import json
import sys

raw_list = json.loads('''$RAW_RESULTS''')
prefill_data = next((x for x in raw_list if x["n_prompt"] > 0), None)
gen_data = next((x for x in raw_list if x["n_gen"] > 0), None)

if not prefill_data or not gen_data:
    print(f"Error: Missing benchmark data in llama-bench output.", file=sys.stderr)
    sys.exit(1)

platform = {
    "os": "$OS",
    "machine": "$MACHINE",
    "cpu": "$CPU_MODEL",
    "cores_logical": int("$CORES"),
    "memory_gb": float("$MEMORY_GB"),
    "llama_cpp_version": prefill_data.get("build_number", "unknown")
}

data = {
    "timestamp": "$TIMESTAMP",
    "model": "$MODEL",
    "platform": platform,
    "results": {
        "prefill_tok_sec": round(prefill_data["avg_ts"], 2),
        "prefill_n": int(prefill_data["n_prompt"]),
        "generation_tok_sec": round(gen_data["avg_ts"], 2),
        "generation_n": int(gen_data["n_gen"])
    }
}
print(json.dumps(data, indent=2))
EOF
)

# 4. Output
echo ""
echo "=== llama.cpp Benchmark Summary ==="
echo "$FINAL_JSON" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f'Prefill:    {d[\"results\"][\"prefill_tok_sec\"]:.2f} tok/s (N={d[\"results\"][\"prefill_n\"]})'); print(f'Generation: {d[\"results\"][\"generation_tok_sec\"]:.2f} tok/s (N={d[\"results\"][\"generation_n\"]})')"
echo "===================================="

if [[ -n "$OUTPUT_PATH" ]]; then
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    echo "$FINAL_JSON" > "$OUTPUT_PATH"
    echo "JSON report saved to: $OUTPUT_PATH"
fi
