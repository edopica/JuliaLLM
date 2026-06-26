# JuliaLLM

A small, Julia-native, checkpoint-compatible implementation of **Qwen3 Base**, validated against a HuggingFace reference and benchmarked against `llama.cpp`.

This is an **educational project**, not a production inference engine.

---

## Goal

Finish a strong vertical slice within one month:

- load one real safetensors checkpoint
- run one modern dense decoder-only model end-to-end in Julia
- validate outputs against a Python/HuggingFace reference
- generate text with a KV cache
- compare narrowly and fairly with `llama.cpp`

---

## Quick start

```bash
# 1. Install dependencies
julia scripts/bootstrap.jl

# 2. Run tests
julia --project -e 'using Pkg; Pkg.test()'

# 3. Inspect a checkpoint
julia --project examples/inspect_checkpoint.jl /path/to/qwen3-0.6b/
```

---

## Repository layout

```
src/         Julia source (config, weights, norm, rope, attention, mlp, cache, model, generate)
test/        Unit tests for each primitive
examples/    Runnable scripts (inspect_checkpoint, run_forward, chat_greedy)
scripts/     Benchmark scripts and Python reference oracle
docs/        (placeholder)
reviews/     (moved to reports/)
reports/     Consolidated project documentation (Architecture, Benchmarks, Guides, Status)
```

---

## Model target

**Qwen3-0.6B Base** — [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

Download the checkpoint with:
```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/qwen3-0.6b
```

---

## Benchmark results

| Model | Precision | Prefill (tok/s) | Generation (tok/s) | Hardware |
| :--- | :--- | :---: | :---: | :--- |
| **Qwen3-0.6B** | Float32 | 13.7 | 3.1 | CPU (4-thread) |
| **Qwen3-0.6B** | Float16 | 19.6 | 9.5 | CPU (4-thread) |
| **Qwen3-4B** | Float32 | 2.0 | 0.5 | CPU (4-thread) |
| **Qwen3-4B** | Float16 | 2.4 | 1.6 | CPU (4-thread) |

See [reports/BENCHMARKS.md](reports/BENCHMARKS.md) for full analysis and comparison.

---

## Limitations

- Greedy decoding only (no sampling)
- Single model family (Qwen3)
- No custom tokenizer — depends on `tokenizer.json` from HuggingFace
- No native weight quantization (Int8/Int4)
