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
reviews/     Progress reports and next-step notes
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

> **Not yet available.** Will be added after Milestone 6 (generation).

---

## Limitations

- CPU only (no GPU support in v1)
- Float32 full precision (no quantization)
- Greedy decoding only (no sampling)
- Single model family (Qwen3)
- No custom tokenizer — depends on `tokenizer.json` from HuggingFace
