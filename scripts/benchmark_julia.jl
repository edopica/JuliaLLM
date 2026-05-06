"""
benchmark_julia.jl — measure prompt processing and token generation speed.

Usage:
    julia --project scripts/benchmark_julia.jl <model_dir>

Prints tokens/sec for:
  1. Prompt prefill (processing N tokens at once)
  2. Greedy decoding (one token at a time with KV cache)
"""

using JuliaLLM
using BenchmarkTools

println("benchmark_julia: not yet implemented. Complete forward() and greedy_generate() first.")
