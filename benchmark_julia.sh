#!/bin/bash
#SBATCH --job-name=julia_benchmark
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=reports/julia_benchmark_%j.out
#SBATCH --error=reports/julia_benchmark_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

export JULIA_NUM_THREADS=4

cd /home/3320522/JuliaLLM
mkdir -p reports/results

echo "Running Qwen3-0.6B Float32..."
julia --project scripts/benchmark_julia.jl /home/3320522/models/qwen3-0.6B f32 reports/results/julia_0.6b_f32.json

echo "Running Qwen3-0.6B Float16..."
julia --project scripts/benchmark_julia.jl /home/3320522/models/qwen3-0.6B f16 reports/results/julia_0.6b_f16.json

echo "Running Qwen3-4B Float32..."
julia --project scripts/benchmark_julia.jl /home/3320522/models/qwen3-4B f32 reports/results/julia_4b_f32.json

echo "Running Qwen3-4B Float16..."
julia --project scripts/benchmark_julia.jl /home/3320522/models/qwen3-4B f16 reports/results/julia_4b_f16.json
