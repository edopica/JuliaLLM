#!/bin/bash
#SBATCH --job-name=julia_bf16_bench
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=reports/julia_0.6b_bf16_%j.out
#SBATCH --error=reports/julia_0.6b_bf16_%j.err

cd /home/3320522/JuliaLLM
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

julia --project scripts/benchmark_julia.jl /home/3320522/models/qwen3-0.6B bf16 reports/results/julia_0.6b_bf16.json
