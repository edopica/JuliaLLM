#!/bin/bash
#SBATCH --job-name=julia_tests
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=reports/julia_tests_%j.out
#SBATCH --error=reports/julia_tests_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

export JULIA_NUM_THREADS=4

cd /home/3320522/JuliaLLM

echo "--- ENVIRONMENT INFO ---"
date
nvidia-smi
julia --project -e 'using CUDA; println("CUDA functional: ", CUDA.functional()); if CUDA.functional() println("Device: ", CUDA.name(CUDA.device())) end'

echo "--- RUNNING TESTS ---"
julia --project -e 'using Pkg; Pkg.test()'
