#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import argparse
import os
import platform
import psutil

def get_platform_info():
    info = {
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cores_logical": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_version": torch.__version__
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info

def benchmark(model_id, device, dtype, n_prompt, n_gen, reps):
    print(f"Loading model {model_id} on {device} with {dtype}...")
    
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Ensure n_prompt matches the requested size by padding/repeating if necessary
    if input_ids.shape[1] < n_prompt:
        # Simple repeat for benchmarking purposes
        input_ids = input_ids.repeat(1, (n_prompt // input_ids.shape[1]) + 1)[:, :n_prompt]
    
    print(f"Running benchmark (n_prompt={n_prompt}, n_gen={n_gen}, reps={reps})...")
    
    prefill_times = []
    gen_times = []
    
    for i in range(reps + 1): # +1 for warmup
        # 1. Benchmark Prefill
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        
        if i > 0: # Skip warmup
            prefill_times.append(end - start)
            
        # 2. Benchmark Generation
        # Use KV Cache to match Julia's optimized methodology
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            for _ in range(n_gen - 1): # -1 because we already got the first token from prefill
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                past_key_values = outputs.past_key_values
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        
        if i > 0: # Skip warmup
            gen_times.append(end - start)

    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_gen_time = sum(gen_times) / len(gen_times)
    
    results = {
        "prefill_tok_sec": round(n_prompt / avg_prefill_time, 2),
        "prefill_n": n_prompt,
        "generation_tok_sec": round(n_gen / avg_gen_time, 2),
        "generation_n": n_gen
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face Transformers")
    parser.add_argument("model_id", type=str, help="Hugging Face model ID or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--n_prompt", type=int, default=8)
    parser.add_argument("--n_gen", type=int, default=50)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=str, help="Path to save JSON output")
    
    args = parser.parse_args()
    
    results = benchmark(args.model_id, args.device, args.dtype, args.n_prompt, args.n_gen, args.reps)
    platform_info = get_platform_info()
    
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model_id,
        "dtype": args.dtype,
        "device": args.device,
        "platform": platform_info,
        "results": results
    }
    
    print("\n=== Hugging Face Benchmark Summary ===")
    print(f"Prefill:    {results['prefill_tok_sec']:.2f} tok/s (N={results['prefill_n']})")
    print(f"Generation: {results['generation_tok_sec']:.2f} tok/s (N={results['generation_n']})")
    print("========================================")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON report saved to: {args.output}")

if __name__ == "__main__":
    main()
