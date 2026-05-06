#!/usr/bin/env python3
"""
compare_reference.py — run Qwen3-0.6B through HuggingFace transformers
and dump logits / top-1 token for a fixed prompt.

The Julia implementation should reproduce these values.

Usage:
    pip install transformers torch
    python scripts/compare_reference.py --model_dir /path/to/qwen3-0.6b --prompt "Hello"
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--prompt", default="Hello", help="Input prompt string")
    parser.add_argument("--output", default="reference_logits.json", help="Output file for logits")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"Loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token ids: {input_ids.tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # (seq_len, vocab_size)
    top1 = logits[-1].argmax().item()
    top1_token = tokenizer.decode([top1])

    result = {
        "prompt": args.prompt,
        "token_ids": input_ids.tolist()[0],
        "top1_token_id": top1,
        "top1_token": top1_token,
        # Save only last-position logits to keep file small
        "last_logits": logits[-1].tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    print(f"Top-1 next token: {top1_token!r} (id={top1})")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
