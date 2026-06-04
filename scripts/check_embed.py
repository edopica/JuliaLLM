import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/home/kurapica/models/qwen3-0.6b", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("/home/kurapica/models/qwen3-0.6b")
input_ids = tok("The capital of France is", return_tensors="pt")["input_ids"]

embed_out = model.model.embed_tokens(input_ids)
weight_lookup = model.model.embed_tokens.weight[input_ids]

print("Diff between embed_tokens and direct lookup: ", (embed_out - weight_lookup).abs().max().item())
