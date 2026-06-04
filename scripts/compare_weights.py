import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/kurapica/models/qwen3-0.6b", torch_dtype=torch.float32)
ref = model.model.embed_tokens.weight.detach().numpy()

with open("julia_weights.json") as f:
    jl_raw = np.array(json.load(f)["embed_raw"])
    
print(f"Raw weight shape: ref={ref.shape}, jl={jl_raw.shape}")
print(f"Raw weight max diff: {np.max(np.abs(ref.flatten() - jl_raw))}")
