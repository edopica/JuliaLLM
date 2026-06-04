import json
import numpy as np

def max_diff(a, b):
    return np.max(np.abs(np.array(a) - np.array(b)))

with open("reference_intermediates.json") as f:
    ref = json.load(f)
with open("julia_intermediates.json") as f:
    jl = json.load(f)

for key in sorted(ref.keys(), key=lambda x: int(x.split("layer")[1].split("_")[0]) if "layer" in x else -1):
    print(f"{key} diff: {max_diff(ref[key], jl[key]):.6f}")
