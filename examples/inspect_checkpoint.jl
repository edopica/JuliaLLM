"""
inspect_checkpoint.jl — print config fields and tensor names/shapes from a checkpoint.

Usage:
    julia --project examples/inspect_checkpoint.jl /path/to/model/

The model directory should contain config.json and one or more .safetensors files.
"""

using JuliaLLM

if length(ARGS) < 1
    println("Usage: julia --project examples/inspect_checkpoint.jl <model_dir>")
    exit(1)
end

model_dir = ARGS[1]

# --- Config ---
cfg_path = joinpath(model_dir, "config.json")
println("Loading config from: ", cfg_path)
cfg = load_config(cfg_path)
println()
println("=== ModelConfig ===")
for field in fieldnames(ModelConfig)
    println("  ", field, " = ", getfield(cfg, field))
end

# --- Weights ---
println()
println("=== Tensors ===")
for fname in filter(f -> endswith(f, ".safetensors"), readdir(model_dir; join=true))
    println("File: ", fname)
    st = load_weights(fname)
    #"""
    for (name, shape) in list_tensors(st)
        #println("  ", name, "  ", size(shape))
        println("  ", name, "  ", shape)
    end
    #"""
    #print(list_tensors(st))
end
