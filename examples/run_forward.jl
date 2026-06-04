using JuliaLLM
using Statistics
using BFloat16s

function main()
    if length(ARGS) < 2
        println("Usage: julia --project examples/run_forward.jl <model_dir> <prompt> [dtype]")
        println("  dtype: f32 (default), f16, bf16")
        return
    end

    model_dir = ARGS[1]
    prompt = ARGS[2]
    
    # Parse dtype
    dtype_str = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "f32"
    dtype = if dtype_str == "f32"
        Float32
    elseif dtype_str == "f16"
        Float16
    elseif dtype_str == "bf16"
        BFloat16
    else
        error("Unsupported dtype: $dtype_str. Use f32, f16, or bf16.")
    end

    println("Loading model from $model_dir (dtype=$dtype)...")
    model = load_model(model_dir; dtype=dtype)
    
    tk = load_tokenizer(joinpath(model_dir, "tokenizer.json"))
    
    token_ids = encode(tk, prompt)
    println("Token IDs: $token_ids")
    
    println("Running forward pass...")
    logits = forward(model, token_ids)
    
    # Get top-1 for the last position
    last_logits = logits[:, end]
    top1_id = argmax(last_logits) - 1
    top1_token = get(tk.vocab, Int(top1_id), "ID:$top1_id")
    
    println("Top-1 next token: '$top1_token' (id=$top1_id)")
end

main()
