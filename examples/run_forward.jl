using JuliaLLM
using Statistics

function main()
    if length(ARGS) < 2
        println("Usage: julia --project examples/run_forward.jl <model_dir> <prompt>")
        return
    end

    model_dir = ARGS[1]
    prompt = ARGS[2]

    println("Loading model from $model_dir...")
    model = load_model(model_dir)
    
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
