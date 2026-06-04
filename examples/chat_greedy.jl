using JuliaLLM
using BFloat16s

function main()
    if length(ARGS) < 1
        println("Usage: julia --project examples/chat_greedy.jl <model_dir> [dtype]")
        println("  dtype: f32 (default), f16, bf16")
        return
    end

    model_dir = ARGS[1]
    
    # Parse dtype
    dtype_str = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "f32"
    dtype = if dtype_str == "f32"
        Float32
    elseif dtype_str == "f16"
        Float16
    elseif dtype_str == "bf16"
        BFloat16
    else
        error("Unsupported dtype: $dtype_str. Use f32, f16, or bf16.")
    end

    println("Loading model and tokenizer from $model_dir (dtype=$dtype)...")
    model = load_model(model_dir; dtype=dtype)
    tk = load_tokenizer(joinpath(model_dir, "tokenizer.json"))

    println("\nEntering chat mode (Greedy Decoding).")
    println("Type 'exit' to quit.\n")

    while true
        print("Prompt > ")
        prompt = readline()
        if isempty(prompt) || prompt == "exit"
            break
        end

        token_ids = encode(tk, prompt)
        
        println("Generating...")
        new_ids = greedy_generate(model, token_ids; max_new_tokens=100, eos_ids=tk.eos_token_id)
        
        # We can also stream the output if we want, but for now just print all at once
        response = decode(tk, new_ids)
        println("\nResponse: $response\n")
    end
end

main()
