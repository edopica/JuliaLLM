using JuliaLLM

function main()
    if length(ARGS) < 1
        println("Usage: julia --project examples/chat_greedy.jl <model_dir>")
        return
    end

    model_dir = ARGS[1]

    println("Loading model and tokenizer from $model_dir...")
    model = load_model(model_dir)
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
        new_ids = greedy_generate(model, token_ids; max_new_tokens=100)
        
        # We can also stream the output if we want, but for now just print all at once
        response = decode(tk, new_ids)
        println("\nResponse: $response\n")
    end
end

main()
