using JuliaLLM
using JSON3
using Printf
using Dates

function main()
    if length(ARGS) < 1
        println("Usage: julia --project scripts/benchmark_julia.jl <model_dir> [output_path]")
        exit(1)
    end

    model_dir = ARGS[1]
    output_path = length(ARGS) >= 2 ? ARGS[2] : nothing
    
    println("Loading model from: ", model_dir)

    model = load_model(model_dir)
    config = model.cfg
    tokenizer = load_tokenizer(joinpath(model_dir, "tokenizer.json"))

    prompt_text = "Hello, my name is"
    prompt_ids = encode(tokenizer, prompt_text)
    N_PROMPT = length(prompt_ids)
    N_GEN = 50

    println("Prompt: '", prompt_text, "' (", N_PROMPT, " tokens)")
    println("Generating ", N_GEN, " tokens...")

    # -----------------
    # 1. Warm-up
    # -----------------
    println("Running warm-up...")
    cache_warmup = KVCache(config, N_PROMPT + N_GEN; like=model.embed)
    logits_warmup = forward(model, prompt_ids; kv_cache=cache_warmup)
    next_id = Int(argmax(view(logits_warmup, :, size(logits_warmup, 2)))) - 1
    forward(model, [next_id]; kv_cache=cache_warmup)

    # -----------------
    # 2. Benchmark Prefill
    # -----------------
    n_runs = 3
    prefill_time = 0.0
    for i in 1:n_runs
        cache = KVCache(config, N_PROMPT + N_GEN; like=model.embed)
        t0 = time_ns()
        logits = forward(model, prompt_ids; kv_cache=cache)
        t1 = time_ns()
        prefill_time += (t1 - t0) / 1e9
    end
    avg_prefill_time = prefill_time / n_runs
    prefill_tok_sec = N_PROMPT / avg_prefill_time

    # -----------------
    # 3. Benchmark Generation
    # -----------------
    gen_time = 0.0
    generated_tokens_total = 0
    for i in 1:n_runs
        cache = KVCache(config, N_PROMPT + N_GEN; like=model.embed)
        logits = forward(model, prompt_ids; kv_cache=cache)
        next_id = Int(argmax(view(logits, :, size(logits, 2)))) - 1
        
        t0 = time_ns()
        n_gen_actual = 0
        for _ in 1:N_GEN
            logits = forward(model, [next_id]; kv_cache=cache)
            next_id = Int(argmax(view(logits, :, 1))) - 1
            n_gen_actual += 1
        end
        t1 = time_ns()
        
        gen_time += (t1 - t0) / 1e9
        generated_tokens_total += n_gen_actual
    end

    avg_gen_time = gen_time / n_runs
    avg_n_gen = generated_tokens_total / n_runs
    gen_tok_sec = avg_n_gen / avg_gen_time

    # -----------------
    # 4. Report
    # -----------------
    cpu_info = Sys.cpu_info()
    cpu_model = length(cpu_info) > 0 ? cpu_info[1].model : "Unknown"
    
    data = Dict(
        "timestamp" => string(now()),
        "model" => model_dir,
        "platform" => Dict(
            "os" => (Sys.islinux() ? "Linux" : Sys.isapple() ? "macOS" : Sys.iswindows() ? "Windows" : "Unknown"),
            "machine" => Sys.MACHINE,
            "cpu" => cpu_model,
            "cores_logical" => Sys.CPU_THREADS,
            "memory_gb" => round(Sys.total_memory() / 1024^3, digits=2),
            "julia_version" => string(VERSION)
        ),
        "results" => Dict(
            "prefill_tok_sec" => round(prefill_tok_sec, digits=2),
            "prefill_n" => N_PROMPT,
            "generation_tok_sec" => round(gen_tok_sec, digits=2),
            "generation_n" => Int(avg_n_gen)
        )
    )

    report_json = JSON3.write(data)
    
    # Print a summary to console
    println("\n=== JuliaLLM Benchmark Summary ===")
    @printf "Prefill:    %.2f tok/s (N=%d)\n" prefill_tok_sec N_PROMPT
    @printf "Generation: %.2f tok/s (N=%d)\n" gen_tok_sec Int(avg_n_gen)
    println("==================================")

    if output_path !== nothing
        mkpath(dirname(output_path))
        open(output_path, "w") do io
            JSON3.pretty(io, data)
        end
        println("JSON report saved to: $output_path")
    end
end

main()
