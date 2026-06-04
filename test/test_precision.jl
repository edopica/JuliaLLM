using Test, JuliaLLM, SafeTensors, JSON3, BFloat16s

@testset "Precision and Generic Support" begin
    # 1. Test rms_norm generic support (Float16 and BFloat16)
    for T in [Float16, BFloat16]
        x = rand(T, 16, 2)
        w = rand(T, 16)
        y = rms_norm(x, w)
        @test eltype(y) == T
    end

    # 2. Test build_rope_cache generic support
    for T in [Float16, BFloat16]
        cos_c, sin_c = build_rope_cache(10, 8, 10000.0; T=T)
        @test eltype(cos_c) == T
        @test eltype(sin_c) == T
    end

    # 3. Test load_model with different dtypes
    for T in [Float16, BFloat16]
        mktempdir() do dir
            # ... (rest of setup)
            cfg_data = Dict(
                "vocab_size" => 100,
                "hidden_size" => 16,
                "num_hidden_layers" => 1,
                "num_attention_heads" => 4,
                "intermediate_size" => 32,
                "head_dim" => 4
            )
            open(joinpath(dir, "config.json"), "w") do io
                JSON3.write(io, cfg_data)
            end

            weights = Dict(
                "model.embed_tokens.weight" => rand(Float32, 100, 16),
                "model.norm.weight" => rand(Float32, 16),
                "model.layers.0.input_layernorm.weight" => rand(Float32, 16),
                "model.layers.0.self_attn.q_proj.weight" => rand(Float32, 16, 16),
                "model.layers.0.self_attn.k_proj.weight" => rand(Float32, 16, 16),
                "model.layers.0.self_attn.v_proj.weight" => rand(Float32, 16, 16),
                "model.layers.0.self_attn.o_proj.weight" => rand(Float32, 16, 16),
                "model.layers.0.self_attn.q_norm.weight" => rand(Float32, 4),
                "model.layers.0.self_attn.k_norm.weight" => rand(Float32, 4),
                "model.layers.0.post_attention_layernorm.weight" => rand(Float32, 16),
                "model.layers.0.mlp.gate_proj.weight" => rand(Float32, 32, 16),
                "model.layers.0.mlp.up_proj.weight" => rand(Float32, 32, 16),
                "model.layers.0.mlp.down_proj.weight" => rand(Float32, 16, 32),
                "lm_head.weight" => rand(Float32, 100, 16),
            )
            SafeTensors.serialize(joinpath(dir, "model.safetensors"), weights)

            model = load_model(dir; dtype=T)
            @test eltype(model.embed) == T
            
            # Check forward pass
            logits = forward(model, [1, 2, 3])
            @test eltype(logits) == T
        end
    end
end
