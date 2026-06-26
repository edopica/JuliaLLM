import JSON3
import SafeTensors
using Test, JuliaLLM, CUDA, Adapt

@testset "CUDA Support" begin
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        @test true
        return
    end

    @info "Testing on GPU: " * CUDA.name(CUDA.device())

    # 1. Test model transfer to GPU
    mktempdir() do dir
        # Minimal config
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

        model = load_model(dir)
        
        # Transfer to GPU
        gpu_model = cu(model)
        @test gpu_model.embed isa CuArray
        @test gpu_model.layers[1].w_q isa CuArray
        
        # Forward pass on GPU
        token_ids = [1, 2, 3]
        logits = forward(gpu_model, token_ids)
        @test logits isa CuArray
        @test size(logits) == (100, 3)
        
        # KV Cache on GPU
        cache = KVCache(model.cfg, 10; like=gpu_model.embed)
        @test cache.k isa CuArray
        
        logits2 = forward(gpu_model, [4]; kv_cache=cache)
        @test logits2 isa CuArray
    end
end
