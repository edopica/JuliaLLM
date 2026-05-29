using JuliaLLM
using Test
using LinearAlgebra

@testset "Greedy Generation" begin
    # Create a tiny model
    cfg = ModelConfig(
        vocab_size=10,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        intermediate_size=8,
        rms_norm_eps=1e-5,
        tie_word_embeddings=true
    )
    
    # Random weights
    embed = randn(Float32, cfg.hidden_size, cfg.vocab_size)
    layer = LayerWeights{Matrix{Float32}, Vector{Float32}}(
        randn(Float32, 4, 4), randn(Float32, 4, 4), randn(Float32, 4, 4), randn(Float32, 4, 4),
        ones(Float32, 4),
        ones(Float32, 2), ones(Float32, 2),
        randn(Float32, 8, 4), randn(Float32, 8, 4), randn(Float32, 4, 8),
        ones(Float32, 4)
    )
    
    model = QwenModel{Matrix{Float32}, Vector{Float32}}(
        cfg,
        embed,
        [layer],
        ones(Float32, 4),
        collect(embed')
    )
    
    # Test generation doesn't crash
    prompt_ids = [1, 2, 3]
    new_ids = greedy_generate(model, prompt_ids; max_new_tokens=5)
    
    @test length(new_ids) == 5
    @test all(0 .<= new_ids .< cfg.vocab_size)
end
