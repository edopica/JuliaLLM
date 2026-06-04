using Test
using JuliaLLM
using LinearAlgebra

@testset "block_forward shape-only test" begin
    cfg = ModelConfig(;
        vocab_size = 100,
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 4,
        intermediate_size = 32,
        rms_norm_eps = 1e-6,
    )

    # Initialize weights
    w_q = randn(Float32, cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    w_k = randn(Float32, cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    w_v = randn(Float32, cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    w_o = randn(Float32, cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
    
    norm_attn = ones(Float32, cfg.hidden_size)
    q_norm = ones(Float32, cfg.head_dim)
    k_norm = ones(Float32, cfg.head_dim)
    
    w_gate = randn(Float32, cfg.intermediate_size, cfg.hidden_size)
    w_up   = randn(Float32, cfg.intermediate_size, cfg.hidden_size)
    w_down = randn(Float32, cfg.hidden_size, cfg.intermediate_size)
    norm_mlp = ones(Float32, cfg.hidden_size)
    
    weights = LayerWeights(
        w_q, w_k, w_v, w_o,
        norm_attn,
        q_norm, k_norm,
        w_gate, w_up, w_down,
        norm_mlp
    )
    
    cos_cache, sin_cache = build_rope_cache(20, cfg.head_dim, 10000.0)
    
    seq_len = 5
    x = randn(Float32, cfg.hidden_size, seq_len)
    
    out = block_forward(x, weights, cfg, cos_cache, sin_cache)
    
    @test size(out) == (cfg.hidden_size, seq_len)
    @test all(isfinite.(out))
    
    # Test with KV cache
    cache = KVCache(cfg, 10)
    out_cached = block_forward(x, weights, cfg, cos_cache, sin_cache; kv_cache=cache, layer=1)
    @test size(out_cached) == (cfg.hidden_size, seq_len)
    
    advance_cache!(cache, seq_len)
    @test cache.seq_len == seq_len
end
