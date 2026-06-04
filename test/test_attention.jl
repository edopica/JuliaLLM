using Test
using JuliaLLM
using LinearAlgebra
using Statistics

@testset "Attention forward pass" begin
    cfg = ModelConfig(;
        vocab_size = 100,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2, # GQA
        head_dim = 4,
        intermediate_size = 32,
    )
    
    # Initialize weights
    # w_q: (num_heads * head_dim, hidden_size)
    # w_k: (num_kv_heads * head_dim, hidden_size)
    # w_v: (num_kv_heads * head_dim, hidden_size)
    # w_o: (hidden_size, num_heads * head_dim)
    w_q = randn(Float32, cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    w_k = randn(Float32, cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    w_v = randn(Float32, cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    w_o = randn(Float32, cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
    
    q_norm = ones(Float32, cfg.head_dim)
    k_norm = ones(Float32, cfg.head_dim)
    
    cos_cache, sin_cache = build_rope_cache(20, cfg.head_dim, 10000.0)

    @testset "Basic forward (no cache)" begin
        seq_len = 5
        x = randn(Float32, cfg.hidden_size, seq_len)
        
        out = attention_forward(
            x, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg
        )
        
        @test size(out) == (cfg.hidden_size, seq_len)
        @test all(isfinite.(out))
    end

    @testset "Forward with KV cache (GQA)" begin
        cache = KVCache(cfg, 10)
        
        # 1. Prompt phase (seq_len = 3)
        seq_len1 = 3
        x1 = randn(Float32, cfg.hidden_size, seq_len1)
        
        out1_l1 = attention_forward(
            x1, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg;
            kv_cache=cache,
            layer=1
        )
        
        @test size(out1_l1) == (cfg.hidden_size, seq_len1)
        @test !all(out1_l1 .== 0) # Should not be all zeros!
        
        out1_l2 = attention_forward(
            x1, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg;
            kv_cache=cache,
            layer=2
        )
        @test size(out1_l2) == (cfg.hidden_size, seq_len1)
        @test !all(out1_l2 .== 0)
        
        advance_cache!(cache, seq_len1)
        @test cache.seq_len == seq_len1
        
        # 2. Generation phase (seq_len = 1)
        x2 = randn(Float32, cfg.hidden_size, 1)
        out2_l1 = attention_forward(
            x2, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg;
            kv_cache=cache,
            layer=1
        )
        @test size(out2_l1) == (cfg.hidden_size, 1)
        @test !all(out2_l1 .== 0)
        
        out2_l2 = attention_forward(
            x2, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg;
            kv_cache=cache,
            layer=2
        )
        @test size(out2_l2) == (cfg.hidden_size, 1)
        @test !all(out2_l2 .== 0)
        
        advance_cache!(cache, 1)
        @test cache.seq_len == seq_len1 + 1
    end

    @testset "Explicit position_ids" begin
        seq_len = 2
        x = randn(Float32, cfg.hidden_size, seq_len)
        pos_ids = [10, 11]
        
        out = attention_forward(
            x, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg;
            position_ids=pos_ids
        )
        @test size(out) == (cfg.hidden_size, seq_len)
        @test all(isfinite.(out))
    end

    @testset "QK-Norm effect" begin
        # Use seq_len > 1 so that softmax isn't trivial
        seq_len = 2
        x = randn(Float32, cfg.hidden_size, seq_len)
        
        # Use very small q_norm/k_norm to see if it affects output
        q_norm_small = fill(0.001f0, cfg.head_dim)
        k_norm_small = fill(0.001f0, cfg.head_dim)
        
        out_normal = attention_forward(
            x, w_q, w_k, w_v, w_o,
            q_norm, k_norm,
            cos_cache, sin_cache,
            cfg
        )
        
        out_small = attention_forward(
            x, w_q, w_k, w_v, w_o,
            q_norm_small, k_norm_small,
            cos_cache, sin_cache,
            cfg
        )
        
        # They should be different
        @test !(out_normal ≈ out_small)
    end
end
