@testset "attention (stub)" begin
    # attention_forward is not yet implemented; just verify it errors predictably.
    # Replace this test once the implementation lands.
    cfg = ModelConfig(;
        vocab_size = 100, hidden_size = 16, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 4,
        intermediate_size = 32,
    )
    @test_throws ErrorException begin
        attention_forward(
            nothing, nothing, nothing, nothing, nothing,   # x, w_q, w_k, w_v, w_o
            nothing, nothing,                              # q_norm, k_norm
            nothing, nothing,                              # cos_cache, sin_cache
            cfg,
        )
    end
end
