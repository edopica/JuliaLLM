@testset "block (stub)" begin
    # block_forward is not yet implemented; confirm it isn't silently callable.
    # Accepts any exception since the stub may throw MethodError or ErrorException.
    cfg = ModelConfig(;
        vocab_size = 100, hidden_size = 16, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 4,
        intermediate_size = 32,
    )
    @test_throws Exception block_forward(nothing, nothing, cfg, nothing, nothing)
end
