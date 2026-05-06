@testset "block (stub)" begin
    # block_forward is not yet implemented; confirm it isn't silently callable.
    # Accepts any exception since the stub may throw MethodError or ErrorException.
    @test_throws Exception begin
        cfg = ModelConfig(100, 16, 2, 4, 2, 4, 32, 1e-6, 10000.0, 512)
        block_forward(nothing, nothing, cfg, nothing, nothing)
    end
end
