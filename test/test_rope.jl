@testset "RoPE cache" begin
    seq_len  = 16
    head_dim = 64
    theta    = 10000.0

    cos_cache, sin_cache = build_rope_cache(seq_len, head_dim, theta)

    # Shape checks
    @test size(cos_cache) == (seq_len, head_dim ÷ 2)
    @test size(sin_cache) == (seq_len, head_dim ÷ 2)

    # Values are in [-1, 1]
    @test all(-1 .<= cos_cache .<= 1)
    @test all(-1 .<= sin_cache .<= 1)

    # Position 0 → angle=0 → cos=1, sin=0
    @test all(isapprox.(cos_cache[1, :], 1.0f0; atol=1e-5))
    @test all(isapprox.(sin_cache[1, :], 0.0f0; atol=1e-5))
end

@testset "apply_rope placeholder" begin
    # apply_rope currently returns input unchanged (stub)
    x = randn(Float32, 64, 4, 8, 1)
    cos_c, sin_c = build_rope_cache(4, 64, 10000.0)
    out = apply_rope(x, cos_c, sin_c, collect(0:3))
    @test size(out) == size(x)
end
