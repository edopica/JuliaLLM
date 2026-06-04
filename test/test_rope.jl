@testset "RoPE cache" begin
    seq_len  = 16
    head_dim = 64
    theta    = 10000.0

    cos_cache, sin_cache = build_rope_cache(seq_len, head_dim, theta)

    # Shape checks
    @test size(cos_cache) == (seq_len, head_dim)
    @test size(sin_cache) == (seq_len, head_dim)

    # Values are in [-1, 1]
    @test all(-1 .<= cos_cache .<= 1)
    @test all(-1 .<= sin_cache .<= 1)

    # Position 0 → angle=0 → cos=1, sin=0
    @test all(isapprox.(cos_cache[1, :], 1.0f0; atol=1e-5))
    @test all(isapprox.(sin_cache[1, :], 0.0f0; atol=1e-5))
end

@testset "apply_rope correctness" begin
    head_dim = 2
    seq_len  = 1
    theta    = 10000.0
    cos_c, sin_c = build_rope_cache(2, head_dim, theta)
    
    # Position 1 (0-indexed)
    # Input x: (head_dim, seq_len, n_heads, batch)
    x = reshape(Float32[1.0, 2.0], head_dim, seq_len, 1, 1)
    
    # For head_dim=2, theta=10000:
    # freqs = [1.0 / 10000^(0/2)] = [1.0]
    # pos = 1 => angle = 1.0 * 1.0 = 1.0
    c = cos(1.0f0)
    s = sin(1.0f0)
    
    # expected: [x1*c - x2*s, x1*s + x2*c]
    expected = [1.0f0*c - 2.0f0*s, 1.0f0*s + 2.0f0*c]
    
    out = apply_rope(x, cos_c, sin_c, [1])
    @test out ≈ reshape(expected, head_dim, seq_len, 1, 1)
    
    # Position 0 (0-indexed)
    # angle = 0 => cos=1, sin=0
    # expected: [1.0, 2.0]
    out0 = apply_rope(x, cos_c, sin_c, [0])
    @test out0 ≈ x
end
