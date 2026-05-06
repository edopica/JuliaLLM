@testset "RMSNorm" begin
    # Scalar sanity: uniform input → output equals weight (after normalization)
    hidden = 8
    x = ones(Float32, hidden, 3)       # all ones, shape (hidden, seq_len)
    w = ones(Float32, hidden)
    eps = 1e-6

    out = rms_norm(x, w, Float64(eps))

    # RMS of ones is 1 → norm is identity → output ≈ weight (all ones)
    @test size(out) == size(x)
    @test all(isapprox.(out, 1.0f0; atol=1e-5))

    # Scale test: weight = 2 → output ≈ 2
    w2 = fill(2.0f0, hidden)
    out2 = rms_norm(x, w2, Float64(eps))
    @test all(isapprox.(out2, 2.0f0; atol=1e-5))

    # Shape preservation
    x3d = ones(Float32, hidden, 4, 2)
    out3d = rms_norm(x3d, w, Float64(eps))
    @test size(out3d) == size(x3d)
end
