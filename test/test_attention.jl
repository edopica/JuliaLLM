@testset "attention (stub)" begin
    # attention_forward is not yet implemented; just verify it errors predictably.
    # Replace this test once the implementation lands.
    @test_throws ErrorException begin
        cfg = ModelConfig(100, 16, 2, 4, 2, 4, 32, 1e-6, 10000.0, 512)
        attention_forward(nothing, nothing, nothing, nothing, nothing,
                          nothing, nothing, cfg)
    end
end
