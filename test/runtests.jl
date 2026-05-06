using Test
using JuliaLLM

@testset "JuliaLLM" begin
    include("test_config.jl")
    include("test_weights.jl")
    include("test_norm.jl")
    include("test_rope.jl")
    include("test_attention.jl")
    include("test_block.jl")
    include("test_generation.jl")
end
