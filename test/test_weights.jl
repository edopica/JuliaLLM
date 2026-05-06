using Test, SafeTensors, JuliaLLM

@testset "weights loading" begin
    mktempdir() do dir
        # 1. Generate test file
        @info "Testing tensorfile serialization"
        data = Dict((
            "Layer1.weights" => rand(Float32, 3, 3),
            "Layer1.biases" => rand(Float32, 3, 3)
        ))
        test_file = joinpath(dir, "test.safetensors")
        SafeTensors.serialize(test_file, data)
        # 2. Test weight loading functions
        @info "Testing tensorfile reading"
        tensors = load_weights(test_file)

        @test "Layer1.weights" in keys(tensors)
        @test collect(values(data)) == collect(values(tensors))
    end
end
