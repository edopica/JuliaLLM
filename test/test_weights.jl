using Test, SafeTensors, JuliaLLM

@testset "weights loading" begin
    mktempdir() do dir
        # 1. Generate test file
        @info "Testing tensorfile serialization"
        data = Dict(
            "Layer1.weights" => rand(Float32, 3, 2),
            "Layer1.biases" => rand(Float32, 3)
        )
        test_file = joinpath(dir, "test.safetensors")
        SafeTensors.serialize(test_file, data)

        # 2. Test weight loading functions
        @info "Testing tensorfile reading"
        st = load_weights(test_file)

        # Test list_tensors
        lt = list_tensors(st)
        @test length(lt) == 2
        @test lt[1] == ("Layer1.biases" => (3,))
        @test lt[2] == ("Layer1.weights" => (3, 2))

        # Test get_tensor
        w = get_tensor(st, "Layer1.weights")
        b = get_tensor(st, "Layer1.biases")

        @test w == data["Layer1.weights"]
        @test b == data["Layer1.biases"]
        @test size(w) == (3, 2)
        @test size(b) == (3,)
    end
end
