using Test, SafeTensors, JSON3, JuliaLLM

@testset "CheckpointBundle" begin
    mktempdir() do dir
        # 1. Single-file case
        sub_dir1 = joinpath(dir, "single")
        mkdir(sub_dir1)
        data1 = Dict(
            "tensor1" => rand(Float32, 2, 2),
            "tensor2" => rand(Float32, 3)
        )
        SafeTensors.serialize(joinpath(sub_dir1, "model.safetensors"), data1)
        
        bundle1 = load_checkpoint(sub_dir1)
        @test length(bundle1.shards) == 1
        @test length(bundle1.index) == 2
        @test get_tensor(bundle1, "tensor1") == data1["tensor1"]
        @test get_tensor(bundle1, "tensor2") == data1["tensor2"]
        
        lt1 = list_tensors(bundle1)
        @test lt1[1] == ("tensor1" => (2, 2))
        @test lt1[2] == ("tensor2" => (3,))

        # 2. Sharded case
        sub_dir2 = joinpath(dir, "sharded")
        mkdir(sub_dir2)
        shard1_data = Dict("layer1.w" => rand(Float32, 4, 4))
        shard2_data = Dict("layer2.w" => rand(Float32, 2, 2))
        
        SafeTensors.serialize(joinpath(sub_dir2, "model-00001-of-00002.safetensors"), shard1_data)
        SafeTensors.serialize(joinpath(sub_dir2, "model-00002-of-00002.safetensors"), shard2_data)
        
        index_data = Dict(
            "metadata" => Dict("total_size" => 1234),
            "weight_map" => Dict(
                "layer1.w" => "model-00001-of-00002.safetensors",
                "layer2.w" => "model-00002-of-00002.safetensors"
            )
        )
        open(joinpath(sub_dir2, "model.safetensors.index.json"), "w") do io
            JSON3.write(io, index_data)
        end
        
        bundle2 = load_checkpoint(sub_dir2)
        @test length(bundle2.shards) == 2
        @test length(bundle2.index) == 2
        @test get_tensor(bundle2, "layer1.w") == shard1_data["layer1.w"]
        @test get_tensor(bundle2, "layer2.w") == shard2_data["layer2.w"]
        
        lt2 = list_tensors(bundle2)
        @test lt2[1] == ("layer1.w" => (4, 4))
        @test lt2[2] == ("layer2.w" => (2, 2))
    end
end
