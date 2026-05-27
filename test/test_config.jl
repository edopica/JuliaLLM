@testset "config loading" begin
    # Write a minimal synthetic config and parse it
    mktempdir() do dir
        cfg_path = joinpath(dir, "config.json")
        write(cfg_path, """
        {
          "vocab_size": 151936,
          "hidden_size": 1024,
          "num_hidden_layers": 28,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "head_dim": 64,
          "intermediate_size": 3072,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "max_position_embeddings": 32768,
          "tie_word_embeddings": true
        }
        """)

        cfg = load_config(cfg_path)

        @test cfg.vocab_size == 151936
        @test cfg.hidden_size == 1024
        @test cfg.num_hidden_layers == 28
        @test cfg.num_attention_heads == 16
        @test cfg.num_key_value_heads == 8
        @test cfg.head_dim == 64
        @test cfg.intermediate_size == 3072
        @test cfg.rms_norm_eps ≈ 1e-6
        @test cfg.rope_theta ≈ 10000.0
        @test cfg.max_position_embeddings == 32768
        @test cfg.tie_word_embeddings == true
    end
end
