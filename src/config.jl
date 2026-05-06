"""
Parse model config.json into a Julia struct.
Targets Qwen3 architecture fields; extend as needed.
"""

struct ModelConfig
    # Vocab / embedding
    vocab_size::Int
    hidden_size::Int
    # Transformer depth
    num_hidden_layers::Int
    # Attention
    num_attention_heads::Int
    num_key_value_heads::Int   # GQA: may be < num_attention_heads
    head_dim::Int
    # MLP
    intermediate_size::Int
    # Normalization
    rms_norm_eps::Float64
    # RoPE
    rope_theta::Float64
    max_position_embeddings::Int
end

"""
    load_config(path::AbstractString) -> ModelConfig

Load and parse a HuggingFace config.json into ModelConfig.
"""
function load_config(path::AbstractString)::ModelConfig
    raw = JSON3.read(read(path, String))

    vocab_size             = Int(raw[:vocab_size])
    hidden_size            = Int(raw[:hidden_size])
    num_hidden_layers      = Int(raw[:num_hidden_layers])
    num_attention_heads    = Int(raw[:num_attention_heads])
    # Qwen3 uses grouped-query attention; fall back to MHA if key absent
    num_key_value_heads    = Int(get(raw, :num_key_value_heads, num_attention_heads))
    head_dim               = Int(get(raw, :head_dim, hidden_size ÷ num_attention_heads))
    intermediate_size      = Int(raw[:intermediate_size])
    rms_norm_eps           = Float64(get(raw, :rms_norm_eps, 1e-6))
    rope_theta             = Float64(get(raw, :rope_theta, 10000.0))
    max_position_embeddings = Int(get(raw, :max_position_embeddings, 4096))

    return ModelConfig(
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
    )
end
