"""
Parse model config.json into a Julia struct.

Targets the Qwen3 dense family (0.6B, 1.7B, 4B, 8B, 14B, 32B). The fields below
are the union of what those variants set in their config.json; values that vary
across sizes (hidden_size, num_hidden_layers, GQA ratio, intermediate_size,
tie_word_embeddings, ...) are all loaded from JSON, so a single code path
handles every dense Qwen3.
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
    head_dim::Int              # explicit in Qwen3 — do not derive from hidden_size
    # MLP
    intermediate_size::Int
    # Normalization
    rms_norm_eps::Float64
    # RoPE
    rope_theta::Float64
    max_position_embeddings::Int
    # Weight tying: true for 0.6B / 1.7B, false for 4B+
    tie_word_embeddings::Bool
end

"""
    ModelConfig(; vocab_size, hidden_size, ...) -> ModelConfig

Keyword constructor with sensible defaults for the optional fields. Use this in
tests to avoid breaking when new fields are added.
"""
function ModelConfig(;
    vocab_size::Integer,
    hidden_size::Integer,
    num_hidden_layers::Integer,
    num_attention_heads::Integer,
    num_key_value_heads::Integer,
    head_dim::Integer,
    intermediate_size::Integer,
    rms_norm_eps::Real = 1e-6,
    rope_theta::Real = 1_000_000.0,        # Qwen3 default
    max_position_embeddings::Integer = 32_768,
    tie_word_embeddings::Bool = false,
)
    return ModelConfig(
        Int(vocab_size),
        Int(hidden_size),
        Int(num_hidden_layers),
        Int(num_attention_heads),
        Int(num_key_value_heads),
        Int(head_dim),
        Int(intermediate_size),
        Float64(rms_norm_eps),
        Float64(rope_theta),
        Int(max_position_embeddings),
        tie_word_embeddings,
    )
end

"""
    load_config(path::AbstractString) -> ModelConfig

Load and parse a HuggingFace config.json into ModelConfig.

All Qwen3 dense variants expose the same set of keys; only the values change.
`tie_word_embeddings` is read explicitly because the larger variants (4B+)
ship a separate `lm_head.weight` and must not alias the embedding matrix.
"""
function load_config(path::AbstractString)::ModelConfig
    raw = JSON3.read(read(path, String))

    vocab_size              = Int(raw[:vocab_size])
    hidden_size             = Int(raw[:hidden_size])
    num_hidden_layers       = Int(raw[:num_hidden_layers])
    num_attention_heads     = Int(raw[:num_attention_heads])
    # Qwen3 uses grouped-query attention; fall back to MHA if key absent
    num_key_value_heads     = Int(get(raw, :num_key_value_heads, num_attention_heads))
    head_dim                = Int(get(raw, :head_dim, hidden_size ÷ num_attention_heads))
    intermediate_size       = Int(raw[:intermediate_size])
    rms_norm_eps            = Float64(get(raw, :rms_norm_eps, 1e-6))
    rope_theta              = Float64(get(raw, :rope_theta, 1_000_000.0))
    max_position_embeddings = Int(get(raw, :max_position_embeddings, 32_768))
    tie_word_embeddings     = Bool(get(raw, :tie_word_embeddings, false))

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
        tie_word_embeddings,
    )
end
