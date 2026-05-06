"""
Full Qwen3 model: embedding → N blocks → final norm → LM head.
Placeholder until block_forward is implemented.
"""

struct QwenModel
    cfg::ModelConfig
    embed::Matrix{Float32}         # (vocab_size, hidden_size) — rows are token embeddings
    layers::Vector{LayerWeights}
    final_norm::Vector{Float32}    # (hidden_size,)
    lm_head::Matrix{Float32}       # (vocab_size, hidden_size)
end

"""
    forward(model::QwenModel, token_ids::Vector{Int};
            kv_cache=nothing) -> Matrix{Float32}

Run the full forward pass.
  token_ids: (seq_len,) — 0-indexed token ids from the tokenizer
  Returns logits: (vocab_size, seq_len)
"""
function forward(model::QwenModel, token_ids::Vector{Int}; kv_cache=nothing)
    error("forward: not yet implemented")
end
