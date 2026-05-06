"""
Single Transformer decoder block:
  residual + attention(RMSNorm(x)) + residual + MLP(RMSNorm(x))

Placeholder until attention_forward is implemented.
"""

struct LayerWeights
    # Attention
    w_q::Matrix{Float32}
    w_k::Matrix{Float32}
    w_v::Matrix{Float32}
    w_o::Matrix{Float32}
    norm_attn::Vector{Float32}   # RMSNorm weight before attention
    # MLP
    w_gate::Matrix{Float32}
    w_up::Matrix{Float32}
    w_down::Matrix{Float32}
    norm_mlp::Vector{Float32}    # RMSNorm weight before MLP
end

"""
    block_forward(x, weights::LayerWeights, cfg::ModelConfig, ...) -> Array

TODO: implement once attention_forward is ready.
"""
function block_forward(x, weights::LayerWeights, cfg::ModelConfig,
                       cos_cache, sin_cache;
                       kv_cache=nothing, layer::Int=1, position_ids=nothing)
    error("block_forward: not yet implemented")
end
