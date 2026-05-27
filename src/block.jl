"""
Single Transformer decoder block:
  residual + attention(RMSNorm(x)) + residual + MLP(RMSNorm(x))

Qwen3 quirk: in addition to the usual pre-attention RMSNorm, every Qwen3 model
also RMS-normalizes the query and key tensors per-head **before RoPE**. The
weights for those QK-norms (`q_norm`, `k_norm`) are stored per layer with
shape (head_dim,) and live inside `LayerWeights` rather than next to the
attention projection matrices.

Placeholder until attention_forward is implemented.
"""

struct LayerWeights
    # Attention projections
    w_q::Matrix{Float32}
    w_k::Matrix{Float32}
    w_v::Matrix{Float32}
    w_o::Matrix{Float32}
    # Pre-attention RMSNorm (applied to the residual stream)
    norm_attn::Vector{Float32}
    # Qwen3 QK-Norm: RMSNorm applied per-head to Q and K before RoPE
    q_norm::Vector{Float32}      # (head_dim,)
    k_norm::Vector{Float32}      # (head_dim,)
    # MLP
    w_gate::Matrix{Float32}
    w_up::Matrix{Float32}
    w_down::Matrix{Float32}
    # Pre-MLP RMSNorm
    norm_mlp::Vector{Float32}
end

"""
    block_forward(x, weights::LayerWeights, cfg::ModelConfig, ...) -> Array

TODO: implement once attention_forward is ready. Order:
  1. y = rms_norm(x, weights.norm_attn, cfg.rms_norm_eps)
  2. y = attention_forward(y, ..., q_norm=weights.q_norm, k_norm=weights.k_norm)
  3. x = x + y
  4. y = rms_norm(x, weights.norm_mlp, cfg.rms_norm_eps)
  5. y = mlp_forward(y, weights.w_gate, weights.w_up, weights.w_down)
  6. x = x + y
"""
function block_forward(x, weights::LayerWeights, cfg::ModelConfig,
                       cos_cache, sin_cache;
                       kv_cache=nothing, layer::Int=1, position_ids=nothing)
    error("block_forward: not yet implemented")
end
