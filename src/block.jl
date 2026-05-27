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

struct LayerWeights{M<:AbstractMatrix, V<:AbstractVector}
    # Attention projections
    w_q::M
    w_k::M
    w_v::M
    w_o::M
    # Pre-attention RMSNorm (applied to the residual stream)
    norm_attn::V
    # Qwen3 QK-Norm: RMSNorm applied per-head to Q and K before RoPE
    q_norm::V      # (head_dim,)
    k_norm::V      # (head_dim,)
    # MLP
    w_gate::M
    w_up::M
    w_down::M
    # Pre-MLP RMSNorm
    norm_mlp::V
end

# Adapt walks each field with `to`, so `adapt(CuArray, layer)` moves all weights to the GPU.
Adapt.adapt_structure(to, w::LayerWeights) = LayerWeights(
    adapt(to, w.w_q),    adapt(to, w.w_k),    adapt(to, w.w_v),  adapt(to, w.w_o),
    adapt(to, w.norm_attn),
    adapt(to, w.q_norm), adapt(to, w.k_norm),
    adapt(to, w.w_gate), adapt(to, w.w_up),   adapt(to, w.w_down),
    adapt(to, w.norm_mlp),
)

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
