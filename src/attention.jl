"""
Multi-head / Grouped-Query Attention forward pass.

All Qwen3 dense variants use GQA (num_key_value_heads ≤ num_attention_heads)
and apply RMSNorm to the per-head Q and K tensors **before RoPE**. The QK-norm
weights are passed in as `q_norm` and `k_norm` of shape (head_dim,).
"""

"""
    attention_forward(
        x, w_q, w_k, w_v, w_o,
        q_norm, k_norm,
        cos_cache, sin_cache,
        cfg::ModelConfig;
        kv_cache=nothing,
        layer::Int=1,
        position_ids=nothing,
    ) -> output

Single attention layer.
  x:        (hidden_size, seq_len)                  — input (no batch dim)
  w_q:      (num_heads    * head_dim, hidden_size)
  w_k:      (num_kv_heads * head_dim, hidden_size)
  w_v:      (num_kv_heads * head_dim, hidden_size)
  w_o:      (hidden_size, num_heads * head_dim)
  q_norm:   (head_dim,)                             — RMSNorm scale on Q (Qwen3 QK-Norm)
  k_norm:   (head_dim,)                             — RMSNorm scale on K (Qwen3 QK-Norm)
  output:   (hidden_size, seq_len)

Algorithm (to be implemented):
  1. Project Q, K, V from x.
  2. Reshape Q to (head_dim, seq_len, num_heads),  K/V to (head_dim, seq_len, num_kv_heads).
  3. RMSNorm Q with q_norm and K with k_norm along the head_dim axis.
  4. Apply RoPE to Q and K (positions from position_ids).
  5. If kv_cache !== nothing: append K, V to the cache and read back full K, V.
  6. Expand K, V from num_kv_heads → num_heads (GQA: repeat each head num_heads/num_kv_heads times).
  7. Scaled dot-product attention with causal mask.
  8. Reshape back to (hidden_size, seq_len) and project with w_o.
"""
function attention_forward(
    x, w_q, w_k, w_v, w_o,
    q_norm, k_norm,
    cos_cache, sin_cache,
    cfg::ModelConfig;
    kv_cache=nothing,
    layer::Int=1,
    position_ids=nothing,
)
    error("attention_forward: not yet implemented")
end
