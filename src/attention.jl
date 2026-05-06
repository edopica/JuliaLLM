"""
Multi-head / Grouped-Query Attention forward pass.
Qwen3-0.6B uses GQA (num_key_value_heads < num_attention_heads).
"""

"""
    attention_forward(
        x, w_q, w_k, w_v, w_o,
        cos_cache, sin_cache,
        cfg::ModelConfig;
        kv_cache=nothing,
        position_ids=nothing,
    ) -> (output, updated_kv_cache)

Single attention layer.
  x:     (hidden_size, seq_len)           — input (no batch dim for simplicity)
  w_q:   (num_heads * head_dim, hidden_size)
  w_k:   (num_kv_heads * head_dim, hidden_size)
  w_v:   (num_kv_heads * head_dim, hidden_size)
  w_o:   (hidden_size, num_heads * head_dim)
  output: (hidden_size, seq_len)

TODO: implement full attention math once config loading is validated.
"""
function attention_forward(
    x, w_q, w_k, w_v, w_o,
    cos_cache, sin_cache,
    cfg::ModelConfig;
    kv_cache=nothing,
    position_ids=nothing,
)
    error("attention_forward: not yet implemented")
end
