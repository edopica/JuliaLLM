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
    hidden_size, seq_len = size(x)

    # 1. Project Q, K, V from x
    q_raw = w_q * x  # (num_heads * head_dim, seq_len)
    k_raw = w_k * x  # (num_kv_heads * head_dim, seq_len)
    v_raw = w_v * x  # (num_kv_heads * head_dim, seq_len)

    # 2. Reshape to (head_dim, n_heads, seq_len)
    q = reshape(q_raw, cfg.head_dim, cfg.num_attention_heads, seq_len)
    k = reshape(k_raw, cfg.head_dim, cfg.num_key_value_heads, seq_len)
    v = reshape(v_raw, cfg.head_dim, cfg.num_key_value_heads, seq_len)

    # 3. RMSNorm Q with q_norm and K with k_norm (QK-Norm)
    # RMS along the head_dim (first dimension)
    T = eltype(x)
    function qk_norm(y, weight)
        ms = mean(y .^ 2, dims=1)
        return (y ./ sqrt.(ms .+ T(cfg.rms_norm_eps))) .* weight
    end
    q = qk_norm(q, q_norm)
    k = qk_norm(k, k_norm)

    # 4. Apply RoPE to Q and K
    # apply_rope expects (head_dim, seq_len, n_heads, batch)
    q = permutedims(q, (1, 3, 2)) # (head_dim, seq_len, num_heads)
    k = permutedims(k, (1, 3, 2)) # (head_dim, seq_len, num_kv_heads)
    v = permutedims(v, (1, 3, 2)) # (head_dim, seq_len, num_kv_heads)

    if position_ids === nothing
        offset = (kv_cache === nothing) ? 0 : kv_cache.seq_len
        # Use similar(x, Int, 0) to get the correct array type (device) with Int eltype
        position_ids = adapt(similar(x, Int, 0), collect(offset:(offset + seq_len - 1)))
    end

    # Add batch dimension for apply_rope
    q_rope = reshape(q, cfg.head_dim, seq_len, cfg.num_attention_heads, 1)
    k_rope = reshape(k, cfg.head_dim, seq_len, cfg.num_key_value_heads, 1)

    q = dropdims(apply_rope(q_rope, cos_cache, sin_cache, position_ids), dims=4)
    k = dropdims(apply_rope(k_rope, cos_cache, sin_cache, position_ids), dims=4)

    # 5. KV Cache: append K, V and read back full K, V
    if kv_cache !== nothing
        # total_len is the length after adding current tokens
        # update_cache! only increments kv_cache.seq_len after the last layer,
        # so we calculate it here based on current seq_len.
        total_len = kv_cache.seq_len + seq_len
        
        # update_cache! expects (head_dim, num_kv_heads, seq_len)
        update_cache!(kv_cache, layer, permutedims(k, (1, 3, 2)), permutedims(v, (1, 3, 2)))
        
        k = kv_cache.keys[layer][:, :, 1:total_len]
        v = kv_cache.values[layer][:, :, 1:total_len]
        
        # Reshape back to (head_dim, total_len, num_kv_heads)
        k = permutedims(k, (1, 3, 2))
        v = permutedims(v, (1, 3, 2))
    else
        total_len = seq_len
    end

    # 6. Expand K, V from num_kv_heads -> num_heads (GQA)
    if cfg.num_key_value_heads < cfg.num_attention_heads
        n_rep = cfg.num_attention_heads ÷ cfg.num_key_value_heads
        k = repeat(k, inner=(1, 1, n_rep))
        v = repeat(v, inner=(1, 1, n_rep))
    end

    # 7. Scaled dot-product attention with causal mask
    # Q: (head_dim, seq_len, num_heads) -> (seq_len, head_dim, num_heads)
    # K: (head_dim, total_len, num_heads)
    q_attn = permutedims(q, (2, 1, 3))
    scores = batched_mul(q_attn, k) ./ T(sqrt(cfg.head_dim))

    if seq_len > 1
        # Causal mask: mask[i, j] = true if j > i + (total_len - seq_len)
        mask = (1:seq_len) .+ (total_len - seq_len) .< (1:total_len)'
        # Move the mask to the GPU as a CuArray (or whichever device scores is on)
        mask_device = adapt(typeof(scores).name.wrapper, collect(mask))
        scores = ifelse.(mask_device, T(-1e9), scores)
    end

    attn_weights = softmax(scores, dims=2)

    # V: (head_dim, total_len, num_heads) -> (total_len, head_dim, num_heads)
    v_attn = permutedims(v, (2, 1, 3))
    attn_out = batched_mul(attn_weights, v_attn) # (seq_len, head_dim, num_heads)

    # 8. Reshape back and project with w_o
    attn_out = permutedims(attn_out, (2, 3, 1)) # (head_dim, num_heads, seq_len)
    attn_out = reshape(attn_out, cfg.num_attention_heads * cfg.head_dim, seq_len)
    
    return w_o * attn_out
end
