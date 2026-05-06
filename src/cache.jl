"""
KV cache for incremental decoding.
Stores previously computed keys and values per layer.
"""

mutable struct KVCache
    # keys[layer]:   (head_dim, num_kv_heads, max_seq_len)
    # values[layer]: (head_dim, num_kv_heads, max_seq_len)
    keys::Vector{Array{Float32,3}}
    values::Vector{Array{Float32,3}}
    seq_len::Int   # how many positions are currently filled
end

"""
    KVCache(cfg::ModelConfig, max_seq_len::Int) -> KVCache

Allocate an empty KV cache for all layers.
"""
function KVCache(cfg::ModelConfig, max_seq_len::Int)
    keys   = [zeros(Float32, cfg.head_dim, cfg.num_key_value_heads, max_seq_len)
              for _ in 1:cfg.num_hidden_layers]
    values = [zeros(Float32, cfg.head_dim, cfg.num_key_value_heads, max_seq_len)
              for _ in 1:cfg.num_hidden_layers]
    return KVCache(keys, values, 0)
end

"""
    update_cache!(cache::KVCache, layer::Int, new_k, new_v)

Append new key/value slices to the cache for a given layer.
  new_k, new_v: (head_dim, num_kv_heads, new_tokens)
"""
function update_cache!(cache::KVCache, layer::Int, new_k, new_v)
    n = size(new_k, 3)
    start = cache.seq_len + 1
    stop  = cache.seq_len + n
    cache.keys[layer][:, :, start:stop]   .= new_k
    cache.values[layer][:, :, start:stop] .= new_v
    if layer == length(cache.keys)
        cache.seq_len += n   # advance position only after last layer
    end
end
