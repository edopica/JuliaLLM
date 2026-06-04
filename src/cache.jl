"""
KV cache for incremental decoding.
Stores previously computed keys and values per layer.
"""

mutable struct KVCache{A<:AbstractArray{<:AbstractFloat,3}}
    # keys[layer]:   (head_dim, num_kv_heads, max_seq_len)
    # values[layer]: (head_dim, num_kv_heads, max_seq_len)
    keys::Vector{A}
    values::Vector{A}
    seq_len::Int   # how many positions are currently filled
end

"""
    KVCache(cfg::ModelConfig, max_seq_len::Int; like=Float32[]) -> KVCache

Allocate an empty KV cache for all layers. Pass `like=model.embed` (or any
weight from the model) to make the cache live on the same device as the model.
"""
function KVCache(cfg::ModelConfig, max_seq_len::Int; like::AbstractArray=Float32[])
    keys   = [fill!(similar(like, cfg.head_dim, cfg.num_key_value_heads, max_seq_len), 0)
              for _ in 1:cfg.num_hidden_layers]
    values = [fill!(similar(like, cfg.head_dim, cfg.num_key_value_heads, max_seq_len), 0)
              for _ in 1:cfg.num_hidden_layers]
    return KVCache(keys, values, 0)
end

Adapt.adapt_structure(to, c::KVCache) = KVCache(
    map(k -> adapt(to, k), c.keys),
    map(v -> adapt(to, v), c.values),
    c.seq_len,
)

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
end

"""
    advance_cache!(cache::KVCache, n::Int)

Advance the internal sequence length by `n` positions. Call this once per
decoding step (after all layers have updated the cache).
"""
function advance_cache!(cache::KVCache, n::Int)
    cache.seq_len += n
end
