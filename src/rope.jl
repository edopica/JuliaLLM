"""
Rotary Position Embedding (RoPE).
Applied to query and key tensors before attention.

Reference: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

"""
    build_rope_cache(seq_len::Int, head_dim::Int, theta::Float64) -> (cos, sin)

Precompute cos/sin rotation tables.
  Returns two matrices of shape (seq_len, head_dim÷2) if using complex form,
  or (seq_len, head_dim) for the paired-dimension layout.
"""
function build_rope_cache(seq_len::Int, head_dim::Int, theta::Float64)
    # Frequencies: shape (head_dim÷2,)
    freqs = 1.0f0 ./ (Float32(theta) .^ (range(0, head_dim - 2; step=2) ./ head_dim))
    # Positions: shape (seq_len,)
    t = Float32.(0:seq_len-1)
    # Outer product: (seq_len, head_dim÷2)
    angles = t * freqs'
    cos_cache = cos.(angles)   # (seq_len, head_dim÷2)
    sin_cache = sin.(angles)   # (seq_len, head_dim÷2)
    return cos_cache, sin_cache
end

"""
    apply_rope(x::AbstractArray, cos_cache, sin_cache, position_ids) -> Array

Apply rotary embeddings to x.
  x: (head_dim, seq_len, n_heads, batch)  — column-major Julia layout
  Returns same shape.
"""
function apply_rope(x::AbstractArray, cos_cache, sin_cache, position_ids::AbstractVector{Int})
    head_dim, seq_len, n_heads, batch = size(x)
    
    # cos_cache/sin_cache: (max_seq_len, head_dim÷2)
    # Selected: (length(position_ids), head_dim÷2)
    # Transposed: (head_dim÷2, seq_len) to match first two dimensions of x slices
    C = cos_cache[position_ids .+ 1, :]'
    S = sin_cache[position_ids .+ 1, :]'
    
    x_1 = x[1:2:end, :, :, :]
    x_2 = x[2:2:end, :, :, :]
    
    out = similar(x)
    out[1:2:end, :, :, :] = x_1 .* C .- x_2 .* S
    out[2:2:end, :, :, :] = x_1 .* S .+ x_2 .* C
    
    return out
end
