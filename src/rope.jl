"""
Rotary Position Embedding (RoPE).
Applied to query and key tensors before attention.

Reference: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

"""
    build_rope_cache(seq_len::Int, head_dim::Int, theta::Real; T=Float32) -> (cos, sin)

Precompute cos/sin rotation tables.
  Returns two matrices of shape (seq_len, head_dim) for the split-half layout.
"""
function build_rope_cache(seq_len::Int, head_dim::Int, theta::Real; T=Float32)
    # Frequencies: shape (head_dim÷2,)
    freqs = T(1.0) ./ (T(theta) .^ (T.(range(0, head_dim - 2; step=2)) ./ T(head_dim)))
    # Positions: shape (seq_len,)
    t = T.(0:seq_len-1)
    # Outer product: (seq_len, head_dim÷2)
    angles = t * freqs'
    
    # Concatenate to match head_dim for split-half layout
    cos_cache = hcat(cos.(angles), cos.(angles))   # (seq_len, head_dim)
    sin_cache = hcat(sin.(angles), sin.(angles))   # (seq_len, head_dim)
    
    return cos_cache, sin_cache
end

"""
    apply_rope(x::AbstractArray, cos_cache, sin_cache, position_ids) -> Array

Apply rotary embeddings to x using split-half layout.
  x: (head_dim, seq_len, n_heads, batch)  — column-major Julia layout
  Returns same shape.
"""
function apply_rope(x::AbstractArray, cos_cache, sin_cache, position_ids::AbstractVector{Int})
    head_dim, seq_len, n_heads, batch = size(x)
    half = head_dim ÷ 2
    
    # cos_cache/sin_cache: (max_seq_len, head_dim)
    # Selected: (length(position_ids), head_dim)
    # Transposed: (head_dim, seq_len) to match first two dimensions of x slices
    C = cos_cache[position_ids .+ 1, :]'
    S = sin_cache[position_ids .+ 1, :]'
    
    x_1 = @view x[1:half, :, :, :]
    x_2 = @view x[half+1:end, :, :, :]
    
    out = similar(x)
    out[1:half, :, :, :] .= x_1 .* (@view C[1:half, :]) .- x_2 .* (@view S[1:half, :])
    out[half+1:end, :, :, :] .= x_2 .* (@view C[half+1:end, :]) .+ x_1 .* (@view S[half+1:end, :])
    
    return out
end
