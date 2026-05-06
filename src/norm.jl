"""
RMSNorm: Root Mean Square Layer Normalization.
Used in Qwen3 in place of LayerNorm.

Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
"""

"""
    rms_norm(x::AbstractArray, weight::AbstractVector, eps::Float64) -> Array

Apply RMSNorm along the last dimension of x.
  x:      (..., hidden_size)
  weight: (hidden_size,)  — learnable scale (gamma)
  output: same shape as x
"""
function rms_norm(x::AbstractArray, weight::AbstractVector, eps::Float64)
    # RMS over the last dimension
    ms = mean(x .^ 2, dims=ndims(x))          # (..., 1)
    x_norm = x ./ sqrt.(ms .+ eps)            # (..., hidden_size)
    return x_norm .* weight                   # broadcast scale
end
