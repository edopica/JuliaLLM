"""
RMSNorm: Root Mean Square Layer Normalization.
Used in Qwen3 in place of LayerNorm.

Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
"""

"""
    rms_norm(x::AbstractArray, weight::AbstractVector, eps::Float64) -> Array

Apply RMSNorm along the first dimension of x.
  x:      (hidden_size, ...)
  weight: (hidden_size,)  — learnable scale (gamma)
  output: same shape as x
"""
function rms_norm(x::AbstractArray{T}, weight::AbstractVector{T}, eps::Real=1e-5) where T
    # Compute entirely in Float32 to avoid overflow of mean(x^2)
    x32 = Float32.(x)
    ms = mean(x32 .^ 2, dims=1)
    x_norm = x32 ./ sqrt.(ms .+ Float32(eps))
    return T.(x_norm) .* weight
end
