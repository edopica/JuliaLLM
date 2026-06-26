"""
Safe matrix multiplication for Float16.
Julia's generic matrix multiplication accumulates in the element type,
which causes catastrophic overflow for Float16 arrays.
"""
function matmul(A::AbstractMatrix{Float16}, B::AbstractVecOrMat{Float16})
    return Float16.(Float32.(A) * Float32.(B))
end
function matmul(A, B)
    return A * B
end

function safe_batched_mul(A::AbstractArray{Float16, 3}, B::AbstractArray{Float16, 3})
    return Float16.(batched_mul(Float32.(A), Float32.(B)))
end
function safe_batched_mul(A, B)
    return batched_mul(A, B)
end
