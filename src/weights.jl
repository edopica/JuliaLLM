"""
Load weights from a safetensors checkpoint.
Provides helpers to list tensor names/shapes and fetch individual tensors.
"""

"""
    load_weights(path::AbstractString) -> SafeTensors.SafeTensor

Open a .safetensors file and return the handle.
Use list_tensors() and get_tensor() to inspect and retrieve.
"""
function load_weights(path::AbstractString)
    return SafeTensors.load_safetensors(path)
    #return SafeTensors.deserialize(path)
end

"""
    list_tensors(st) -> Vector{Pair{String, Tuple}}

Return a sorted list of (name => shape) pairs from the safetensors handle.
"""
function list_tensors(st)::Vector{Pair{String,Any}}
    
    weights = sort(collect(st), by = first)
    weights = map(p -> p[1] => size(p[2]), weights)
    return weights
end

"""
    get_tensor(st, name::AbstractString) -> Array

Retrieve and materialize a single tensor by name.
"""
function get_tensor(st, name::AbstractString)
    return Array(st[name])
end
