"""
Load weights from a safetensors checkpoint.

Two checkpoint layouts must be supported:

  - Single-file: `model.safetensors`              (e.g. Qwen3-0.6B, 1.7B)
  - Sharded:     `model-00001-of-0000N.safetensors`, ...
                 `model.safetensors.index.json`    (e.g. Qwen3-4B and larger)

`load_weights` is the low-level single-file loader. `load_checkpoint` is the
high-level entry point that picks the right path based on what's in the model
directory and returns a `CheckpointBundle` that supports the same
`list_tensors` / `get_tensor` interface.
"""

"""
    CheckpointBundle

Unified handle over one or more safetensors files. Hides whether the checkpoint
is single-file or sharded so the model loader does not have to care.

Fields:
  - `shards::Vector` — one SafeTensor handle per `.safetensors` file
  - `index::Dict{String,Int}` — tensor name → shard index (1-based into `shards`)
"""
struct CheckpointBundle
    shards::Vector{Any}
    index::Dict{String,Int}
end

"""
    load_weights(path::AbstractString) -> SafeTensors.SafeTensor

Open a single `.safetensors` file and return the handle.
Use `list_tensors` and `get_tensor` to inspect and retrieve.
"""
function load_weights(path::AbstractString)
    return SafeTensors.load_safetensors(path)
end

"""
    load_checkpoint(model_dir::AbstractString) -> CheckpointBundle

Inspect `model_dir` and return a `CheckpointBundle` covering every tensor in
the checkpoint, regardless of whether the checkpoint is single-file or sharded.

Behavior:
  1. If `model.safetensors.index.json` exists, parse it. The `weight_map` field
     maps tensor names → shard filenames. Open each unique shard once, build
     `index` mapping tensor name → position in `shards`.
  2. Otherwise, look for exactly one `*.safetensors` file in `model_dir`,
     open it, and build an index from its tensor names.

"""
function load_checkpoint(model_dir::AbstractString)::CheckpointBundle
    index_name = "model.safetensors.index.json"
    index_path = joinpath(model_dir, index_name)

    if isfile(index_path)
        # Sharded case
        index_data = JSON3.read(read(index_path, String))
        weight_map = index_data.weight_map
        
        # Load unique shards
        shard_files = unique(values(weight_map))
        shards = Any[]
        shard_to_idx = Dict{String, Int}()
        for (i, f) in enumerate(shard_files)
            fname = String(f)
            push!(shards, load_weights(joinpath(model_dir, fname)))
            shard_to_idx[fname] = i
        end
        
        # Map tensor names to shard indices
        tensor_index = Dict{String, Int}()
        for (tensor_name, shard_file) in weight_map
            tensor_index[String(tensor_name)] = shard_to_idx[String(shard_file)]
        end
        
        return CheckpointBundle(shards, tensor_index)
    else
        # Single-file case
        st_files = filter(f -> endswith(f, ".safetensors"), readdir(model_dir))
        
        if isempty(st_files) error("No .safetensors files found in $model_dir") end
        
        target_file = ""
        if length(st_files) == 1
            target_file = st_files[1]
        elseif "model.safetensors" in st_files
            target_file = "model.safetensors"
        else
            error("Multiple .safetensors files found in $model_dir, but no index file. Ambiguous checkpoint.")
        end
        
        st = load_weights(joinpath(model_dir, target_file))
        
        # Build index: all tensors in this file map to shard 1
        tensor_index = Dict{String, Int}()
        for (name, _) in st
            tensor_index[String(name)] = 1
        end
        
        return CheckpointBundle([st], tensor_index)
    end
end

"""
    list_tensors(st) -> Vector{Pair{String, Tuple}}

Return a sorted list of (name => shape) pairs from a single safetensors handle.
"""
function list_tensors(st)::Vector{Pair{String,Any}}
    weights = sort(collect(st), by = first)
    weights = map(p -> p[1] => size(p[2]), weights)
    return weights
end

"""
    list_tensors(b::CheckpointBundle) -> Vector{Pair{String, Tuple}}

Same as the single-handle version but aggregated across every shard.
"""
function list_tensors(b::CheckpointBundle)::Vector{Pair{String,Any}}
    tensors = Pair{String, Any}[]
    for (name, shard_idx) in b.index
        st = b.shards[shard_idx]
        push!(tensors, name => size(st[name]))
    end
    sort!(tensors, by=first)
    return tensors
end

"""
    get_tensor(st, name::AbstractString) -> Array

Retrieve and materialize a single tensor by name from a single safetensors
handle.
"""
function get_tensor(st, name::AbstractString)
    return Array(st[name])
end

"""
    get_tensor(b::CheckpointBundle, name::AbstractString) -> Array

Resolve `name` to its shard via `b.index` and materialize the tensor.
"""
function get_tensor(b::CheckpointBundle, name::AbstractString)
    if !haskey(b.index, name)
        error("Tensor '$name' not found in checkpoint.")
    end
    shard_idx = b.index[name]
    st = b.shards[shard_idx]
    return get_tensor(st, name)
end
