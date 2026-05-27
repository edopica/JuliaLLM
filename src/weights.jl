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

TODO: implement. Stub raises so the model loader fails loudly until done.
"""
function load_checkpoint(model_dir::AbstractString)::CheckpointBundle
    error("load_checkpoint: not yet implemented")
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
TODO: implement when `load_checkpoint` lands.
"""
function list_tensors(b::CheckpointBundle)::Vector{Pair{String,Any}}
    error("list_tensors(::CheckpointBundle): not yet implemented")
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
TODO: implement when `load_checkpoint` lands.
"""
function get_tensor(b::CheckpointBundle, name::AbstractString)
    error("get_tensor(::CheckpointBundle, ...): not yet implemented")
end
