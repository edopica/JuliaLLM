module JuliaLLM

using JSON3
using SafeTensors
using LinearAlgebra
using Statistics
using NNlib
using Adapt

include("config.jl")
include("weights.jl")
include("tokenizer.jl")
include("math.jl")
include("norm.jl")
include("rope.jl")
include("attention.jl")
include("mlp.jl")
include("cache.jl")
include("block.jl")
include("model.jl")
include("generate.jl")

export ModelConfig, load_config
export load_weights, load_checkpoint, CheckpointBundle, list_tensors, get_tensor
export load_tokenizer, encode, decode, format_chat_prompt
export rms_norm
export build_rope_cache, apply_rope
export matmul, safe_batched_mul
export attention_forward
export mlp_forward
export KVCache, update_cache!, advance_cache!
export LayerWeights, block_forward
export QwenModel, load_model, forward
export greedy_generate

end # module JuliaLLM
