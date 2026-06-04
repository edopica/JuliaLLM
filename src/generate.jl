"""
Greedy token generation with KV cache.
"""

"""
    greedy_generate(model::QwenModel, prompt_ids::AbstractVector{Int};
                    max_new_tokens::Int=50,
                    max_seq_len::Int=512,
                    eos_ids::Union{Nothing, Set{Int}, AbstractVector{Int}, Int}=nothing) -> Vector{Int}

Run greedy decoding and return the list of generated token ids
(not including the prompt).
"""
function greedy_generate(model::QwenModel, prompt_ids::AbstractVector{Int};
                         max_new_tokens::Int=50,
                         max_seq_len::Int=512,
                         eos_ids::Union{Nothing, Set{Int}, AbstractVector{Int}, Int}=nothing)
    # Convert eos_ids to a Set for O(1) lookup
    eos_set = if eos_ids === nothing
        Set{Int}()
    elseif eos_ids isa Int
        Set{Int}([eos_ids])
    else
        Set{Int}(eos_ids)
    end

    # Use `like=model.embed` to ensure cache is on the same device as the model
    cache = KVCache(model.cfg, max_seq_len; like=model.embed)
    generated = Int[]

    # Prefill: process the full prompt in one forward pass
    logits = forward(model, prompt_ids; kv_cache=cache)
    
    # Get the last token's logits
    # argmax(logits[:, end]) returns a CartesianIndex or Int depending on dims
    # We want the index of the max value in the last column
    next_id = Int(argmax(view(logits, :, size(logits, 2)))) - 1   # 0-indexed

    if next_id in eos_set
        return generated
    end
    push!(generated, next_id)

    # Decode one token at a time
    for _ in 2:max_new_tokens
        if length(prompt_ids) + length(generated) >= max_seq_len
            break
        end

        logits = forward(model, [next_id]; kv_cache=cache)
        next_id = Int(argmax(view(logits, :, 1))) - 1
        
        if next_id in eos_set
            break
        end
        
        push!(generated, next_id)
    end

    return generated
end
