"""
Greedy token generation with KV cache.
"""

"""
    greedy_generate(model::QwenModel, prompt_ids::AbstractVector{Int};
                    max_new_tokens::Int=50,
                    max_seq_len::Int=512) -> Vector{Int}

Run greedy decoding and return the list of generated token ids
(not including the prompt).
"""
function greedy_generate(model::QwenModel, prompt_ids::AbstractVector{Int};
                         max_new_tokens::Int=50,
                         max_seq_len::Int=512)
    # Use `like=model.embed` to ensure cache is on the same device as the model
    cache = KVCache(model.cfg, max_seq_len; like=model.embed)
    generated = Int[]

    # Prefill: process the full prompt in one forward pass
    logits = forward(model, prompt_ids; kv_cache=cache)
    # argmax(logits[:, end]) returns a CartesianIndex or Int depending on dims
    # We want the index of the max value in the last column
    next_id = Int(argmax(logits[:, end])) - 1   # 0-indexed

    push!(generated, next_id)

    # Decode one token at a time
    for _ in 2:max_new_tokens
        if next_id == model.cfg.vocab_size   # placeholder EOS check
            break
        end
        logits = forward(model, [next_id]; kv_cache=cache)
        next_id = Int(argmax(logits[:, end])) - 1
        push!(generated, next_id)
    end

    return generated
end
