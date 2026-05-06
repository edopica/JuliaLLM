"""
Greedy token generation with KV cache.
"""

"""
    greedy_generate(model::QwenModel, prompt_ids::Vector{Int};
                    max_new_tokens::Int=50,
                    max_seq_len::Int=512) -> Vector{Int}

Run greedy decoding and return the list of generated token ids
(not including the prompt).
"""
function greedy_generate(model::QwenModel, prompt_ids::Vector{Int};
                         max_new_tokens::Int=50,
                         max_seq_len::Int=512)
    cache = KVCache(model.cfg, max_seq_len)
    generated = Int[]

    # Prefill: process the full prompt in one forward pass
    logits = forward(model, prompt_ids; kv_cache=cache)
    next_id = argmax(logits[:, end]) - 1   # 0-indexed

    push!(generated, next_id)

    # Decode one token at a time
    for _ in 2:max_new_tokens
        if next_id == model.cfg.vocab_size   # placeholder EOS check
            break
        end
        logits = forward(model, [next_id]; kv_cache=cache)
        next_id = argmax(logits[:, end]) - 1
        push!(generated, next_id)
    end

    return generated
end
