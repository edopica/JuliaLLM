"""
Thin wrapper around tokenizer.json for BPE encode/decode.
We reuse the HuggingFace tokenizer.json rather than reimplementing BPE.

TODO: This is a stub. Encoding/decoding can be done via a Python bridge
(subprocess call to `tokenize_reference.py`) or by adding a Julia BPE package.
For now, expose the raw vocab so tests can verify loading.
"""

struct Tokenizer
    # token_id => string
    vocab::Dict{Int,String}
    # string => token_id
    token_to_id::Dict{String,Int}
    bos_token_id::Int
    eos_token_id::Int
end

"""
    load_tokenizer(path::AbstractString) -> Tokenizer

Parse tokenizer.json and build vocab lookup tables.
"""
function load_tokenizer(path::AbstractString)::Tokenizer
    raw = JSON3.read(read(path, String))

    vocab_raw = raw[:model][:vocab]
    token_to_id = Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(vocab_raw))
    vocab = Dict{Int,String}(v => k for (k, v) in token_to_id)

    # Read special tokens if present
    bos_id = 0
    eos_id = 0
    if haskey(raw, :added_tokens)
        for tok in raw[:added_tokens]
            content = String(tok[:content])
            id = Int(tok[:id])
            if content == "<|endoftext|>" || content == "</s>"
                eos_id = id
            end
            if content == "<s>"
                bos_id = id
            end
        end
    end

    return Tokenizer(vocab, token_to_id, bos_id, eos_id)
end

"""
    vocab_size(t::Tokenizer) -> Int
"""
vocab_size(t::Tokenizer) = length(t.vocab)
