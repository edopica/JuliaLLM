"""
BPE Tokenizer implementation for Qwen3.
Supports ByteLevel BPE as represented in HuggingFace tokenizer.json.
"""

struct Tokenizer
    vocab::Dict{Int,String}
    token_to_id::Dict{String,Int}
    merges::Dict{Tuple{String,String},Int}
    byte_encoder::Dict{UInt8,Char}
    byte_decoder::Dict{Char,UInt8}
    bos_token_id::Int
    eos_token_id::Int
end

"""
    get_byte_encoder() -> Dict{UInt8, Char}

Returns the standard GPT-2 byte-to-unicode mapping.
"""
function get_byte_encoder()
    bs = vcat(collect(Int('!'):Int('~')), collect(Int('¡'):Int('¬')), collect(Int('®'):Int('ÿ')))
    cs = copy(bs)
    n = 0
    for b in 0:255
        if !(b in bs)
            push!(bs, b)
            push!(cs, 256 + n)
            n += 1
        end
    end
    return Dict{UInt8, Char}(UInt8(b) => Char(c) for (b, c) in zip(bs, cs))
end

"""
    load_tokenizer(path::AbstractString) -> Tokenizer

Parse tokenizer.json and build vocab lookup tables and merge rules.
"""
function load_tokenizer(path::AbstractString)::Tokenizer
    raw = JSON3.read(read(path, String))

    vocab_raw = raw[:model][:vocab]
    token_to_id = Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(vocab_raw))
    vocab = Dict{Int,String}(v => k for (k, v) in token_to_id)

    # Load merges: either ["A", "B"] or "A B"
    merges = Dict{Tuple{String,String},Int}()
    if haskey(raw[:model], :merges)
        for (i, m) in enumerate(raw[:model][:merges])
            if m isa JSON3.Array
                if length(m) == 2
                    merges[(string(m[1]), string(m[2]))] = i
                end
            else
                parts = split(string(m), ' ')
                if length(parts) == 2
                    merges[(parts[1], parts[2])] = i
                end
            end
        end
    end

    # Byte mapping
    be = get_byte_encoder()
    bd = Dict{Char,UInt8}(v => k for (k, v) in be)

    # Read special tokens
    bos_id = 151643 # Default for Qwen2.5/3
    eos_id = 151643
    if haskey(raw, :added_tokens)
        for tok in raw[:added_tokens]
            content = String(tok[:content])
            id = Int(tok[:id])
            if content == "<|endoftext|>" || content == "</s>"
                eos_id = id
                bos_id = id # Qwen often uses same for both
            end
            if content == "<s>"
                bos_id = id
            end
        end
    end

    return Tokenizer(vocab, token_to_id, merges, be, bd, bos_id, eos_id)
end

"""
    encode(tk::Tokenizer, text::String) -> Vector{Int}

Tokenize a string into token IDs.
"""
function encode(tk::Tokenizer, text::String)::Vector{Int}
    # 1. Byte-level pre-encoding
    # Convert each byte of UTF-8 to its Unicode character counterpart
    bytes = collect(Vector{UInt8}(text))
    words = [string(tk.byte_encoder[b]) for b in bytes]
    
    # 2. Iterative BPE merging
    # In a real implementation, we'd use a regex split first, but here we
    # do a simple greedy merge on the whole sequence.
    while true
        # Find all possible pairs
        pairs = Set{Tuple{String,String}}()
        for i in 1:length(words)-1
            push!(pairs, (words[i], words[i+1]))
        end
        
        if isempty(pairs) break end
        
        # Find the pair with the highest priority (lowest index in merges)
        best_pair = nothing
        min_rank = typemax(Int)
        
        for p in pairs
            rank = get(tk.merges, p, typemax(Int))
            if rank < min_rank
                min_rank = rank
                best_pair = p
            end
        end
        
        if best_pair === nothing break end
        
        # Merge the best pair
        new_words = String[]
        i = 1
        while i <= length(words)
            if i < length(words) && (words[i], words[i+1]) == best_pair
                push!(new_words, words[i] * words[i+1])
                i += 2
            else
                push!(new_words, words[i])
                i += 1
            end
        end
        words = new_words
    end
    
    # 3. Map to IDs
    return [get(tk.token_to_id, w, 0) for w in words]
end

"""
    decode(tk::Tokenizer, ids::AbstractVector{Int}) -> String

Convert token IDs back to a string.
"""
function decode(tk::Tokenizer, ids::AbstractVector{Int})::String
    # 1. Map IDs to BPE-encoded strings
    bpe_strings = [get(tk.vocab, id, "") for id in ids]
    
    # 2. Join and map back to bytes
    full_bpe_str = join(bpe_strings)
    bytes = UInt8[]
    for c in full_bpe_str
        push!(bytes, get(tk.byte_decoder, c, UInt8('?')))
    end
    
    return String(bytes)
end

vocab_size(t::Tokenizer) = length(t.vocab)
