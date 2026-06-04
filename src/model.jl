"""
Full Qwen3 model: embedding → N blocks → final norm → LM head.

Designed to be size-agnostic across the Qwen3 dense family. Every variant uses
the same architecture; differences (depth, width, GQA ratio, head_dim,
tie_word_embeddings, sharded vs single-file checkpoint) are absorbed by
ModelConfig and the weight-loading helpers.
"""

struct QwenModel{M<:AbstractMatrix, V<:AbstractVector}
    cfg::ModelConfig
    embed::M                                # (hidden_size, vocab_size) — columns are token embeddings
    layers::Vector{LayerWeights{M,V}}
    final_norm::V                           # (hidden_size,)
    lm_head::M                              # (vocab_size, hidden_size); aliases `embed` when cfg.tie_word_embeddings
end

# `adapt(CuArray, model)` (or `cu(model)` from CUDA.jl) returns a GPU-resident copy.
# `cfg` is plain ints and is left untouched.
Adapt.adapt_structure(to, m::QwenModel) = QwenModel(
    m.cfg,
    adapt(to, m.embed),
    map(layer -> adapt(to, layer), m.layers),
    adapt(to, m.final_norm),
    adapt(to, m.lm_head),
)

"""
    load_model(model_dir::AbstractString; dtype=Float32) -> QwenModel

Convenience loader. Reads `config.json`, opens the checkpoint (single-file or
sharded via `load_checkpoint`), then maps HuggingFace tensor names onto
`QwenModel` fields.

Checkpoints are typically stored as BF16/FP16. Weights are converted to `dtype`
during loading.
"""
function load_model(model_dir::AbstractString; dtype=Float32)::QwenModel
    cfg = load_config(joinpath(model_dir, "config.json"))
    bundle = load_checkpoint(model_dir)

    # Convert weights to the requested dtype during load
    T = dtype
    get_T(name) = T.(get_tensor(bundle, name))

    # HF embed_tokens.weight: (vocab_size, hidden_size)
    # Julia QwenModel.embed: (hidden_size, vocab_size)
    # Use collect() to turn Adjoint into a contiguous Matrix{T}
    embed = collect(get_T("model.embed_tokens.weight")')

    M = typeof(embed)
    final_norm = get_T("model.norm.weight")
    V = typeof(final_norm)

    layers = LayerWeights{M, V}[]
    for i in 0:cfg.num_hidden_layers-1
        prefix = "model.layers.$i."
        
        # Load attention weights
        w_q = get_T(prefix * "self_attn.q_proj.weight")
        w_k = get_T(prefix * "self_attn.k_proj.weight")
        w_v = get_T(prefix * "self_attn.v_proj.weight")
        w_o = get_T(prefix * "self_attn.o_proj.weight")
        
        # Load MLP weights
        w_gate = get_T(prefix * "mlp.gate_proj.weight")
        w_up   = get_T(prefix * "mlp.up_proj.weight")
        w_down = get_T(prefix * "mlp.down_proj.weight")
        
        # Load norms
        norm_attn = get_T(prefix * "input_layernorm.weight")
        norm_mlp  = get_T(prefix * "post_attention_layernorm.weight")
        q_norm    = get_T(prefix * "self_attn.q_norm.weight")
        k_norm    = get_T(prefix * "self_attn.k_norm.weight")
        
        push!(layers, LayerWeights{M, V}(
            w_q, w_k, w_v, w_o,
            norm_attn,
            q_norm, k_norm,
            w_gate, w_up, w_down,
            norm_mlp
        ))
    end

    # lm_head: (vocab_size, hidden_size)
    if cfg.tie_word_embeddings
        # HF uses x @ embed.T. In Julia, if embed is (H, V), then x' * embed is (1, V)
        # So lm_head should be embed' if we do lm_head * x
        # But lm_head must be of type M (Matrix{T})
        lm_head = collect(embed')
    else
        lm_head = get_T("lm_head.weight")
    end

    return QwenModel{M, V}(cfg, embed, layers, final_norm, lm_head)
end

"""
    forward(model::QwenModel, token_ids::AbstractVector{Int};
            kv_cache=nothing) -> AbstractMatrix

Run the full forward pass.
  token_ids: (seq_len,) — 0-indexed token ids from the tokenizer
  Returns logits: (vocab_size, seq_len)
"""
function forward(model::QwenModel, token_ids::AbstractVector{Int}; kv_cache=nothing)
    cfg = model.cfg
    seq_len = length(token_ids)
    
    # 1. Embedding lookup
    # embed is (hidden_size, vocab_size); token_ids are 0-indexed
    x = model.embed[:, token_ids .+ 1] # (hidden_size, seq_len)

    # 2. RoPE cache & Position IDs
    # Start position for RoPE depends on whether we are using KV cache
    start_pos = (kv_cache === nothing) ? 0 : kv_cache.seq_len
    cos_cache, sin_cache = build_rope_cache(start_pos + seq_len, cfg.head_dim, cfg.rope_theta; T=eltype(x))
    
    # Move RoPE cache to the same device as x
    cos_cache = adapt(typeof(x), cos_cache)
    sin_cache = adapt(typeof(x), sin_cache)
    
    position_ids = adapt(similar(x, Int, 0), collect(start_pos : start_pos + seq_len - 1))

    # 3. Transformer Layers
    for (i, layer) in enumerate(model.layers)
        x = block_forward(x, layer, cfg, cos_cache, sin_cache; 
                          kv_cache=kv_cache, layer=i, position_ids=position_ids)
    end

    # Advance KV cache sequence length if it exists
    if kv_cache !== nothing
        advance_cache!(kv_cache, seq_len)
    end

    # 4. Final normalization
    x = rms_norm(x, model.final_norm, cfg.rms_norm_eps)

    # 5. LM Head projection
    # lm_head: (vocab_size, hidden_size), x: (hidden_size, seq_len)
    logits = model.lm_head * x # (vocab_size, seq_len)
    
    return logits
end
