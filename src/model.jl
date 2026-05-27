"""
Full Qwen3 model: embedding → N blocks → final norm → LM head.

Designed to be size-agnostic across the Qwen3 dense family. Every variant uses
the same architecture; differences (depth, width, GQA ratio, head_dim,
tie_word_embeddings, sharded vs single-file checkpoint) are absorbed by
ModelConfig and the weight-loading helpers.
"""

struct QwenModel
    cfg::ModelConfig
    embed::Matrix{Float32}            # (hidden_size, vocab_size) — columns are token embeddings
    layers::Vector{LayerWeights}
    final_norm::Vector{Float32}       # (hidden_size,)
    lm_head::Matrix{Float32}          # (vocab_size, hidden_size); aliases `embed` when cfg.tie_word_embeddings
end

"""
    load_model(model_dir::AbstractString) -> QwenModel

Convenience loader. Reads `config.json`, opens the checkpoint (single-file or
sharded via `load_checkpoint`), then maps HuggingFace tensor names onto
`QwenModel` fields.

Tensor-name mapping (per layer i, 0-indexed in HF, 1-indexed in Julia):

    model.embed_tokens.weight                          → embed
    model.layers.{i}.input_layernorm.weight            → layers[i+1].norm_attn
    model.layers.{i}.self_attn.q_proj.weight           → layers[i+1].w_q
    model.layers.{i}.self_attn.k_proj.weight           → layers[i+1].w_k
    model.layers.{i}.self_attn.v_proj.weight           → layers[i+1].w_v
    model.layers.{i}.self_attn.o_proj.weight           → layers[i+1].w_o
    model.layers.{i}.self_attn.q_norm.weight           → layers[i+1].q_norm     (Qwen3 QK-Norm)
    model.layers.{i}.self_attn.k_norm.weight           → layers[i+1].k_norm     (Qwen3 QK-Norm)
    model.layers.{i}.post_attention_layernorm.weight   → layers[i+1].norm_mlp
    model.layers.{i}.mlp.gate_proj.weight              → layers[i+1].w_gate
    model.layers.{i}.mlp.up_proj.weight                → layers[i+1].w_up
    model.layers.{i}.mlp.down_proj.weight              → layers[i+1].w_down
    model.norm.weight                                  → final_norm
    lm_head.weight                                     → lm_head
        (absent when cfg.tie_word_embeddings == true; alias `embed` in that case)

Checkpoints are typically stored as BF16/FP16. Cast to Float32 at load time
unless/until a typed implementation is introduced.

TODO: implement.
"""
function load_model(model_dir::AbstractString)::QwenModel
    error("load_model: not yet implemented")
end

"""
    forward(model::QwenModel, token_ids::Vector{Int};
            kv_cache=nothing) -> Matrix{Float32}

Run the full forward pass.
  token_ids: (seq_len,) — 0-indexed token ids from the tokenizer
  Returns logits: (vocab_size, seq_len)
"""
function forward(model::QwenModel, token_ids::Vector{Int}; kv_cache=nothing)
    error("forward: not yet implemented")
end
