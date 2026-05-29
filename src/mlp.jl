"""
SwiGLU MLP block used in Qwen3.
out = (silu(x @ gate_proj.T) * (x @ up_proj.T)) @ down_proj.T
"""

"""
    mlp_forward(x, w_gate, w_up, w_down) -> Array

  x:      (hidden_size, seq_len)
  w_gate: (intermediate_size, hidden_size)
  w_up:   (intermediate_size, hidden_size)
  w_down: (hidden_size, intermediate_size)
  output: (hidden_size, seq_len)
"""
function mlp_forward(x, w_gate, w_up, w_down)
    g_raw = w_gate * x
    gate  = NNlib.sigmoid_fast.(g_raw) .* g_raw   # silu = x * sigmoid(x)
    up    = w_up * x
    return w_down * (gate .* up)
end
