"""Real Phi-3-mini-4k-instruct linear-layer shapes (Doc/implementation_plan.md Phase 1
step 1) -- read directly from the model already downloaded for the sibling
`Evol_inference` project (`model.safetensors.index.json` + the shard's tensor shapes),
not guessed from the architecture doc. No bias on any of these four projections
(confirmed: only `.weight` entries exist for them in the safetensors index).

Layer name  -> (out_features, in_features), matching nn.Linear's weight convention.
"""

PHI3_LAYER_SHAPES = {
    "qkv_proj": (9216, 3072),
    "o_proj": (3072, 3072),
    "gate_up_proj": (16384, 3072),
    "down_proj": (3072, 8192),
}

# Every in_features above is divisible by 128 (24, 24, 24, and 64 groups respectively) --
# confirmed before picking the quantization group size, not assumed.
