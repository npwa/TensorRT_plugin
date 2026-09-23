"""Phase 1 steps 2-3: the quantization scheme, documented and decided here, plus the
PyTorch reference implementation every later phase (naive kernel, optimized kernel,
plugin, engine) is validated against.

## The scheme: groupwise symmetric INT4

Chosen deliberately over replicating bitsandbytes' NF4 (Doc/implementation_plan.md
Phase 1 step 2): this project's point is kernel authorship, not quantization-scheme
novelty, so a standard, well-understood scheme keeps this phase tractable.

- Weight matrix `W`: `[out_features, in_features]`, row-major, matching `nn.Linear`.
- Grouped along `in_features` (the K / reduction dimension) in groups of `GROUP_SIZE`
  (128) -- every Phi-3 linear layer's `in_features` (3072 or 8192) divides evenly into
  128-element groups (24 or 64 groups respectively; verified against the real model
  shapes in `shapes.py`, not assumed).
- Per group: `scale = max(abs(group)) / 7` (float32 intermediate, stored fp16). Signed
  4-bit range is `[-8, 7]`; dividing by 7 rather than 8 keeps the representable range
  symmetric around the group's actual max magnitude (the `-8` codepoint exists but is
  reachable only via rounding, standard practice for symmetric quantization).
- Quantized value: `q = clamp(round(w / scale), -8, 7)`, stored as a signed 4-bit
  integer, **packed two values per byte** (low nibble = even index, high nibble = odd
  index within the pair) -- this is real bit-packing, not a placeholder: storage
  density is what "INT4" actually means, so it belongs in Phase 1 even though the
  *compute* pattern here is still naive (Phase 2 is where the memory-access pattern
  gets optimized, not the storage format).
- Dequant: `w_approx = q * scale`, broadcast back over each group.
- No bias: confirmed empirically (`shapes.py`) that none of Phi-3's four projection
  types carry a bias tensor, so the reference and every kernel after it can skip it
  entirely rather than handling an unused code path.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """`q`: int8 tensor in [-8, 7], last dim even. Returns uint8, last dim halved --
    two signed 4-bit values packed per byte (low nibble first)."""
    assert q.shape[-1] % 2 == 0, "last dimension must be even to pack two nibbles per byte"
    q_u = (q & 0xF).to(torch.uint8)  # two's-complement 4-bit representation
    lo = q_u[..., 0::2]
    hi = q_u[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_int4`. Returns int8 in [-8, 7], last dim doubled."""
    lo = (packed & 0xF).to(torch.int16)
    hi = ((packed >> 4) & 0xF).to(torch.int16)
    lo_signed = torch.where(lo >= 8, lo - 16, lo).to(torch.int8)
    hi_signed = torch.where(hi >= 8, hi - 16, hi).to(torch.int8)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo_signed
    out[..., 1::2] = hi_signed
    return out


def quantize_groupwise_int4(w: torch.Tensor, group_size: int = GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """`w`: `[out_features, in_features]` fp16/fp32. Returns `(packed_uint8, scale_fp16)`
    where `packed` is `[out_features, in_features // 2]` and `scale` is
    `[out_features, in_features // group_size]`."""
    out_features, in_features = w.shape
    assert in_features % group_size == 0, f"in_features={in_features} not divisible by group_size={group_size}"
    n_groups = in_features // group_size

    w32 = w.float().reshape(out_features, n_groups, group_size)
    absmax = w32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(w32 / scale), -8, 7).to(torch.int8)

    packed = pack_int4(q.reshape(out_features, in_features))
    return packed, scale.squeeze(-1).half()


def dequantize_groupwise_int4(packed: torch.Tensor, scale: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Inverse of `quantize_groupwise_int4`. Returns `[out_features, in_features]` fp32."""
    out_features = packed.shape[0]
    q = unpack_int4(packed)  # [out_features, in_features], int8
    in_features = q.shape[-1]
    n_groups = in_features // group_size
    q = q.reshape(out_features, n_groups, group_size).float()
    w = q * scale.float().unsqueeze(-1)
    return w.reshape(out_features, in_features)


def quantized_linear_reference(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """`x`: `[..., in_features]`. Returns `[..., out_features]`. The numerical ground
    truth every kernel/plugin/engine in this project is validated against -- plain
    `torch` ops only, no custom kernels involved."""
    w = dequantize_groupwise_int4(packed, scale, group_size)
    return x.float() @ w.T
