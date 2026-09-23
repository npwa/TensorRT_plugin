"""Phase 5B: drop-in `nn.Linear` replacements for Phi-3's four projection types, used to
module-substitute a real Phi-3-mini forward/generate pass -- the same mechanism
`Evol_inference`'s `WeightBank` already uses to swap precision per layer, applied here to
swap *compute path* instead (plain PyTorch dequant-then-matmul vs. the Phase 2 optimized
CUDA kernel), holding the quantized weights themselves identical between the two so the
only variable being measured is which code computes the matmul.

Both classes are built from the *same* `(packed, scale)` pair (`quantize_groupwise_int4`,
Phase 1's scheme) -- this mirrors Phase 5's own benchmark table methodology (same
weights, same input, only the compute path differs).
"""

from __future__ import annotations

import gc
import os

import torch
from torch import nn
from torch.utils.cpp_extension import load

from python.reference.quant import GROUP_SIZE, quantize_groupwise_int4, quantized_linear_reference

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels")

_ext = None


def _kernel_ext():
    """JIT-compiled once, reused across all 128 layers -- compiling per-layer would pay
    the ninja build cost 128 times over for identical source files."""
    global _ext
    if _ext is None:
        _ext = load(
            name="dequant_gemm_ext_phase5b",
            sources=[
                os.path.join(_KERNEL_DIR, "torch_binding.cpp"),
                os.path.join(_KERNEL_DIR, "dequant_gemm_naive.cu"),
                os.path.join(_KERNEL_DIR, "dequant_gemm_optimized.cu"),
            ],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    return _ext


class _Int4LinearBase(nn.Module):
    """Shared setup: quantize `fp16_linear`'s weight once (Phase 1 scheme, group size
    128), keep `packed`/`scale` as buffers so `.to(device)` moves them like any other
    module state. Phi-3's four projection types are bias-free (confirmed in Phase 1 --
    `python/reference/quant.py`'s docstring), so bias handling is deliberately omitted
    rather than dead code for a case that can't occur here."""

    def __init__(self, fp16_linear: nn.Linear, group_size: int = GROUP_SIZE):
        super().__init__()
        assert fp16_linear.bias is None, "Phi-3 projections are bias-free; unexpected bias here"
        packed, scale = quantize_groupwise_int4(fp16_linear.weight.data, group_size)
        self.out_features = fp16_linear.out_features
        self.in_features = fp16_linear.in_features
        self.group_size = group_size
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scale", scale.float(), persistent=False)


class EagerInt4Linear(_Int4LinearBase):
    """Configuration (b): plain PyTorch ops (dequantize the full weight tile, then
    matmul) -- today's slow path, included so the end-to-end comparison isolates the
    same effect Phase 5's op-level benchmark table measured."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        out = quantized_linear_reference(x2d, self.packed, self.scale, self.group_size)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)


class KernelInt4Linear(_Int4LinearBase):
    """Configuration (c): the Phase 2 optimized fused dequant+GEMM CUDA kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous().float()
        out = _kernel_ext().dequant_gemm_optimized(x2d, self.packed, self.scale, self.group_size)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)


# Mirrors Evol_inference/evol_inference/weight_bank.py's LINEAR_NAMES and
# _get_parent_and_attr -- deliberately re-declared locally (a few lines) rather than
# importing across repos, keeping this project's venv/dependencies self-contained per
# Phase 0's isolation rationale.
N_TRANSFORMER_BLOCKS = 32
LINEAR_NAMES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def iter_phi3_linears(model: nn.Module):
    """Yield (block_idx, linear_name, parent_module, attr_name, fp16_linear) for every
    quantizable linear layer in every transformer block."""
    for block_idx in range(N_TRANSFORMER_BLOCKS):
        block = model.model.layers[block_idx]
        for linear_name in LINEAR_NAMES:
            parent, attr = _get_parent_and_attr(block, linear_name)
            yield block_idx, linear_name, parent, attr, getattr(parent, attr)


def substitute_all(model: nn.Module, module_cls, device: str = "cuda") -> list[nn.Linear]:
    """Replace every quantizable linear layer's module with `module_cls(fp16_linear)`,
    moved to `device`. The original fp16 layer is quantized from (it's still needed on
    its current device to do that) and then relocated to CPU -- keeping both the fp16
    original and the new INT4 module GPU-resident at once across all 128 layers would
    roughly double peak VRAM for no reason, the same waste WeightBank.assemble avoids.
    Returns the original fp16 modules (now CPU-resident) in traversal order so the
    caller can restore them afterward (module substitution, not tensor mutation --
    same discipline as Evol_inference's WeightBank.assemble)."""
    originals = []
    for _block_idx, _name, parent, attr, fp16_linear in list(iter_phi3_linears(model)):
        new_module = module_cls(fp16_linear).to(device)
        setattr(parent, attr, new_module)
        originals.append(fp16_linear.to("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    return originals


def restore_all(model: nn.Module, originals: list[nn.Linear], device: str = "cuda") -> None:
    """Inverse of `substitute_all`: put the original fp16 layers back on `device`, in
    the same traversal order they were collected in, freeing the INT4 modules they
    replace."""
    it = iter(originals)
    for _block_idx, _name, parent, attr, _current in list(iter_phi3_linears(model)):
        setattr(parent, attr, next(it).to(device))
    gc.collect()
    torch.cuda.empty_cache()
