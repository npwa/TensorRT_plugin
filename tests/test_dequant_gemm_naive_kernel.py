"""Phase 1 step 4: validate the naive CUDA kernel against the PyTorch reference --
small synthetic shapes first, then the real Phi-3 layer shapes. JIT-compiled via
torch's cpp_extension loader (see src/kernels/torch_binding.cpp for why).
"""

import os

import pytest
import torch
from torch.utils.cpp_extension import load

from python.reference.quant import quantize_groupwise_int4, quantized_linear_reference
from python.reference.shapes import PHI3_LAYER_SHAPES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel requires a CUDA device")

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "kernels")


@pytest.fixture(scope="session")
def kernel():
    return load(
        name="dequant_gemm_naive_ext",
        sources=[
            os.path.join(_KERNEL_DIR, "torch_binding.cpp"),
            os.path.join(_KERNEL_DIR, "dequant_gemm_naive.cu"),
        ],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def _run_and_compare(kernel, m, out_features, in_features, group_size=128, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, in_features, device="cuda", dtype=torch.float32)
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32) * 0.02
    packed, scale = quantize_groupwise_int4(w, group_size)

    expected = quantized_linear_reference(x, packed, scale, group_size)
    actual = kernel.dequant_gemm_naive(x, packed, scale.float(), group_size)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_small_synthetic_shape(kernel):
    _run_and_compare(kernel, m=4, out_features=32, in_features=128)  # 1 group


def test_small_synthetic_shape_multi_group(kernel):
    _run_and_compare(kernel, m=3, out_features=17, in_features=256)  # 2 groups, odd dims


@pytest.mark.parametrize("name,shape", list(PHI3_LAYER_SHAPES.items()))
def test_real_phi3_shapes(kernel, name, shape):
    out_features, in_features = shape
    _run_and_compare(kernel, m=1, out_features=out_features, in_features=in_features)


def test_real_phi3_shapes_batched(kernel):
    # batch size > 1, matching how these layers are actually invoked during a forward pass
    out_features, in_features = PHI3_LAYER_SHAPES["gate_up_proj"]
    _run_and_compare(kernel, m=8, out_features=out_features, in_features=in_features)
