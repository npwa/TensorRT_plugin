"""Phase 1 step 3 validation: the PyTorch reference implementation this whole project
is grounded on. Run with `PYTHONPATH=. .venv/bin/python -m pytest tests/`.
"""

import pytest
import torch

from python.reference.quant import (
    GROUP_SIZE,
    dequantize_groupwise_int4,
    pack_int4,
    quantize_groupwise_int4,
    quantized_linear_reference,
    unpack_int4,
)
from python.reference.shapes import PHI3_LAYER_SHAPES

torch.manual_seed(0)


def test_pack_unpack_roundtrip_is_exact():
    q = torch.randint(-8, 8, (17, 128), dtype=torch.int8)  # odd leading dim on purpose
    packed = pack_int4(q)
    assert packed.shape == (17, 64)
    assert packed.dtype == torch.uint8
    restored = unpack_int4(packed)
    assert torch.equal(restored, q)


def test_pack_unpack_covers_full_range():
    # every representable value, not just a random sample
    q = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


@pytest.mark.parametrize("out_features,in_features", [(4, 128), (17, 256), (64, 3072)])
def test_quantize_dequantize_error_is_bounded_by_half_scale(out_features, in_features):
    w = torch.randn(out_features, in_features)
    packed, scale = quantize_groupwise_int4(w)
    w_hat = dequantize_groupwise_int4(packed, scale)

    n_groups = in_features // GROUP_SIZE
    max_abs_err = (w - w_hat).abs().reshape(out_features, n_groups, GROUP_SIZE).amax(dim=-1)
    # rounding to the nearest representable multiple of `scale` bounds the error at
    # half a scale step, modulo fp16 storage of the scale itself
    scale_bound = scale.float() / 2 + 1e-3
    assert (max_abs_err <= scale_bound).all()


def test_quantized_linear_matches_fp32_linear_within_tolerance():
    x = torch.randn(5, 256)
    w = torch.randn(64, 256)
    fp32_out = x @ w.T

    packed, scale = quantize_groupwise_int4(w)
    quant_out = quantized_linear_reference(x, packed, scale)

    rel_err = (quant_out - fp32_out).norm() / fp32_out.norm()
    # ~12% measured on real Phi-3 weights below (this scheme has no calibration, no
    # outlier handling, no NF4-style non-uniform levels -- plain symmetric INT4 really
    # does lose this much); 20% bounds the sanity check without being vacuous against
    # an actual bug (e.g. a scale-factor error would push this well past 50%).
    assert rel_err < 0.20


@pytest.mark.parametrize("name,shape", list(PHI3_LAYER_SHAPES.items()))
def test_real_phi3_shapes_quantize_and_run(name, shape):
    out_features, in_features = shape
    assert in_features % GROUP_SIZE == 0, f"{name}: in_features must divide GROUP_SIZE"

    w = torch.randn(out_features, in_features) * 0.02  # roughly weight-scale magnitude
    packed, scale = quantize_groupwise_int4(w)
    assert packed.shape == (out_features, in_features // 2)
    assert scale.shape == (out_features, in_features // GROUP_SIZE)

    x = torch.randn(1, in_features)
    out = quantized_linear_reference(x, packed, scale)
    assert out.shape == (1, out_features)
    assert torch.isfinite(out).all()


def test_real_phi3_down_proj_weights_quantize_within_tolerance():
    """Not random data -- the actual down_proj weights from block 0 of the real model,
    reused from the already-downloaded Evol_inference model rather than re-downloading."""
    from safetensors import safe_open

    path = "/home/npalmass/work/Evol_inference/models/phi-3-mini-4k-instruct/model-00001-of-00002.safetensors"
    with safe_open(path, framework="pt") as f:
        keys = f.keys()
        name = "model.layers.0.mlp.down_proj.weight"
        if name not in keys:
            pytest.skip(f"{name} not in this shard; check the other shard")
        w = f.get_tensor(name).float()

    assert tuple(w.shape) == PHI3_LAYER_SHAPES["down_proj"]

    packed, scale = quantize_groupwise_int4(w)
    w_hat = dequantize_groupwise_int4(packed, scale)
    rel_err = (w - w_hat).norm() / w.norm()
    print(f"\ndown_proj groupwise-INT4 relative Frobenius error: {rel_err.item():.4f}")
    # measured ~0.12 on the real weights; see the tolerance note in
    # test_quantized_linear_matches_fp32_linear_within_tolerance for why 0.20 is the
    # right bound here rather than a tighter, aspirational number.
    assert rel_err < 0.20
