"""Phase 3 step 3: generate the test vectors `validate_plugin_engine.cpp` builds a
TensorRT engine around and checks. Reuses the already-validated `quantize_groupwise_int4`
/ `quantized_linear_reference` from `quant.py` rather than reimplementing quantization in
C++ -- the plugin's correctness bar is "matches the same numerical ground truth every
earlier phase was validated against," not "matches a second, independent implementation."

Writes raw float32/uint8 binaries plus a small text header (shape + group_size) into
`build/plugin_test_vectors/` (gitignored, regenerate by re-running this script).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.reference.quant import quantize_groupwise_int4, quantized_linear_reference
from python.reference.shapes import PHI3_LAYER_SHAPES

GROUP_SIZE = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "build", "plugin_test_vectors"))
    parser.add_argument("--layer", default="o_proj", choices=list(PHI3_LAYER_SHAPES.keys()))
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_features, in_features = PHI3_LAYER_SHAPES[args.layer]

    x = torch.randn(args.m, in_features, dtype=torch.float32)
    w = torch.randn(out_features, in_features, dtype=torch.float32) * 0.02
    packed, scale = quantize_groupwise_int4(w, GROUP_SIZE)
    reference = quantized_linear_reference(x, packed, scale, GROUP_SIZE)

    os.makedirs(args.out_dir, exist_ok=True)
    x.numpy().tofile(os.path.join(args.out_dir, "x.bin"))
    packed.numpy().tofile(os.path.join(args.out_dir, "packed.bin"))
    scale.float().numpy().tofile(os.path.join(args.out_dir, "scale.bin"))
    reference.numpy().tofile(os.path.join(args.out_dir, "reference.bin"))

    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"M={args.m}\nN={out_features}\nK={in_features}\ngroup_size={GROUP_SIZE}\nlayer={args.layer}\n")

    print(f"wrote test vectors for {args.layer} (M={args.m}, N={out_features}, K={in_features}) to {args.out_dir}")


if __name__ == "__main__":
    main()
