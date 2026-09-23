"""Phase 5 step 3: the benchmark table requirements.md §5 point 6 asks for --
PyTorch eager, `torch.compile`, the standalone CUDA kernel, and the full TensorRT-plugin
engine via the C++ harness -- same shape, same weights, same warmup+median-of-N timing
methodology throughout (matching `evol_inference.fitness.measure_latency_ms` and this
project's own `bench_naive.cu`/`bench_optimized.cu` convention).

Reuses the exact test vectors (`build/plugin_test_vectors`) the TensorRT engine was
validated and timed against, rather than generating fresh random data for the PyTorch/
kernel rows -- all four rows run the identical weights and input, so the table is a real
apples-to-apples comparison, not four separately-generated numbers that happen to share a
shape.

Run after the Phase 5 ctest suite (`ctest` in `build/`), which builds
`build/plugin_test_vectors/engine.plan` and the harness binaries this script shells out
to for the fourth row.
"""

import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

# torch.utils.cpp_extension.load needs `ninja` on PATH -- the project runs via
# `.venv/bin/python` directly rather than an activated venv, so PATH needs `.venv/bin`
# prepended explicitly (Doc/implementation_plan.md, Phase 1 completion note).
os.environ["PATH"] = os.path.join(_PROJECT_ROOT, ".venv", "bin") + os.pathsep + os.environ.get("PATH", "")

from python.reference.quant import quantized_linear_reference  # noqa: E402

VECTORS_DIR = os.path.join(_PROJECT_ROOT, "build", "plugin_test_vectors")
RUN_CONCURRENT_BIN = os.path.join(_PROJECT_ROOT, "build", "src", "harness", "run_concurrent")
KERNEL_DIR = os.path.join(_PROJECT_ROOT, "src", "kernels")

N_WARMUP = 3
N_RUNS = 50


def _read_meta():
    meta = {}
    with open(os.path.join(VECTORS_DIR, "meta.txt")) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            meta[key] = value
    return int(meta["M"]), int(meta["N"]), int(meta["K"]), int(meta["group_size"])


def _load_tensors(m, n, k, group_size):
    x = torch.from_numpy(np.fromfile(os.path.join(VECTORS_DIR, "x.bin"), dtype="float32")).reshape(m, k).cuda()
    packed = torch.from_numpy(np.fromfile(os.path.join(VECTORS_DIR, "packed.bin"), dtype="uint8")).reshape(
        n, k // 2
    ).cuda()
    scale = torch.from_numpy(np.fromfile(os.path.join(VECTORS_DIR, "scale.bin"), dtype="float32")).reshape(
        n, k // group_size
    ).cuda()
    return x, packed, scale


def _median_latency_ms(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def benchmark_pytorch_eager(x, packed, scale, group_size):
    return _median_latency_ms(lambda: quantized_linear_reference(x, packed, scale, group_size))


def benchmark_torch_compile(x, packed, scale, group_size):
    compiled = torch.compile(quantized_linear_reference)
    return _median_latency_ms(lambda: compiled(x, packed, scale, group_size))


def benchmark_standalone_kernel(x, packed, scale, group_size):
    from torch.utils.cpp_extension import load

    ext = load(
        name="dequant_gemm_ext_benchmark",
        sources=[
            os.path.join(KERNEL_DIR, "torch_binding.cpp"),
            os.path.join(KERNEL_DIR, "dequant_gemm_naive.cu"),
            os.path.join(KERNEL_DIR, "dequant_gemm_optimized.cu"),
        ],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    scale_f32 = scale.float()
    return _median_latency_ms(lambda: ext.dequant_gemm_optimized(x, packed, scale_f32, group_size))


def benchmark_tensorrt_engine():
    """Shells out to the already-built C++ harness rather than re-implementing TensorRT
    engine invocation in Python -- the harness (src/harness/run_concurrent.cpp) is the
    one real, validated measurement of this number; parsing its own reported median
    keeps this table's fourth row and the harness's own completion-note number as a
    single source of truth instead of two numbers that could silently drift apart."""
    if not os.path.exists(RUN_CONCURRENT_BIN):
        raise FileNotFoundError(
            f"{RUN_CONCURRENT_BIN} not found -- build the project first (`cmake --build build`)"
        )
    engine_plan = os.path.join(VECTORS_DIR, "engine.plan")
    if not os.path.exists(engine_plan):
        raise FileNotFoundError(f"{engine_plan} not found -- run `ctest` in build/ first")

    result = subprocess.run(
        [RUN_CONCURRENT_BIN, engine_plan, "1", str(N_RUNS)],
        capture_output=True, text=True, check=True,
    )
    match = re.search(r"single-stream baseline: median ([\d.]+) ms/request", result.stdout)
    if not match:
        raise RuntimeError(f"could not parse run_concurrent output:\n{result.stdout}")
    return float(match.group(1))


def main():
    m, n, k, group_size = _read_meta()
    x, packed, scale = _load_tensors(m, n, k, group_size)
    print(f"shape: M={m} N={n} K={k} group_size={group_size} (warmup={N_WARMUP}, median of {N_RUNS} runs)\n")

    rows = [
        ("PyTorch eager", benchmark_pytorch_eager(x, packed, scale, group_size)),
        ("torch.compile", benchmark_torch_compile(x, packed, scale, group_size)),
        ("standalone CUDA kernel", benchmark_standalone_kernel(x, packed, scale, group_size)),
        ("TensorRT plugin engine (C++ harness)", benchmark_tensorrt_engine()),
    ]

    width = max(len(name) for name, _ in rows)
    for name, ms in rows:
        print(f"{name.ljust(width)}  {ms:8.4f} ms/request")


if __name__ == "__main__":
    main()
