# TensorRT_plugin

**Custom CUDA kernel + TensorRT plugin for quantized LLM inference** — a weight-only
INT4 dequantize-and-GEMM kernel, wrapped as a TensorRT plugin, driven from a
production-style C++ inference harness, benchmarked with the same rigorous
parity-checking discipline as [`Evol_inference`](../Evol_inference).

**Status: planning complete, implementation not started.** See
[`Doc/requirements.md`](Doc/requirements.md) for the scope and why this project exists
as a standalone thing rather than an extension of `Evol_inference`, and
[`Doc/implementation_plan.md`](Doc/implementation_plan.md) for the phase-by-phase plan,
verified environment, and open decisions.

## Why this exists

`Evol_inference` is an evolutionary-search project: it treats per-layer quantization
choice as a combinatorial optimization problem and searches it. That's an
algorithm-and-search story. This project is a different, complementary one:
**systems/kernel engineering** — actually writing the low-level CUDA code that makes a
quantized layer fast (or doesn't), integrating it into a real inference-serving
toolchain (TensorRT), and wrapping it in production-style C++.

Same target model (Phi-3-mini-4k-instruct), same hardware (RTX 3080), same honest
benchmarking ethos — real numbers, real hardware disclosed, findings reported whether or
not they're flattering — but a different half of the stack.

## What this will cover

- A hand-written CUDA kernel for weight-only INT4 dequantize + GEMM, validated against a
  PyTorch reference, optimized (shared-memory tiling, coalesced/vectorized access),
  profiled with Nsight Compute — benchmarked against both its own naive version and a
  cuBLAS baseline. **Stated success criterion: correct, profiled, and honestly
  reported — not "must beat cuBLAS."**
- The kernel wrapped as a TensorRT plugin (`IPluginV3` or `IPluginV2DynamicExt`,
  decided once the installed TensorRT version is confirmed).
- A C++20 inference harness: RAII ownership of TensorRT objects, real concurrency
  (multiple CUDA streams / a thread pool — not a single-threaded call into TensorRT).
- A three-level parity ladder (kernel → plugin → full engine, each checked against a
  PyTorch reference) and a benchmark table spanning PyTorch eager, `torch.compile`, the
  standalone kernel, and the full TensorRT-plugin engine.
- **Stretch (selected): a minimal speculative-decoding harness** (small draft model +
  Phi-3 as target, accept/reject loop) — the cheapest of the three stretch options
  considered, since it needs no new CUDA work.

## Effort estimate

~4–7 weeks part-time for the core kernel/plugin/harness/benchmark work, +1–2 weeks for
the speculative-decoding stretch. Full reasoning and risk ranking in
[`Doc/implementation_plan.md`](Doc/implementation_plan.md).

## Layout (planned)

```
Doc/                requirements.md, implementation_plan.md
src/kernels/         the CUDA kernel(s)
src/plugin/          TensorRT plugin wrapper
src/harness/         C++ inference harness
python/reference/    PyTorch reference implementation + benchmark scripts
tests/               the parity ladder as an actual test suite
```

Not yet created — this is the planned layout from `Doc/implementation_plan.md`, filled
in as each phase lands.
