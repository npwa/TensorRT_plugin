# TensorRT_plugin

**Custom CUDA kernel + TensorRT plugin for quantized LLM inference** — a weight-only
INT4 dequantize-and-GEMM kernel, wrapped as a TensorRT plugin, driven from a
production-style C++ inference harness, benchmarked with the same rigorous
parity-checking discipline as
[`Evol_inference`](https://github.com/npwa/Evol_inference/blob/main/README.md).

**Status: Phases 0–6 complete** (environment, quantization scheme + naive kernel,
optimized kernel, TensorRT plugin, C++ harness, validation + benchmarking, and the
selected speculative-decoding stretch). See
[`Doc/requirements.md`](Doc/requirements.md) for the scope and why this project exists
as a standalone thing rather than an extension of `Evol_inference`, and
[`Doc/implementation_plan.md`](Doc/implementation_plan.md) for the phase-by-phase build
log, including every dead end and fix along the way.

## Key findings

1. **The hand-written kernel and the TensorRT plugin that wraps it are, within
   measurement noise, the same speed** — 0.0759ms vs. 0.0768ms per request at the
   benchmarked shape. A thin plugin wrapper *should* add no real overhead; here that's a
   measured result, not an assumption.
2. **Dequantizing into a dense matrix before the matmul is the slow path, and it shows up
   twice, independently.** A cuBLAS baseline built the "obvious alternative" way
   (unpack INT4 to a dense fp32 matrix, then call `cublasSgemv`) came in 4.6x slower than
   the fused kernel — slower even than the naive fused kernel. The same pattern reappears
   in the benchmark table below: PyTorch eager (which also materializes a dense
   dequantized weight before matmul) is the slowest of all four implementations.
3. **The optimized kernel is 4.3x faster than the naive one**, from a concrete
   Nsight-Compute diagnosis (86% of memory sectors wasted to uncoalesced access) rather
   than guessing at what to optimize — re-profiling after the fix confirmed sectors/request
   dropped from 10.67 to 1.60 and SM throughput rose from 38% to 81%, i.e. the fix worked
   for the diagnosed reason.
4. **`torch.compile` closes part of the gap to the hand-written kernel but not all of
   it**: 4.1x faster than eager, but still ~2.2x slower than the kernel/engine.
5. **Multi-stream concurrency in the C++ harness gives a real but modest throughput
   gain that saturates fast**: 1.37x aggregate throughput over a fully-serial baseline at
   4 concurrent streams, and no further gain at 8. Reported as measured — this is a small,
   single-op workload where H2D/D2H copy time is already a small fraction of the request,
   so there's limited copy-vs-compute overlap left to exploit.

## The approach

- **Quantization**: plain groupwise symmetric INT4 (group size 128, real 2-per-byte bit
  packing) — a standard, well-understood scheme, deliberately not an attempt to
  replicate bitsandbytes' NF4. The point of this project is kernel/systems authorship,
  not quantization-scheme novelty.
- **Kernel**: a warp-per-output-row CUDA kernel — activations staged into shared memory
  once per row, packed weights read via vectorized loads, partial sums combined with a
  warp-shuffle reduction. Validated against a PyTorch reference on real Phi-3 layer
  shapes before being optimized, and re-validated after.
- **Plugin**: the kernel wrapped as a TensorRT `IPluginV3` (the current-generation
  plugin API in TensorRT 11.3.0.99, confirmed against the installed headers rather than
  assumed) — weights baked in as plugin attributes at network-build time, the activation
  tensor as the only runtime input.
- **Harness**: a C++20 RAII wrapper around the TensorRT runtime (plain `std::unique_ptr`,
  no custom deleter — confirmed this TensorRT version doesn't need the legacy
  `->destroy()` pattern) with genuine multi-stream concurrency: each worker owns its own
  execution context, CUDA stream, and pinned host buffers, so one request's copy can
  overlap another's compute.
- **Validation**: a three-level parity ladder (kernel → plugin → full engine, each
  checked against a PyTorch reference), wired into a real `ctest` suite for the
  plugin/engine rungs and the existing `pytest` suite for the kernel rung — not one-off
  manual checks.

## Benchmark table

Same weights, same input, same warmup(3) + median-of-50 timing methodology across all
four rows (shape M=2, N=3072, K=3072, group_size=128, on the hardware below):

| Implementation | ms/request |
|---|---|
| PyTorch eager | 0.6778 |
| `torch.compile` | 0.1647 |
| Standalone CUDA kernel | 0.0759 |
| TensorRT plugin engine (C++ harness) | 0.0768 |

Reproduce with `python/benchmarks/benchmark_table.py` (after `cmake --build build` and
`ctest` in `build/`, which produce the harness binary and the serialized engine plan the
fourth row shells out to).

## Hardware

Every result above was produced on the same machine as `Evol_inference`:

| Component | Spec |
|-----------|------|
| CPU | 11th Gen Intel Core i7-11700K @ 3.60GHz |
| Memory | 64GiB (4×16GiB DDR4, 2667 MHz) |
| Disk | 1TB NVMe + 2TB SSD |
| GPU | NVIDIA GeForce RTX 3080, 10GB VRAM |
| PCIe | 4.0 over a x16 lane width (16GT/s) |

## Why this exists

`Evol_inference` is an evolutionary-search project: it treats per-layer quantization
choice as a combinatorial optimization problem and searches it. That's an
algorithm-and-search story. This project is a different, complementary one:
**systems/kernel engineering** — actually writing the low-level CUDA code that makes a
quantized layer fast (or doesn't), integrating it into a real inference-serving
toolchain (TensorRT), and wrapping it in production-style C++.

Same target model (Phi-3-mini-4k-instruct), same hardware (RTX 3080), same real
benchmarking ethos — real numbers, real hardware disclosed, findings reported whether or
not they're flattering — but a different half of the stack.

## Stretch: speculative decoding (Phase 6)

Greedy speculative decoding (`TinyLlama-1.1B-Chat-v1.0` draft, Phi-3-mini target) —
tokenizer compatibility with the target was empirically verified before picking the
draft model (identical shared 32000-token Llama-2 vocabulary, confirmed with zero
mismatches), and the implementation's output was independently checked byte-for-byte
against plain greedy generation before any timing number was trusted.

**Negative Result: this draft/target pairing is slower, not faster** — mean 0.26x (~3.8x
slower) at the standard k=4 draft window. Root cause, measured directly rather than
assumed: only a 9% draft acceptance rate, because TinyLlama and Phi-3-mini are
independently-trained models that happen to share a tokenizer, not models from the same
lineage — sharing a vocabulary turned out to be necessary for this to work at all, but
far from sufficient for the draft's guesses to actually agree with the target's often
enough to pay for themselves. See
[`Doc/implementation_plan.md`](Doc/implementation_plan.md)'s Phase 6 section for the two
real bugs the correctness check caught before this number could be trusted, and the full
k=2 vs. k=4 comparison.

## Adapting this to other models

Not a blanket "easily portable" — it splits cleanly into a part that is and a part
that isn't.

**Portable with modest effort**: the kernel (`dequant_gemm_optimized`) and the
quantization scheme (`quantize_groupwise_int4`) have no Phi-3-specific logic at all —
they operate on any bias-free `[out_features, in_features]` weight matrix, from any
model. The TensorRT plugin (`DequantGemmPluginV3`) is the same story: `N`/`K`/
`group_size`/weights are just build-time attributes. The one place with real
Phi-3-specific plumbing is the PyTorch-level module-substitution path
(`python/benchmarks/phase5b_modules.py`), which hardcodes this model's attribute names
(`self_attn.qkv_proj`, `mlp.gate_up_proj`, ...) and block count — porting to another
dense-transformer architecture (Llama/Mistral-family models typically split `q_proj`/
`k_proj`/`v_proj` instead of a fused `qkv_proj`) means updating those names to match,
roughly an hour's work per new architecture, not a redesign.

**Not easy — real, currently-missing engineering**: the plugin/engine only ever wraps
**one linear layer in isolation**; there is no full-model TensorRT engine, since
attention, RMSNorm, embeddings, and KV-cache don't exist as TensorRT ops here.
Deploying a different model at scale through TensorRT would mean writing a new
network-construction script chaining many plugin instances together for that model's
full layer stack — mechanical, but real additional work. Every engine is also built for
one **static shape** (Phase 3's explicit v1 scope decision), so varying request sizes in
production would need dynamic-shape support that isn't built. Most importantly,
attention and KV-cache as TensorRT ops were explicitly **deferred** in
[`Doc/requirements.md`](Doc/requirements.md) §8 as separate, harder stretch goals — that
gap, not the kernel or plugin code, is what actually stands between "this technique
works" and "this is a deployable engine for a different model at production scale."
