# Requirements: Custom CUDA Kernel + TensorRT Plugin for Quantized LLM Inference

This is the scope sketch for a new, standalone project — not an extension of
`Evol_inference`. It exists to extend what that project with systems/ kernel-engineering
features.

## 1. Goal

Write a custom CUDA kernel for a real LLM inference operation, wrap it as a TensorRT
plugin, drive it from a small production-style C++ inference harness, and benchmark the
whole stack (PyTorch eager → PyTorch `torch.compile` → standalone CUDA kernel →
TensorRT engine with the custom plugin) with the same rigorous parity-checking discipline
already established in `Evol_inference`.

## 2. Why this project, and how it relates to `Evol_inference`

`Doc/implementation_plan_benchmark.md` in `Evol_inference` (an ONNX/TensorRT
*benchmarking* extension to that project) covers one assignment — conversion pipelines,
parity checking, latency benchmarking — using TensorRT's built-in APIs. These features
are not currently covered:

- "Develop and optimize custom ML OPs and TensorRT Plugins with efficient CUDA kernel
  implementations" — a core qualification.
- "Production-level, low latency, and memory-safe C++ ... for real-time inference".
- Efficient attention / KV-cache optimization / speculative decoding.

These are a different skill axis (systems/kernel engineering) from `Evol_inference`'s
algorithm-and-search story, and stretching that project to cover them would read as
forced. This project's job is to cover them directly, using the same target model
(Phi-3-mini) and the same benchmarking ethos as a throughline, so the two projects read
as one coherent portfolio rather than two unrelated ones.

## 3. Scope for v1

- **Target op**: a fused weight-only quantized (INT4, falling back to INT8 if INT4
  proves too slow to get correct) dequantize-and-GEMM kernel — the actual low-level
  operation that makes a quantized linear layer fast (or doesn't). Chosen over a
  simpler elementwise op because it's genuinely load-bearing (this is what
  `evol_inference`'s bitsandbytes kernels do under the hood, and what TensorRT-LLM's
  weight-only quantization paths do), and chosen over a full attention kernel because a
  correct, reasonably optimized GEMM is achievable in the time budget below where a
  from-scratch FlashAttention-equivalent is not (§7).
- **Shapes**: real Phi-3-mini-4k-instruct linear-layer shapes (`gate_up_proj`,
  `down_proj`, `qkv_proj`, `o_proj`), reusing domain knowledge and, where useful, actual
  weight tensors from `Evol_inference`'s already-downloaded model.
- **Hardware**: the same RTX 3080 (Ampere) already in use — not vehicle SoC hardware
  (unavailable), so the project's framing is explicit that this demonstrates the
  *methodology* (kernel authorship, plugin integration, C++ harness, rigorous
  benchmarking) rather than literal on-vehicle power/thermal validation.
- **Explicitly out of scope for v1**: full FlashAttention-equivalent kernel, real
  KV-cache paging, speculative decoding, QAT, multi-GPU/distributed anything. Candidates
  for v2 (§8), not committed.

## 4. Effort estimate

| Scope | Estimate | Confidence |
|---|---|---|
| v1 (kernel → plugin → C++ harness → benchmarks) | **4–7 weeks part-time** (~17–28 focused days) | Moderate — Phase 2 (kernel optimization) and Phase 3 (plugin API) are the open-ended/fiddly items |
| v2 stretch (one of: attention kernel, KV-cache paging, speculative decoding) | **+1–3 weeks part-time**, depending on which | Low — pick one, not all three (§8) |

This is a noticeably bigger lift than the `Evol_inference` TensorRT benchmark extension
(2–4 weeks), which makes sense: that project calls existing TensorRT APIs, this one
authors new CUDA code and a new plugin from scratch, then wraps it in a new language
(C++) with its own toolchain.

## 5. Phases (high level — a full implementation plan is the next document, once this
scope is agreed)

1. **Environment**: TensorRT C++ SDK (headers + libs, not just the Python bindings used
   in the sibling project), CMake, a pinned CUDA/TensorRT version combination (verify
   compatibility before committing — this project has hit exactly this kind of
   version-friction twice already in `Evol_inference`, expect it again).
2. **Reference + naive kernel**: PyTorch reference implementation for ground truth; a
   correctness-first (not performance-first) CUDA kernel, validated against it on real
   Phi-3 layer shapes before any optimization work starts.
3. **Optimize**: shared-memory tiling, coalesced/vectorized memory access, profiled with
   Nsight Compute — time-boxed explicitly, since kernel optimization has no natural
   stopping point. Benchmarked against both the naive kernel and a cuBLAS/cuBLASLt
   dequantize-then-GEMM baseline.
4. **TensorRT Plugin**: wrap the optimized kernel as an `IPluginV3` (or
   `IPluginV2DynamicExt` if the installed TensorRT version forces it — verify in Phase
   1, don't assume). Validate a minimal test engine's output against the PyTorch
   reference.
5. **C++ inference harness**: CMake project; RAII ownership of TensorRT objects (no raw
   `new`/`delete`); a small concurrent-execution demo (multiple CUDA streams, a thread
   pool or async queue) to genuinely exercise "concurrent, real-time inference code,"
   not just a single-threaded `main()` that calls into TensorRT once.
6. **Parity + benchmarking write-up**: the three-level validation ladder below, plus a
   results table spanning PyTorch eager, `torch.compile` (cheap to add, and explicitly
   named in the JD alongside ONNX/TensorRT), the standalone kernel, and the full
   TensorRT-plugin engine.

## 6. Validation philosophy

A three-level parity ladder, mirroring the "test what doesn't need the GPU separately
from what does" discipline already used in `Evol_inference`:

1. **Kernel-level**: unit tests, standalone kernel output vs. a naive PyTorch reference,
   small shapes first.
2. **Plugin-level**: the kernel wrapped in a minimal TensorRT test engine vs. the same
   PyTorch reference.
3. **Engine-level**: the full assembled engine's output vs. a full PyTorch forward pass
   on real Phi-3 layer shapes.

Explicit, stated-up-front success criterion: **"correct, profiled, and honestly
reported" — not "must beat cuBLAS."** A hand-written kernel losing to a vendor library
is a completely legitimate, reportable outcome (directly analogous to the "INT4 is
slower than FP16 in practice" finding already documented in `Evol_inference`), not a
failure condition. Setting the bar here up front matters because it's a realistic one —
beating cuBLAS is a genuinely hard, open-ended optimization problem.

## 7. Risks, ranked

1. **Kernel correctness/debugging time is unpredictable** if CUDA kernel authorship from
   scratch is new territory — indexing bugs, race conditions, and uncoalesced-memory
   bugs that are silently wrong rather than loudly wrong are the classic time sinks.
   Budget slack into Phase 2 rather than treating the estimate as tight.
2. **The TensorRT Plugin API is genuinely fiddly and has shifted across versions**
   (`IPluginV2` → `IPluginV2DynamicExt` → `IPluginV3`) — pin the target version in Phase
   1 and write against its actual documented API, not whichever version's tutorial is
   easiest to find.
3. **No guarantee of a speedup over cuBLAS** — see §6's stated success criterion; this
   is a risk to the "impressiveness" of the headline number, not to the project's
   validity.
4. **C++/CMake/TensorRT-C++-SDK setup friction** is a different failure mode than
   anything hit so far (everything in `Evol_inference` has been Python) — budget real
   time in Phase 1 for this specifically.
5. **Scope creep via §8** — each stretch option is substantial on its own; resist doing
   more than one.

## 8. v2 stretch options (pick at most one)

- DEFERRED: **A second, harder kernel**: a simplified fused attention kernel (QK^T →
  softmax → V), explicitly *not* claiming FlashAttention-level performance or numerical
  sophistication — a real fused kernel, honestly scoped down. Directly answers the
  "Efficient Attention mechanisms" bullet.
- DEFERRED: **Basic paged KV-cache allocator** in the C++ harness — block-based KV-cache
  allocation and reuse across concurrent requests. More a systems/memory-management
  problem than a kernel-authorship one, so it fits naturally alongside Phase 5's C++
  harness work rather than requiring new CUDA expertise. Answers the "KV-cache
  optimization (PagedAttention)" bullet.
- SELECTED: **Minimal speculative decoding harness** — small draft model + Phi-3 as the
  target model, accept/reject loop. The cheapest of the three: doesn't require new CUDA
  work at all, could start as a pure PyTorch proof-of-concept before (optionally) routing
  through the C++ harness. Answers the "Speculative Decoding" bullet directly.

## 9. Deliverables

- Code: the kernel, the plugin, the C++ harness, and the PyTorch reference — clean
  enough to read, matching `Evol_inference`'s existing bar (tests for what doesn't need
  a GPU, honest comments only where the *why* isn't obvious).
- A benchmark table and write-up in the same voice as `Evol_inference`'s README: real
  numbers, real hardware disclosed, findings reported whether or not they're flattering.
- Naming: this directory (`~/work/TensorRT_plugin`) is a placeholder; fine to keep, or
  rename once the kernel target is finalized in Phase 2.

## 10. Next step

Once this scope is agreed, the next document is a phase-by-phase implementation plan
(package lists, concrete file layout, verified environment versions) — the same two-step
process `Evol_inference` used (`requirements.md` first, then `implementation_plan.md`).
Not written yet; this file is the scope sketch only.
