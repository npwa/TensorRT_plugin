# Implementation Plan: Custom CUDA Kernel + TensorRT Plugin

Planning only — no code yet. Same two-step process as `Evol_inference`
(`requirements.md` first, this document second), and the same phase-by-phase style:
each phase produces something checkable before the next depends on it.

## 0. Verified environment

Checked directly on the dev host rather than assumed, since this project's sibling
(`Evol_inference`) hit real version-compatibility friction twice already:

- **GPU**: RTX 3080, 10GB VRAM, driver 580.126.09 (CUDA 13.0 runtime, per `nvidia-smi`).
- **CUDA toolkit (`nvcc`)**: 12.0.140 — older than the driver's max-supported 13.0.
  CUDA maintains driver backward-compatibility, so an nvcc-12.0-compiled kernel should
  run fine against this driver; **confirmed, not assumed, in Phase 0 step 2** before
  anything else is built on top of it.
- **CMake**: 3.28.3 (present).
- **g++**: 13.3.0 (Ubuntu 24.04), confirmed C++20-capable.
- **Nsight Compute (`ncu`)**: already installed — no separate profiler install needed.
- **TensorRT**: not installed anywhere on this machine (no apt package, no pip package,
  no `NvInfer.h` found by filesystem search) — a full install is real Phase 0 work, not
  a formality.
- **Disk**: 463GB free — not a constraint.
- Reference PyTorch stack: `Evol_inference`'s existing venv has `torch==2.14.0+cu130`,
  reusable for Phase 1's reference implementation rather than reinstalling.

## 1. Phased steps

### Phase 0 — Environment
1. Install TensorRT. Two acquisition paths, resolve which is actually needed before
   picking one: the `tensorrt`/`tensorrt-cu13` **pip wheel** (matches this driver's
   CUDA 13 runtime per NVIDIA's current wheel-suffix scheme) is the fast path, but it's
   unclear until checked whether the pip wheel bundles the C++ headers
   (`NvInfer.h` and friends) this project actually needs, or just the Python bindings
   and runtime `.so` files. If headers are missing, fall back to NVIDIA's TensorRT SDK
   tar/deb package from the NVIDIA Developer site — that download requires an
   interactive login and EULA click-through, **not scriptable**; flag this to the user
   as a manual step rather than attempting to automate around it.
2. Verify the nvcc-12.0-vs-driver-580 pairing: compile and run a trivial `.cu` file,
   confirm it executes correctly. This is the one environment assumption in §0 that
   hasn't been directly tested yet.
3. Verify TensorRT itself: `import tensorrt; print(tensorrt.__version__)`, and build +
   run NVIDIA's own minimal quickstart engine, before writing any project-specific code.
4. **Decide the plugin API target from the actual installed version**: `IPluginV3`
   (TensorRT ≥10.0) if available, `IPluginV2DynamicExt` otherwise. Don't pre-commit —
   requirements.md §7 already flags this API has shifted across versions.
5. Set up the project skeleton: `CMakeLists.txt`, `src/kernels/`, `src/plugin/`,
   `src/harness/`, `python/reference/`, `tests/`. Separate Python venv from
   `Evol_inference`'s (TensorRT's pinned dependencies are a real candidate for conflicts
   with the existing bitsandbytes/torch install — keep them isolated).

### Phase 1 — Reference implementation + naive kernel
1. Pull real linear-layer shapes from Phi-3-mini-4k-instruct's `qkv_proj`, `o_proj`,
   `gate_up_proj`, `down_proj` — read directly from `Evol_inference`'s already-downloaded
   model at `~/work/Evol_inference/models/phi-3-mini-4k-instruct`, no re-download needed.
2. **Decide and document the quantization scheme now, since it drives kernel design**:
   plain groupwise symmetric INT4 (FP16 scale per group of N weights, N TBD — 128 is a
   common, reasonable default) — not an attempt to replicate bitsandbytes' NF4 exactly.
   The project's point is kernel authorship, not quantization-scheme novelty; a
   well-understood, standard scheme keeps Phase 1 tractable.
3. Write the PyTorch reference (plain ops: unpack, dequantize, matmul) as the numerical
   ground truth every later phase validates against.
4. Write a naive CUDA kernel — one thread per output element, correctness over
   performance. Validate against the PyTorch reference on small synthetic shapes first
   (e.g. 32×32), then the real Phi-3 shapes from step 1.

### Phase 2 — Optimize (time-boxed)
1. Profile the naive kernel with `ncu`; confirm whether it's memory- or compute-bound
   before optimizing blind (a batch-1 dequant+GEMM is very likely memory-bandwidth-bound
   — consistent with `Evol_inference`'s own finding that INT4 dequant overhead dominates
   at small batch sizes on this same GPU).
2. Add shared-memory tiling for weight-tile reuse, then coalesced/vectorized loads
   (e.g. packed `uchar4`/`int4`-width loads for the 4-bit weights).
3. Build a cuBLAS/cuBLASLt "dequantize-then-GEMM" baseline as the comparison point, not
   just the naive kernel.
4. **Hard stop rule**: time-box this phase (recommend 1 week) and report whatever
   speedup exists at that point, win or lose against cuBLAS — per requirements.md §6's
   stated success criterion. Kernel optimization has no natural stopping point; the
   time-box is what prevents it from eating the rest of the schedule.

### Phase 3 — TensorRT Plugin
1. Implement the plugin class decided in Phase 0 step 4, wrapping the Phase 2 kernel:
   output-shape inference, `configurePlugin`, `enqueue`, serialization of the group-size/
   scale-tensor attributes.
2. Build a minimal single-layer `INetworkDefinition` using only this custom op via the
   C++ builder API; register the plugin creator.
3. Validate engine output against the Phase 1 PyTorch reference, same shapes.
4. Scope decision, stated explicitly rather than defaulted into: **static shapes for
   v1** (no dynamic batch/sequence support) — defensible since the rest of the project's
   methodology already uses one fixed 2048-token workload; revisit only if a real need
   for dynamic shapes shows up later.

### Phase 4 — C++ inference harness
1. CMake, C++20. RAII wrappers (custom-deleter smart pointers) for `IRuntime`,
   `ICudaEngine`, `IExecutionContext` — confirm in Phase 0 whether the installed
   TensorRT version's objects are destroyed via `->destroy()` or plain `delete` (this
   changed across TensorRT major versions; write the deleter against the confirmed API,
   not a guess).
2. Single-threaded correctness pass first: load the engine, run one inference, diff
   against a reference output serialized from Phase 3's Python-side validation. Get this
   right before adding any concurrency.
3. Add concurrency: multiple CUDA streams plus a small thread pool (or one
   `std::jthread` per worker — C++20, auto-joining, fits the memory-safety framing)
   submitting independent requests, with real synchronization discipline (no shared
   mutable state races). The goal is genuine overlap (e.g. one request's H2D copy
   overlapping another's compute) — not just "multiple threads call the same function."
4. Timing: warmup + median-of-N per request, same methodology as
   `evol_inference.fitness.measure_latency_ms`, so the numbers are directly comparable
   in the final write-up.

### Phase 5 — Parity + benchmarking write-up
1. Turn requirements.md §6's three-level validation ladder (kernel → plugin → engine)
   into an actual test suite (`ctest`/a small Python test harness), not one-off manual
   checks — matches `Evol_inference`'s testing discipline.
2. Expose the standalone kernel to Python (a thin `pybind11` wrapper — small, cheap
   addition) so it can sit in the same benchmark table as the PyTorch numbers.
3. Benchmark table: PyTorch eager, `torch.compile`, standalone CUDA kernel, full
   TensorRT-plugin engine via the C++ harness. Same shapes, same warmup/median-of-N
   methodology throughout.
4. README write-up in `Evol_inference`'s voice: real numbers, real hardware disclosed,
   findings reported whether or not they're flattering.

### Phase 6 — Speculative decoding (selected stretch, requirements.md §8)
1. Pick a draft model. Real constraint to resolve here, not before: it needs a
   tokenizer compatible with Phi-3's for the accept/reject comparison to work cleanly —
   candidates to evaluate in this phase, not pre-committed.
2. Implement the standard loop: draft model proposes K tokens, Phi-3 verifies in one
   batched forward pass, accept/reject per the standard speculative-decoding rule.
3. Start as a pure PyTorch proof-of-concept (requirements.md §8 already notes this needs
   no new CUDA work) — a complete, honest deliverable on its own. Routing the target
   model's forward pass through the Phase 4 C++/TensorRT harness is a further stretch
   only if time allows, not a v1-of-the-stretch requirement.
4. Benchmark: tokens/sec with vs. without speculative decoding, same fixed-workload
   methodology as the rest of the project.

## 2. Packages / tooling

- **CUDA**: `nvcc` 12.0.140 (already installed; Phase 0 step 2 confirms it against the
  580 driver).
- **TensorRT**: SDK — pip wheel (`tensorrt-cu13`) or NVIDIA direct download; resolved in
  Phase 0 step 1, may require a manual interactive step.
- **CMake** ≥3.28, **g++** 13.3.0 / C++20 (both already installed).
- **Nsight Compute** (`ncu`, already installed) — Phase 2 profiling.
- **Python**: a fresh venv, isolated from `Evol_inference`'s (avoid TensorRT/torch
  dependency conflicts); `torch` for the Phase 1 reference; `pybind11` for Phase 5's
  kernel-benchmark wrapper.
- **Draft model** for Phase 6: TBD, resolved in Phase 6 itself.

## 3. Open decisions carried forward (not resolved yet, by design)

- TensorRT SDK acquisition path (pip vs. NVIDIA direct download) — Phase 0.
- `IPluginV3` vs `IPluginV2DynamicExt` — Phase 0, once the installed version is known.
- TensorRT object destruction API (`destroy()` vs `delete`) — Phase 0/4.
- Quantization group size (128 proposed, not fixed) — Phase 1.
- Draft model for speculative decoding — Phase 6.

Each of these is deferred deliberately — the pattern this whole project (and
`Evol_inference` before it) follows is verify-then-decide, not guess-then-hope.
