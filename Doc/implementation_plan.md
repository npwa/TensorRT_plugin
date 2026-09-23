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

**Phase 1 complete.** Real shapes confirmed from the safetensors index, not the
architecture doc (`python/reference/shapes.py`) -- all four projection types are
bias-free, and every `in_features` divides evenly into 128-element groups. Scheme
documented and implemented in `python/reference/quant.py` (groupwise symmetric INT4,
group size 128, real 2-per-byte bit-packing). The naive kernel
(`src/kernels/dequant_gemm_naive.cu`) matches the PyTorch reference to `rtol=1e-4`
across small synthetic shapes, all four real Phi-3 shapes, and a batched case (18 tests
total, `tests/`). Two things worth carrying forward:

- **Tolerance calibration, not a bug**: the first test run used an arbitrary 5%
  weight-reconstruction-error bound, written before any real data existed. Real
  Phi-3 `down_proj` weights measured ~12% relative Frobenius error under this scheme
  (no calibration, no outlier handling, no NF4-style non-uniform levels -- plain
  symmetric INT4 really does lose this much). Corrected the test bound to 20%,
  grounded in the measured number rather than a guess.
- **Phase 5's Python kernel-benchmark wrapper got pulled forward**: validating the
  kernel against the reference needed *some* way to call it from Python, so
  `src/kernels/torch_binding.cpp` (a thin `torch.utils.cpp_extension` wrapper, JIT-compiled
  via `torch.utils.cpp_extension.load`) exists now instead of in Phase 5. The pure
  kernel (`dequant_gemm_naive.cu`/`.cuh`) has no torch dependency and never will --
  Phase 3's plugin links it directly.
- **Environment note**: `torch.utils.cpp_extension.load` needs `ninja` on `PATH`, not
  just importable -- installed into `.venv`, but since this project's commands run via
  `.venv/bin/python` directly rather than an activated venv, `PATH` needs
  `.venv/bin` prepended explicitly when running anything that JIT-compiles
  (`PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. .venv/bin/python -m pytest tests/`).

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

**Phase 2 complete.** Shape: `gate_up_proj` (M=1, N=16384, K=3072), the largest of the
four real Phi-3 layers, on the RTX 3080.

- **Profiling (step 1)**: `ncu` on the naive kernel showed 86% of memory sectors wasted
  to uncoalesced access (10.67 sectors/request against a 2-sector ideal for this access
  pattern) and only 38% SM throughput — clearly memory-access-pattern-bound, not
  compute-bound, so the optimization target was the load pattern, not more ALU work.
- **Optimization (step 2)**: rewrote as a warp-per-output-row kernel
  (`src/kernels/dequant_gemm_optimized.cu`) — activations staged once into shared
  memory per output row, packed weights read via vectorized `uint32_t` loads (4 bytes /
  8 weights per lane per iteration), partial sums combined with a warp-shuffle tree
  reduction. Validated bit-for-bit consistent with the naive kernel's formula (8 tests,
  `tests/test_dequant_gemm_optimized_kernel.py`, including a direct optimized-vs-naive
  comparison), then re-profiled: sectors/request dropped from 10.67 to 1.60, SM
  throughput rose from 38% to 81%. Benchmarked
  (`src/kernels/bench_naive.cu`/`bench_optimized.cu`, median of 50 runs): **0.7058 ms
  (naive) -> 0.1642 ms (optimized), a 4.3x speedup**, confirming the fix worked for the
  diagnosed reason rather than coincidentally.
- **cuBLAS baseline (step 3)**: the "obvious alternative" — unpack INT4 to a dense fp32
  weight matrix (`src/kernels/dequant_only.cu`, deliberately unoptimized) and call
  `cublasSgemv` — needed its own row-major-vs-column-major transpose derivation
  (row-major `[N,K]` storage read as column-major `[K,N]` = `W^T`, so `CUBLAS_OP_T` with
  `m=K, n=N, lda=K` recovers `W @ x`). Verified numerically against the optimized
  kernel's output before trusting any timing (`src/kernels/bench_cublas_baseline.cu`;
  `rel_err ~ 1e-6`, MATCH) — the project's standing "verify, don't assume" discipline
  paying off a third time, this time by *confirming* rather than catching a bug.
  Result: **0.7588 ms/call**, essentially tied with the naive fused kernel and **4.6x
  slower than the optimized fused kernel**. The dense-dequant intermediate (16384 x
  3072 x 4 bytes, ~201MB) round-trips through global memory once to be written and
  again for cuBLAS to read, and that extra traffic outweighs whatever cuBLAS gains from
  being a mature, tuned GEMM.
- **Hard stop (step 4)**: reporting this as the final Phase 2 result per requirements.md
  §6 — the fused optimized kernel beats both the naive fused kernel (4.3x) and the
  dequantize-then-vendor-GEMM baseline (4.6x). No further optimization attempted;
  moving on to Phase 3.
- **Full-suite regression caught before commit**: adding `dequant_gemm_optimized` to
  the shared `torch_binding.cpp` broke `test_dequant_gemm_naive_kernel.py`'s fixture,
  which still only compiled `torch_binding.cpp` against `dequant_gemm_naive.cu` —
  `launch_dequant_gemm_optimized` was an undefined symbol at link time. The optimized
  kernel's own test file happened to compile both `.cu` files and so never caught it.
  Fixed by adding `dequant_gemm_optimized.cu` to that fixture's sources too. A reminder
  to run the full suite, not just the new test file, after changing a file shared
  across test fixtures.

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

**Phase 3 complete.** `src/plugin/dequant_gemm_plugin.{h,cpp}` implements
`DequantGemmPluginV3` (`IPluginV3` + `IPluginV3OneCore`/`OneBuild`/`OneRuntime`, the
"one object, three capability interfaces" pattern the installed TensorRT 11.3.0.99
headers themselves define) and `DequantGemmPluginCreator` (`IPluginCreatorV3One`). Class
names, method signatures and the capability-split design were read directly out of
`/usr/include/x86_64-linux-gnu/NvInferRuntime.h` and `NvInferPluginBase.h` rather than
recalled from memory, given the plan's own Phase 0 note that this API has shifted across
TensorRT versions.

- **Weight-only plugin contract**: the packed INT4 weights, per-group scales, `N`, `K`
  and `group_size` are baked in as plugin attributes at network-build time (via
  `PluginFieldCollection`), not runtime tensor inputs — the only runtime input is the
  activation `x [M, K]`, the only output is `out [M, N]`. `enqueue()` is a thin call into
  the Phase 2 optimized kernel; no new kernel code was written.
- **Clone/serialization correctness**: `clone()` must produce an object whose
  `getFieldsToSerialize()` pointers reference *its own* members, not the source object's
  — a default-copy-constructor clone would leave the cloned `PluginFieldCollection`
  pointing at the original's now-possibly-destroyed `N`/`K`/`group_size` ints. Solved
  with a dedicated "cheap clone" constructor that shares the refcounted host/device
  weight buffers (`shared_ptr`, no re-upload) but always rebuilds its own
  `PluginFieldCollection` against its own members.
- **CMake pitfall**: `dequant_kernels` (the kernel static lib) initially had
  `CUDA_SEPARABLE_COMPILATION ON`, left over from assuming relocatable device code would
  be needed. It isn't — nothing here calls a `__device__` function across translation
  units — and turning it on broke the link of `validate_plugin_engine` (a pure-C++
  executable) with an undefined `__cudaRegisterLinkedBinary` reference, since CMake
  doesn't automatically add a device-link step for a plain executable consuming an
  `-rdc=true` static lib. Fixed by turning it off.
- **Validation (step 3)**: `python/reference/gen_plugin_test_vectors.py` generates
  input/weight/reference tensors using the same already-validated
  `quantize_groupwise_int4`/`quantized_linear_reference` functions every earlier phase
  was checked against (not a second, independent C++ implementation of quantization —
  the point is confirming the plugin wrapping preserves behavior, not re-deriving the
  math). `src/plugin/validate_plugin_engine.cpp` builds a real single-layer engine via
  the C++ builder API (`addPluginV3`), runs it, and diffs against those vectors. Passed
  on two different real Phi-3 shapes/batch sizes (`o_proj`, M=2: rel_err ~1e-6;
  `down_proj`, M=4: rel_err ~1e-6) — checked more than one shape deliberately, since a
  single passing case could hide a bug that only shows up when `N != K` or `M` changes.

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

**Phase 4 complete.** `src/harness/engine_harness.{h,cpp}` (`EngineHarness`) loads a
serialized engine plan, registers the plugin creator the deserializer needs to resolve
the `DequantGemmInt4` layer inside it, and hands out `IExecutionContext`s. Confirmed
Phase 0's destructor-API finding actually holds in working code, not just in the docs:
plain `std::unique_ptr<IRuntime>`/`unique_ptr<ICudaEngine>`/`unique_ptr<IExecutionContext>`
throughout, no custom deleter needed.

- **Build-once-deploy-many-times pipeline**: `validate_plugin_engine.cpp` (Phase 3) now
  takes an optional second argument and, only if its own PASS check succeeds, serializes
  the validated engine to a `.plan` file. The harness never rebuilds the network — it
  only ever loads bytes a validation step already vouched for, matching how a real
  deployment separates the build step (slow, done once) from serving (fast, done many
  times).
- **Correctness pass (step 2)**: `src/harness/run_single.cpp` loads the plan, runs one
  inference, diffs against the same `reference.bin` Phase 3 validated the in-process-built
  engine against. PASS, rel_err ~1e-6 — confirms serialize/deserialize round-tripped the
  plugin's baked-in weights correctly, not just that the in-process build path works.
- **Concurrency (step 3)**: `src/harness/run_concurrent.cpp`. Each worker owns its own
  `IExecutionContext`, `cudaStream_t`, and pinned host input/output buffers (`cudaMallocHost`
  — plain pageable memory can't participate in true async H2D/D2H overlap); a shared
  `ICudaEngine` is read-only and safe to hand out contexts from concurrently, but a single
  context is not safe for concurrent `enqueue` calls, so each thread gets its own. Workers
  submit their async H2D-copy/enqueue/async-D2H-copy sequence with no synchronization
  between requests, only a final `sync()` after all are submitted — real overlap, not
  "N threads calling the same function."
- **Timing (step 4)**: warmup(3) + median-of-50, individually-synchronized runs, matching
  both `evol_inference.fitness.measure_latency_ms`'s methodology and this project's own
  `bench_naive.cu`/`bench_optimized.cu` convention. Measured (shape M=2, N=3072, K=3072,
  the `o_proj` test vectors, on the RTX 3080):
  - Single-stream baseline: **0.0768 ms/request** (median of 50).
  - 4 concurrent streams x 50 requests: **17873.9 req/s** aggregate, a **1.37x** speedup
    over the fully-serial single-stream throughput.
  - 8 concurrent streams x 50 requests: **17633.7 req/s** aggregate, **1.35x** — no
    further gain over 4 streams.
  - **Honest read**: the overlap is real but small and saturates immediately. This
    workload is a single GEMV-shaped op with a ~24KB input and ~24KB output per
    request — H2D/D2H copy time is already a small fraction of the ~0.077ms round trip,
    so there's little copy-vs-compute overlap left to exploit, and going from 4 to 8
    streams gained nothing (likely fixed per-launch/scheduling overhead, not bandwidth or
    SM occupancy, is now the limit). A larger per-request workload (e.g. batched decode
    across many sequences) would likely show a bigger overlap benefit; not measured here
    since it would require a different engine shape than the one already validated.

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

**Phase 5 complete.**

- **Validation ladder (step 1)**: `tests/CMakeLists.txt` wires the plugin-level and
  engine-level rungs into real `ctest` cases (`gen_test_vectors` → `plugin_level_validation`
  → `engine_level_validation`, ordered via ctest fixtures since each depends on the last
  one's output — a shared vectors directory, then a serialized engine plan). All three
  pass. The kernel-level rung is deliberately left as the existing pytest suite rather
  than duplicated into ctest — it already needs the project venv's torch, and adding a
  second test runner for the same coverage would be process for its own sake, not more
  rigor.
- **pybind11 exposure (step 2)**: already satisfied, not new work — `src/kernels/torch_binding.cpp`
  (written in Phase 1, when the kernel first needed a way to be called from Python for
  testing) already exposes the standalone kernel via pybind11 under the hood of
  `torch.utils.cpp_extension.load`. No second wrapper was written.
- **Benchmark table (step 3)**: `python/benchmarks/benchmark_table.py`. All four rows run
  the identical weights and input (the same `build/plugin_test_vectors` the TensorRT
  engine was validated against, loaded directly rather than regenerated), and use the
  same warmup(3) + median-of-50 methodology throughout; the fourth row shells out to the
  already-built `run_concurrent` harness binary rather than re-implementing TensorRT
  invocation in Python, so there's one source of truth for that number, not two that
  could drift. Measured (shape M=2, N=3072, K=3072, group_size=128, RTX 3080):

  | Implementation | ms/request |
  |---|---|
  | PyTorch eager | 0.6778 |
  | `torch.compile` | 0.1647 |
  | Standalone CUDA kernel | 0.0759 |
  | TensorRT plugin engine (C++ harness) | 0.0768 |

  **Reading it honestly**: the TensorRT engine essentially ties the standalone kernel
  (within run-to-run noise) — confirming the plugin wrapping adds no measurable
  overhead, which is the result a thin wrapper *should* produce, not a given until
  measured. `torch.compile` gets partway there (4.1x over eager) by fusing the
  dequantize+matmul into fewer kernel launches, but doesn't reach the hand-written
  kernel's coalesced-access pattern. PyTorch eager is slowest for the same reason the
  cuBLAS baseline was in Phase 2: materializing a dense dequantized weight matrix in
  global memory before the matmul, rather than fusing dequantization into the GEMM
  itself, is the dominant cost at this shape and batch size — the same finding, showing
  up a second time through an independent measurement path.

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

- ~~TensorRT SDK acquisition path~~ — **resolved**: NVIDIA's public CUDA apt repo
  (`developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64`) carries
  `libnvinfer-dev` and friends with no login/EULA click-through needed — the anticipated
  manual-download fallback wasn't necessary. Installed `libnvinfer11`,
  `libnvinfer-plugin11`, `libnvinfer-dev`, `libnvinfer-headers-dev`,
  `libnvinfer-headers-plugin-dev`, `libnvinfer-plugin-dev`, all pinned to
  `11.3.0.99-1+cuda12.9` — the `+cuda12.9` build variant deliberately, to stay on the
  single already-verified `nvcc` 12.0 toolchain rather than pulling in a separate CUDA
  13.x toolkit just for TensorRT's linkage (the `+cuda13.4` variant apt initially
  preferred was rejected for this reason). Exact version match with the
  `tensorrt-cu13==11.3.0.99` pip wheel used for the Python side.
- ~~`IPluginV3` vs `IPluginV2DynamicExt`~~ — **resolved**: TensorRT 11.3.0.99 has both;
  targeting `IPluginV3` per the plan's stated preference.
- ~~TensorRT object destruction API~~ — **resolved**: `IBuilder` and friends have plain
  virtual destructors in this version (no legacy `->destroy()` pattern) — Phase 4's RAII
  wrappers can use plain `std::unique_ptr`, no custom deleter needed.
- Quantization group size (128 proposed, not fixed) — Phase 1.
- Draft model for speculative decoding — Phase 6.

Phase 0 is otherwise complete: `nvcc`-12.0-vs-driver-580 verified with a running kernel,
`import tensorrt` verified (11.3.0.99), a minimal C++ program compiles/links/runs
against the installed SDK (`nvinfer1::createInferBuilder` succeeds), and the CMake
skeleton (`src/kernels`, `src/plugin`, `src/harness`, `tests`, all with placeholder
`CMakeLists.txt` files) configures cleanly end to end.
